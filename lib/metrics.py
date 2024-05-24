import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import lib.layers as layers
import lib.layers_vit as layers_vit
from open_clip.transformer import VisionTransformer

def summary(model, scores, flops, prunable):
    r"""Summary of compression results for a model.
    """
    rows = []
    for name, module in model.named_modules():
        for pname, param in module.named_parameters(recurse=False):
            pruned = prunable(module) and id(param) in scores.keys()
            if pruned:
                sparsity = getattr(module, pname+'_mask').detach().cpu().numpy().mean()
                score = scores[id(param)].detach().cpu().numpy()
            else:
                sparsity = 1.0
                score = np.zeros(1)
            shape = param.detach().cpu().numpy().shape
            flop = flops[name][pname]
            score_mean = score.mean()
            score_var = score.var()
            score_sum = score.sum()
            score_abs_mean = np.abs(score).mean()
            score_abs_var  = np.abs(score).var()
            score_abs_sum  = np.abs(score).sum()
            rows.append([name, pname, sparsity, np.prod(shape), shape, flop,
                         score_mean, score_var, score_sum, 
                         score_abs_mean, score_abs_var, score_abs_sum, 
                         pruned])

    columns = ['module', 'param', 'sparsity', 'size', 'shape', 'flops', 'score mean', 'score variance', 
               'score sum', 'score abs mean', 'score abs variance', 'score abs sum', 'prunable']
    return pd.DataFrame(rows, columns=columns)

def flop(model, input_shape, device):

    total = {}
    def count_flops(name):
        def hook(module, input, output):
            flops = {}
            if isinstance(module, (layers.Linear, layers_vit.Linear, nn.Linear)):
                in_features = module.in_features
                out_features = module.out_features
                flops['weight'] = in_features * out_features
                if module.bias is not None:
                    flops['bias'] = out_features
            if isinstance(module, (layers.Conv2d, layers_vit.Conv2d, nn.Conv2d)):
                in_channels = module.in_channels
                out_channels = module.out_channels
                kernel_size = int(np.prod(module.kernel_size))
                output_size = output.size(2) * output.size(3)
                flops['weight'] = in_channels * out_channels * kernel_size * output_size
                if module.bias is not None:
                    flops['bias'] = out_channels * output_size
            if isinstance(module, (VisionTransformer)):
                hidden_dim = module.width
                num_tokens = module.grid_size[0] * module.grid_size[1] + 1
                flops['class_embedding'] = hidden_dim # TODO:CHECK concat or 0?
                flops['positional_embedding'] = num_tokens * hidden_dim # added to intput
                out_dim = module.output_dim
                flops['proj'] = hidden_dim * out_dim # final linear projection
            if isinstance(module, (layers_vit.MultiheadAttention, nn.MultiheadAttention())):
                embed_dim = module.embed_dim
                flops['in_proj_weight'] = 3 * (embed_dim * embed_dim)  # Query, Key, Value linear projections
                flops['in_proj_bias'] = 3 * embed_dim
                # TODO:CHECK to be added or not?
                #num_heads = module.num_heads
                #head_dim = module.head_dim
                #attn_scores_flops = num_heads * (head_dim * head_dim)  # Attention score calculations
                #attn_weighted_sum_flops = num_heads * (head_dim * head_dim)  # Weighted sum of values
                #flops['in_proj_weight'] += attn_scores_flops + attn_weighted_sum_flops
            if isinstance(module, (layers_vit.LayerNorm, nn.LayerNorm)):
                normalized_shape = torch.prod(module.normalized_shape).item() # Number of elements on which normaliz is applied
                if module.elementwise_affine:
                    flops['weight'] = normalized_shape  # Scale
                    flops['bias'] = normalized_shape  # Shift
            if isinstance(module, (layers.BatchNorm1d, nn.BatchNorm1d)):
                if module.affine:
                    flops['weight'] = module.num_features
                    flops['bias'] = module.num_features
            if isinstance(module, (layers.BatchNorm2d, nn.BatchNorm2d)):
                output_size = output.size(2) * output.size(3)
                if module.affine:
                    flops['weight'] = module.num_features * output_size
                    flops['bias'] = module.num_features * output_size
            if isinstance(module, layers.Identity1d):
                flops['weight'] = module.num_features
            if isinstance(module, layers.Identity2d):
                output_size = output.size(2) * output.size(3)
                flops['weight'] = module.num_features * output_size
            total[name] = flops
        return hook
    
    for name, module in model.named_modules():
        module.register_forward_hook(count_flops(name))

    input = torch.ones([1] + list(input_shape)).to(device)
    model(input)

    return total

#### Segmentation Meters
# Adapted from: https://github.com/VainF/DeepLabV3Plus-Pytorch
import numpy as np
from sklearn.metrics import confusion_matrix

class _StreamMetrics(object):
    def __init__(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def update(self, gt, pred):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def get_results(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def to_str(self, metrics):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def reset(self):
        """ Overridden by subclasses """
        raise NotImplementedError()      

class StreamSegMetrics(_StreamMetrics):
    """
    Stream Metrics for Semantic Segmentation Task
    """
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist( lt.flatten(), lp.flatten() )
    
    @staticmethod
    def to_str(results):
        string = "\n"
        for k, v in results.items():
            if k!="Class IoU":
                string += "%s: %f\n"%(k, v)
        
        #string+='Class IoU:\n'
        #for k, v in results['Class IoU'].items():
        #    string += "\tclass %d: %f\n"%(k, v)
        return string

    def _fast_hist(self, label_true, label_pred):
        mask = (label_true >= 0) & (label_true < self.n_classes)
        hist = np.bincount(
            self.n_classes * label_true[mask].astype(int) + label_pred[mask],
            minlength=self.n_classes ** 2,
        )[:self.n_classes*self.n_classes]
        hist = hist.reshape(self.n_classes, self.n_classes)
        return hist

    def get_results(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))

        return {
                "Overall Acc": acc,
                "Mean Acc": acc_cls,
                "FreqW Acc": fwavacc,
                "Mean IoU": mean_iu,
                "Class IoU": cls_iu,
            }
        
    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))

class AverageMeter(object):
    """Computes average values"""
    def __init__(self):
        self.book = dict()

    def reset_all(self):
        self.book.clear()
    
    def reset(self, id):
        item = self.book.get(id, None)
        if item is not None:
            item[0] = 0
            item[1] = 0

    def update(self, id, val):
        record = self.book.get(id, None)
        if record is None:
            self.book[id] = [val, 1]
        else:
            record[0]+=val
            record[1]+=1

    def get_results(self, id):
        record = self.book.get(id, None)
        assert record is not None
        return record[0] / record[1]