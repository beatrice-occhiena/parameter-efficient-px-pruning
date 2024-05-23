from lib import layers
from lib import layers_vit

def masks(module):
    r"""Returns an iterator over modules masks, yielding the mask.
    """
    for name, buf in module.named_buffers():
        if "mask" in name:
            yield buf

def trainable(module):
    r"""Returns boolean whether a module is trainable.
    """
    return not isinstance(module, (layers.Identity1d, layers.Identity2d))

def prunable(module, batchnorm, residual, layernorm=True, embedding=False):
    r"""Returns boolean whether a module is prunable.
    """
    # Original maskable layer extension
    isprunable = isinstance(module, (layers.Linear, layers.Conv2d))
    if batchnorm:
        isprunable |= isinstance(module, (layers.BatchNorm1d, layers.BatchNorm2d))
    if residual:
        isprunable |= isinstance(module, (layers.Identity1d, layers.Identity2d))

    # Maskable layer extension for ViT models
    isprunable |= isinstance(module, (layers_vit.Linear, layers_vit.Conv2d, layers_vit.MultiheadAttention))
    if layernorm:
        isprunable |= isinstance(module, (layers_vit.LayerNorm))
    if embedding:
        isprunable |= isinstance(module, (layers_vit.Embedding))
        
    return isprunable

def parameters(model):
    r"""Returns an iterator over models trainable parameters, yielding just the
    parameter tensor.
    """
    for module in filter(lambda p: trainable(p), model.modules()):
        for param in module.parameters(recurse=False):
            yield param

def masked_parameters(model, bias=False, batchnorm=False, residual=False):
    r"""Returns an iterator over models prunable parameters, yielding both the
    mask and parameter tensors.
    """
    for module in filter(lambda p: prunable(p, batchnorm, residual), model.modules()):
        for mask, name_param in zip(masks(module), module.named_parameters(recurse=False)): #TODO:CHECK removed recurse=False
            name, param = name_param
            if "bias" not in name or bias is True:
                #print(name, param.shape, mask.shape)
                yield mask, param