from re import A
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
        for mask, name_param in zip(masks(module), module.named_parameters(recurse=False)):
            pname, param = name_param
            if "bias" not in pname or bias is True:
                yield mask, param

def get_decomposition_matrices(module, pname):
    r"""Returns the 2 decomposition matrices where they're applied.
    """
    if "bias" in pname:
        return None, None
    elif isinstance(module, (layers.Linear, layers_vit.Linear, layers.Conv2d, layers_vit.Conv2d, layers_vit.MultiheadAttention)):
        return module.A, module.B
    else:
        return None, None

def masked_parameters_LoRAinspired(model, bias=False, batchnorm=False, residual=False):
    r"""Returns an iterator over models prunable parameters, yielding:
     - the mask tensor
     - the parameter tensors
     - the 2 decomposition matrices where they're applied
    """
    for module in filter(lambda p: prunable(p, batchnorm, residual), model.modules()):
        for mask, name_param in zip(masks(module), module.named_parameters(recurse=False)):
            pname, param = name_param            
            if "bias" not in pname or bias is True:
                A, B = get_decomposition_matrices(module, pname)
                yield mask, param, A, B