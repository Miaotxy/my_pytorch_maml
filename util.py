import sys
import torch.nn as nn
from collections import OrderedDict

_internal_attrs = {'_backend', '_parameters', '_buffers', '_backward_hooks', '_forward_hooks', '_forward_pre_hooks', '_modules'}


class Scope(object):
    def __init__(self):
        self._modules = OrderedDict()


def _make_functional(module, params_box, params_offset):
    self = Scope()
    num_params = len(module._parameters)
    param_names = list(module._parameters.keys())
    forward = type(module).forward
    if isinstance(module, nn.Conv2d):
        setattr(self, "conv2d_forward", module.conv2d_forward)
    if isinstance(module, nn.BatchNorm2d):
        setattr(self, "_check_input_dim", module._check_input_dim)
        setattr(self, "num_batches_tracked", module.num_batches_tracked)
        setattr(self, "running_mean", module.running_mean)
        setattr(self, "running_var", module.running_var)

    for name, attr in module.__dict__.items():
        if name in _internal_attrs:
            continue
        setattr(self, name, attr)

    child_params_offset = params_offset + num_params
    for name, child in module.named_children():
        child_params_offset, fchild = _make_functional(child, params_box, child_params_offset)
        self._modules[name] = fchild
        setattr(self, name, fchild)

    def fmodule(*args, **kwargs):
        for name, param in zip(param_names, params_box[0][params_offset:params_offset + num_params]):
            setattr(self, name, param)
        return forward(self, *args, **kwargs)

    return child_params_offset, fmodule


def make_functional(module):
    params_box = [None]
    _, fmodule_internal = _make_functional(module, params_box, 0)

    def fmodule(*args, **kwargs):
        params_box[0] = kwargs.pop('params')
        return fmodule_internal(*args, **kwargs)

    return fmodule
