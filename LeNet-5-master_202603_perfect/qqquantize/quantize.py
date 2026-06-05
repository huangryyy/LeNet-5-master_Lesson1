import copy, re
import torch
import torch.nn as nn

import qqquantize.qmodules as qm
from qqquantize.qmodules import InputStub

DEFAULT_QAT_MODULE_MAPPING = {
    nn.Linear: qm.QLinear,
    nn.Conv2d: qm.QConv2d,
    nn.Conv1d: qm.QConv1d,
    qm.InputStub: qm.QStub,
}
"""
    nn.Softmax: qm.QSoft_ibert,
    nn.BatchNorm2d: qm.QBatchNorm2d,
    nn.BatchNorm1d: qm.QBatchNorm1d,
    
"""
class ModelConverter:
    def __init__(self, qconfig, mapping=None, pattern='', extra_attr=None):
        self.qconfig = qconfig
        self.mapping = mapping if mapping is not None else DEFAULT_QAT_MODULE_MAPPING
        self.pattern = pattern # used by propaget_qconfig
        assert extra_attr is None or isinstance(extra_attr, list)
        self.extra_attr = extra_attr # used by swapmodule
    
    def __call__(self, model):
        device = list(get_unique_devices_(model))[0]
        self._propagate_qconfig(model)
        self._convert(model, device)
        return model.to(device)
    
    def _propagate_qconfig(self, model):
        r"""Propagate qconfig through the module hierarchy and assign `qconfig`
        attribute on each leaf module
        """
        add_cfg_lst =  list(self.mapping.keys()) + [InputStub]
        for name, mod in model.named_modules():
            if any([isinstance(mod, valid_type) for valid_type in add_cfg_lst]):
                if re.search(self.pattern, name):
                    mod.qconfig = self.qconfig        
    
    def _convert(self, model, device):
        reassign = {}

        swappable_modules = list(self.mapping.keys())

        for name, mod in model.named_children():
            if type(mod) not in swappable_modules or not hasattr(mod, 'qconfig'):
                self._convert(mod, device)
            else:
                reassign[name] = swap_module(mod, self.mapping, self.extra_attr).to(device)

        for key, value in reassign.items():
            model._modules[key] = value

        return model

def swap_module(mod, mapping, extra_attr=None):
    r"""Swaps the module if it has a quantized counterpart and it has an
    `observer` attached.
    Args:
        mod: input module
        mapping: a dictionary that maps from nn module to nnq module
    Return:
        The corresponding quantized module of `mod`
    """
    new_mod = mod
    # Always replace dequantstub with dequantize
    if hasattr(mod, 'qconfig') and mod.qconfig is not None:
        if type(mod) in mapping:
            # respect device affinity when swapping modules
            devices = get_unique_devices_(mod)
            assert len(devices) <= 1, (
                "swap_module only works with cpu or single-device CUDA modules, "
                "but got devices {}".format(devices)
            )
            new_mod = mapping[type(mod)].from_float(mod)
        if extra_attr is not None:
            for attr in extra_attr:
                if hasattr(mod, attr):
                    new_mod.__setattr__(attr, mod.__dict__['f'])
    return new_mod


def get_unique_devices_(module):
    return {p.device for p in module.parameters()} | \
        {p.device for p in module.buffers()}


"""fuse conv and bn. Linear also supported"""
def fuse_conv_bn(conv, bn):
    bn_st_dict = bn.state_dict()
    conv_st_dict = conv.state_dict()

    # BatchNorm params
    eps = bn.eps
    running_mean = bn_st_dict['running_mean']
    running_var = bn_st_dict['running_var']
    gamma = bn_st_dict['weight']
    if 'bias' in bn_st_dict:
        beta = bn_st_dict['bias']
    else:
        beta = torch.zeros(gamma.size(0)).float().to(gamma.device)

    weight = conv_st_dict['weight']
    if 'bias' in conv_st_dict:
        bias = conv_st_dict['bias']
    else:
        bias = torch.zeros(weight.shape[0]).float().to(gamma.device)

    var_sqrt = torch.sqrt(running_var + eps)
    weight = weight * (gamma / var_sqrt).reshape([weight.shape[0], 1, 1, 1])
    bias = (bias - running_mean) / var_sqrt * gamma + beta

    fused_conv = copy.deepcopy(conv)
    fused_conv.weight.data.copy_(weight)
    if fused_conv.bias is None:
        fused_conv.bias = torch.nn.Parameter(bias)
    else:
        fused_conv.bias.data.copy_(bias)
    return fused_conv