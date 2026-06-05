import torch
import torch.nn as nn
import torch.nn.functional as F


from torch.nn.modules.utils import _single

__all__ = ['QConv2d']

class QConv2d(nn.Conv2d):
    _FLOAT_MODULE = nn.Conv2d

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', qconfig=None):
        super().__init__(in_channels, out_channels, kernel_size,
                                     stride=stride, padding=padding, dilation=dilation,
                                     groups=groups, bias=bias, padding_mode=padding_mode)
        assert qconfig, 'qconfig must be provided for QAT module'
        self.qconfig = qconfig
        self.act_quant = qconfig.activation()
        self.weight_quant = qconfig.weight()
        if bias:
            self.bias_quant = qconfig.bias()

    def forward(self, input):
        weight = self.weight_quant(self.weight)
        res=self._conv_forward(input, weight, self.bias)
        return self.act_quant(res)

    @classmethod
    def from_float(cls, mod):
        r"""Create a qat module from a float module or qparams_dict

            Args: `mod` a float module, either produced by torch.quantization utilities
            or directly from user
        """
        assert type(mod) == cls._FLOAT_MODULE, 'qat.' + cls.__name__ + '.from_float only works for ' + \
            cls._FLOAT_MODULE.__name__
        qconfig = mod.qconfig
        qat_conv = cls(mod.in_channels, mod.out_channels, mod.kernel_size,
                       stride=mod.stride, padding=mod.padding, dilation=mod.dilation,
                       groups=mod.groups, bias=mod.bias is not None,
                       padding_mode=mod.padding_mode, qconfig=qconfig)
        with torch.no_grad():
            qat_conv.weight.copy_(mod.weight)
            if mod.bias is not None:
                qat_conv.bias.copy_(mod.bias)
        return qat_conv

class QConv1d(nn.Conv1d):
    _FLOAT_MODULE = nn.Conv1d
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size, stride, padding,
        dilation=1, groups=1, bias=True, padding_mode='zeros', qconfig=None
    ):
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding,
            dilation, groups, bias, padding_mode)
        
        assert qconfig, 'qconfig must be provided for QAT module'
        self.qconfig = qconfig
        self.act_quant = qconfig.activation()
        self.weight_quant = qconfig.weight()

    def forward(self, input):
        weight = self.weight_quant(self.weight)
        bias = self.bias
        if bias is not None:
            bias = self.act_quant(bias, not_observe=True)
            
        return self.act_quant(
            self._conv_forward(input, weight, bias)
        )
    
    @classmethod
    def from_float(cls, mod):
        r"""Create a qat module from a float module or qparams_dict

            Args: `mod` a float module, either produced by torch.quantization utilities
            or directly from user
        """
        assert type(mod) == cls._FLOAT_MODULE, 'qat.' + cls.__name__ + '.from_float only works for ' + \
            cls._FLOAT_MODULE.__name__
        qconfig = mod.qconfig
        qat_conv = cls(mod.in_channels, mod.out_channels, mod.kernel_size,
                       stride=mod.stride, padding=mod.padding, dilation=mod.dilation,
                       groups=mod.groups, bias=mod.bias is not None,
                       padding_mode=mod.padding_mode, qconfig=qconfig)
        with torch.no_grad():
            qat_conv.weight.copy_(mod.weight)
            if mod.bias is not None:
                qat_conv.bias.copy_(mod.bias)
        return qat_conv