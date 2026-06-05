import torch
import torch.nn as nn
import torch.nn.functional as F

class QSoftmax(nn.Softmax):
    _FLOAT_MODULE = nn.Softmax
    def __init__(self, dim=None, qconfig=None):
        super().__init__(dim)
        assert qconfig, 'qconfig must be provided for QAT module'
        self.qconfig = qconfig
        self.act_quant = qconfig.activation()
    
    def forward(self, x):
        return self.act_quant(super().forward(x))

    @classmethod
    def from_float(cls, mod):
        assert type(mod) == cls._FLOAT_MODULE, 'qat.' + cls.__name__ + '.from_float only works for ' + \
            cls._FLOAT_MODULE.__name__
        qconfig = mod.qconfig
        qmod = cls(mod.dim, qconfig)
        return qmod

