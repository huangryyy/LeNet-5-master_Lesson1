import torch
import torch.nn as nn
import torch.nn.functional as F
import math
__all__ = ['QSoft']
class QSoft(nn.Softmax):
    _FLOAT_MODULE = nn.Softmax
    def __init__(self, dim=None, qconfig=None):
        super().__init__(dim)
        assert qconfig, 'qconfig must be provided for QAT module'
        self.qconfig = qconfig
        self.act_quant = qconfig.activation()


    def forward(self, x):
        b = 0.96875
        ln2=0.6875
        log2e=1.4375
        max,_ = torch.max(x,1)
        exp = (max.reshape(x.shape[0],1)-x)*log2e
        e = exp.floor()
        exp=exp-e
        
        en = (b-(0.5*exp))*(0.5**e)
        F=en.sum(1)
        w = torch.log2(F).floor()
        m = F/(2**w)
        k = w*ln2+m-1
        exp=log2e*((k+max).reshape(x.shape[0],1)-x)
        e = exp.floor()
        exp=exp-e
        
        en = (b-(0.5*exp))*(0.5**e)


        return self.act_quant(en)

    
    @classmethod
    def from_float(cls, mod):
        assert type(mod) == cls._FLOAT_MODULE, 'qat.' + cls.__name__ + '.from_float only works for ' + \
            cls._FLOAT_MODULE.__name__
        qconfig = mod.qconfig
        qmod = cls(mod.dim, qconfig)
        return qmod

