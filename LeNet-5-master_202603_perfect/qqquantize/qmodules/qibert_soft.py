import torch
import torch.nn as nn
import torch.nn.functional as F
import math
__all__ = ['QSoft_ibert']
class QSoft_ibert(nn.Softmax):
    _FLOAT_MODULE = nn.Softmax
    def __init__(self, dim=None, qconfig=None):
        super().__init__(dim)
        assert qconfig, 'qconfig must be provided for QAT module'
        self.qconfig = qconfig
        self.act_quant = qconfig.activation()


    def forward(self, x):
        t_ln=1.4375
        ln2 =0.6875
        max,_ = torch.max(x,1)
        exp = max.reshape(x.shape[0],1)-x
        e = (exp*t_ln).floor()
        exp=e*ln2-exp
        # 0.3585  0.359375  0.3562  0.356201171875
        # 1.353  1.3515625  0.9625  0.96240234375
        # 0.344 0.34375  0.9979  0.998046875
        en = 0.359375*(exp+1.3515625)*(exp+1.3515625)+0.34375
        res = en*(0.5**e)
        en = res/res.sum(1).reshape(x.shape[0],1)


        return self.act_quant(en)

    
    @classmethod
    def from_float(cls, mod):
        assert type(mod) == cls._FLOAT_MODULE, 'qat.' + cls.__name__ + '.from_float only works for ' + \
            cls._FLOAT_MODULE.__name__
        qconfig = mod.qconfig
        qmod = cls(mod.dim, qconfig)
        return qmod