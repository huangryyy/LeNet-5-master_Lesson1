
import torch
import torch.nn as nn
import torch.nn.functional as F

class QBatchMatMul(nn.Module):
    def __init__(self, qconfig):
        super().__init__()
        self.qconfig = qconfig
        self.act_quant = qconfig.activation()
    
    def forward(self, x, y):
        return self.act_quant(
            torch.bmm(x, y)
        )
    
    @classmethod
    def from_float(cls, mod):
        return cls(mod.qconfig)
    