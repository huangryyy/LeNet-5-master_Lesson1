import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.modules.batchnorm import _BatchNorm

class QBatchNormBase(_BatchNorm):
    _FLOAT_MODULE = None
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, qconfig=None):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        assert qconfig, 'qconfig must be provided for QAT module'
        self.qconfig = qconfig
        self.weight_quant = qconfig.weight()
        self.act_quant = qconfig.activation()

    def forward(self, input):
        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:  # type: ignore
                self.num_batches_tracked = self.num_batches_tracked + 1  # type: ignore
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        assert self.running_mean is None or isinstance(self.running_mean, torch.Tensor)
        assert self.running_var is None or isinstance(self.running_var, torch.Tensor)
        F.batch_norm(
            input,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean if not self.training or self.track_running_stats else None,
            self.running_var if not self.training or self.track_running_stats else None,
            self.weight, self.bias, bn_training, exponential_average_factor, self.eps
        )
        
        assert len(input.shape) in [2,3,4]
        if len(input.shape) == 2:
            t = input
        elif len(input.shape) == 3:
            _c = input.shape[1]
            t = input.permute([0, 2, 1]).reshape([-1, _c])
        elif len(input.shape) == 4:
            _c = input.shape[1]
            t = input.permute([0, 2, 3, 1]).reshape([-1, _c])

        var, mean = torch.var_mean(t, unbiased=True, dim=0)
        var_sqrt = torch.sqrt(var)
        gamma = self.weight
        beta = self.bias

        weight = gamma / var_sqrt
        bias = - mean / var_sqrt * gamma + beta
        
        shape = [1] * len(input.shape)
        shape[1] = -1
        w = self.weight_quant(weight).reshape(shape)
        b = self.act_quant(bias, not_observe=True).reshape(shape)
        return self.act_quant(
            input * w + b
        )
    
    @classmethod
    def from_float(cls, mod):
        assert type(mod) == cls._FLOAT_MODULE, 'qat.' + cls.__name__ + '.from_float only works for ' + \
            cls._FLOAT_MODULE.__name__
        qconfig = mod.qconfig
        qat_bn = cls(mod.num_features, mod.eps, mod.momentum, mod.affine, mod.track_running_stats, qconfig)
        with torch.no_grad():
            qat_bn.weight.copy_(mod.weight)
            qat_bn.bias.copy_(mod.bias)
            qat_bn.running_mean.copy_(mod.running_mean)
            qat_bn.running_var.copy_(mod.running_var)
            qat_bn.num_batches_tracked.copy_(mod.num_batches_tracked)
        
        return qat_bn

class QBatchNorm2d(QBatchNormBase):
    _FLOAT_MODULE = nn.BatchNorm2d
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

class QBatchNorm1d(QBatchNormBase):
    _FLOAT_MODULE = nn.BatchNorm1d
    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'
                             .format(input.dim()))