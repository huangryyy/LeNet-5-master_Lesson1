import torch
import torch.nn as nn
from torch.autograd.function import InplaceFunction, Function
from .minmaxobserver import MovingAverageMinMaxObserver
from .observerbase import _with_args
import re

__all__ = [
    'Fake_quantize_per_tensor',
    'Fake_quantize_per_channel',
    'FakeQuantize',
    'enable_fake_quant',
    'disable_fake_quant',
    'enable_observer',
    'disable_observer',
    'calc_qparam',
]

_EXPORT_MODE = False # FIX SOME PRECISION PROBLEM
def set_export_mode(flag=True):
    global _EXPORT_MODE
    _EXPORT_MODE = flag

class Fake_quantize_per_tensor(Function):
    """return a quantized and dequantized a float tensor"""
    @staticmethod
    def forward(ctx, X, scale, zero_point, qmin, qmax):
        with torch.no_grad():
            if _EXPORT_MODE:
                X = torch.round(X * (2**16)) * (0.5**16)
            Xq = torch.floor(X / scale + zero_point)
            Xq = torch.clip(Xq, qmin, qmax)
            Xqf = (Xq - zero_point) * scale
            Xqf.scale = scale
            Xqf.zero_point = zero_point
            # with open('D:\\ruanjian_zzzzzzz\\123\LeNet-5-master\checkpoints\ori.txt', 'a') as f:
            #     print(X,file=f)
            # with open('D:\\ruanjian_zzzzzzz\\123\LeNet-5-master\checkpoints\quant.txt', 'a') as f:
            #     print(Xq,file=f)
            # with open('D:\\ruanjian_zzzzzzz\\123\LeNet-5-master\checkpoints\dequant.txt', 'a') as f:
            #     print(Xqf,file=f)
            with open('D:\\ruanjian_zzzzzzz\\123\LeNet-5-master\checkpoints\scale.txt', 'a') as f:
                print(scale,file=f)
            return Xqf

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None, None

class FakeQuantize(nn.Module):
    def __init__(self, observer=MovingAverageMinMaxObserver, quantize_func=Fake_quantize_per_tensor, **observer_kwargs):
        super().__init__()
        self.register_buffer('fake_quant_enabled', torch.tensor([0], dtype=torch.uint8))
        self.register_buffer('observer_enabled', torch.tensor([0], dtype=torch.uint8))
        self.register_buffer('calc_qparam', torch.tensor([0], dtype=torch.uint8))
        self.observer = observer(**observer_kwargs)
        self.register_buffer('scale', torch.tensor([1.0]))
        self.register_buffer('zero_point', torch.tensor([0]))
        self.quantize_func = quantize_func

    @torch.jit.export
    def enable_fake_quant(self):
        self.fake_quant_enabled[0] = 1
        return self

    @torch.jit.export
    def disable_fake_quant(self):
        self.fake_quant_enabled[0] = 0
        return self

    @torch.jit.export
    def enable_observer(self):
        self.observer_enabled[0] = 1
        return self

    @torch.jit.export
    def disable_observer(self):
        self.observer_enabled[0] = 0
        return self
    
    @torch.jit.export
    def enable_calc_qparam(self):
        self.calc_qparam[0] = 1
        return self
    
    @torch.jit.export
    def disable_calc_qparam(self):
        self.calc_qparam[0] = 0
        return self

    @torch.jit.export
    def calculate_qparam(self, inplace=False):
        _scale, _zero_point = self.observer.calculate_qparam()
        _scale, _zero_point = _scale.to(self.scale.device), _zero_point.to(self.zero_point.device)
        if inplace:
            self.scale.resize_(_scale.shape)
            self.scale.copy_(_scale)
            self.zero_point.resize_(_zero_point.shape)
            self.zero_point.copy_(_zero_point)
        else:
            return _scale, _zero_point

    def forward(self, X, not_observe=False):
        if self.observer_enabled[0] == 1 and not not_observe:
            self.observer(X.detach())
        
        if self.calc_qparam[0] == 1 and not not_observe:
            self.calculate_qparam(inplace=True)

        if self.fake_quant_enabled[0] == 1:
            X = self.quantize_func.apply(
                X, self.scale, self.zero_point,
                self.observer.qmin, self.observer.qmax
            )
        return X

    with_args = classmethod(_with_args)

    @torch.jit.export
    def extra_repr(self):
        return 'fake_quant_enabled={}, observer_enabled={},\
            scale={}, zero_point={}'.format(
            self.fake_quant_enabled, self.observer_enabled,
            self.scale, self.zero_point)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        # We cannot currently register scalar values as buffers, so need to manually
        # specify serialization here.
        super(FakeQuantize, self)._save_to_state_dict(destination, prefix, keep_vars)
        destination[prefix + 'scale'] = self.scale
        destination[prefix + 'zero_point'] = self.zero_point

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Removing this function throws an error that the the size of the loaded tensor does not match the original size
        # i.e., These buffers start out with numel 0 and become numel 1 once they have their first forward pass.
        local_state = ['scale', 'zero_point']
        for name in local_state:
            key = prefix + name
            if key in state_dict:
                val = state_dict[key]
                setattr(self, name, val)
            elif strict:
                missing_keys.append(key)
        super(FakeQuantize, self)._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                                        missing_keys, unexpected_keys, error_msgs)

def toggle(module, quantize=None, observer=None, calc_qparam=None):
    if quantize is not None:
        if quantize:
            enable_fake_quant(module)
        else:
            disable_fake_quant(module)
    if observer is not None:
        if observer:
            enable_observer(module)
        else:
            disable_observer(module)
    if calc_qparam is not None:
        if calc_qparam:
            enable_calc_qparam(module)
        else:
            disable_calc_qparam(module)

def enable_fake_quant(module):
    for mod in module.modules():
        if hasattr(mod, 'enable_fake_quant'):
            mod.enable_fake_quant()

def disable_fake_quant(module):
    for mod in module.modules():
        if hasattr(mod, 'disable_fake_quant'):
            mod.disable_fake_quant()

def enable_observer(module, pattern=''):
    for name, mod in module.named_modules():
        if hasattr(mod, 'enable_observer') and re.search(pattern, name):
            mod.enable_observer()

def disable_observer(module):
    for mod in module.modules():
        if hasattr(mod, 'disable_observer'):
            mod.disable_observer()

def enable_calc_qparam(module):
    for mod in module.modules():
        if hasattr(mod, 'enable_calc_qparam'):
            mod.enable_calc_qparam()

def disable_calc_qparam(module):
    for mod in module.modules():
        if hasattr(mod, 'disable_calc_qparam'):
            mod.disable_calc_qparam()

def calc_qparam(module):
    for mod in module.modules():
        if isinstance(mod, FakeQuantize):
            mod.calculate_qparam(inplace=True)