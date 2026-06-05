from collections import OrderedDict
from . import observers

class _Hook:
    def __init__(self, name:str):
        self.name = name
        self.data = []
    
    def __call__(self, module, inp, out):
        d = dict()
        d['_module_type'] = type(module).__name__
        if isinstance(out, tuple):
            d['values'] = tuple(x.detach().cpu().numpy() for x in out)
        else:
            d['values'] = out.detach().cpu().numpy()
        for buf_name, buf in module.named_buffers():
            d[buf_name] = buf.detach().cpu().numpy()
        self.data.append(d)

class SaveIntermediateHook:
    def __init__(self):
        self.hooks = {}
    
    def get_hook(self, name):
        self.hooks[name] = _Hook(name)
        return self.hooks[name]

    def output_data(self):
        OUTPUT = OrderedDict()
        for name, hook in self.hooks.items():
            OUTPUT[name] = hook.data
        return OUTPUT
    
    def reset(self):
        for name, hook in self.hooks.items():
            hook.data.clear()

DEFAULT_INTERMEDIATE_HOOKS_WHITE_LIST = {
    observers.fake_quantize.FakeQuantize
}

def register_intermediate_hooks(model, white_lst=None):
    """Args: 
    white_lst, black_lst: module type
    """
    hook = SaveIntermediateHook()
    if white_lst is None:
        white_lst = DEFAULT_INTERMEDIATE_HOOKS_WHITE_LIST
    for name, mod in model.named_modules():
        if any([isinstance(mod, wtype) for wtype in white_lst]):
            mod.register_forward_hook(hook.get_hook(name))
    return hook
        