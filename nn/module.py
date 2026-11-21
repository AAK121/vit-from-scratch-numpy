
from nn.backend import to_device, xp


class Module:
    def __init__(self):
        self._params = {}
        self._grads = {}
        self._children = {}

    def add_param(self, name, array):
        array = to_device(array)
        self._params[name] = array
        self._grads[name] = xp.zeros_like(array)

    def add_child(self, name, module):
        self._children[name] = module
        setattr(self, name, module)
        return module

    def set_grad(self, name, value):
        self._grads[name] = value

    def named_parameters(self, prefix=""):
        items = [(f"{prefix}{k}", v) for k, v in self._params.items()]
        for cname, child in self._children.items():
            items.extend(child.named_parameters(prefix=f"{prefix}{cname}."))
        return items

    def named_grads(self, prefix=""):
        items = [(f"{prefix}{k}", v) for k, v in self._grads.items()]
        for cname, child in self._children.items():
            items.extend(child.named_grads(prefix=f"{prefix}{cname}."))
        return items

    def num_parameters(self):
        return sum(p.size for _, p in self.named_parameters())

    def zero_grad(self):
        for _, g in self.named_grads():
            g[...] = 0

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def backward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
