import torch


class FrozenModule(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        for p in self.module.parameters():
            p.requires_grad_(False)

    def forward(self, *args, **kwargs):
        with torch.no_grad():
            return self.module(*args, **kwargs)


def remove_module(model, name):
    for n, m in model.named_children():
        if n == name:
            setattr(model, name, torch.nn.Identity())
        remove_module(m, name)
