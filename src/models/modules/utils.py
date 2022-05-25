import torch


def freeze_model_(model):
    for p in model.parameters():
        p.requires_grad_(False)


class FrozenModule(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        freeze_model_(self.module)

    def forward(self, *args, **kwargs):
        with torch.no_grad():
            return self.module(*args, **kwargs)


def replace_module(model, name, new_module):
    for n, m in model.named_children():
        if n == name:
            setattr(model, name, new_module)
        remove_module(m, name)


def remove_module(model, name):
    replace_module(model, name, torch.nn.Identity())
