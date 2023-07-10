import torch
import torch.nn as nn
import logging
from torch import Tensor
from typing import List, Optional
from torch.distributions.chi2 import Chi2


@torch.jit.script
def surrogate_norm(x: Tensor, running_mean: Tensor, running_var: Tensor,
                   weight: Optional[Tensor] = None, bias: Optional[Tensor] = None,
                   training: bool = True, momentum: float = 0.1, eps: float = 1e-5,
                   num_surrogates: int = 1, dims: Optional[List[int]] = None):
    if dims is None:
        dims = [0, 2, 3]
    if len(dims) == 0:
        dims = [0, 2, 3]
    if training:
        surr = x.narrow(0, 0, num_surrogates)
        surr = surr.float()  # avoid numerical issues for half training
        var, mean = torch.var_mean(surr, dims,
                                   unbiased=False,
                                   keepdim=True)
        running_mean.lerp_(mean.data.flatten(), momentum)
        running_var.lerp_(var.data.flatten(), momentum)
    else:
        mean = running_mean.view(1, -1, 1, 1)
        var = running_var.view(1, -1, 1, 1)
    scale = (var + eps).rsqrt()
    if weight is not None:
        scale = scale * weight.view_as(var)
    shift = -mean * scale
    if bias is not None:
        shift = shift + bias.view_as(mean)
    scale = scale.to(dtype=x.dtype)
    shift = shift.to(dtype=x.dtype)
    x = x * scale + shift
    return x


class SNorm(torch.nn.modules.batchnorm._BatchNorm):
    """BatchNorm with mean-only normalization"""

    def __init__(self, num_features, eps=1e-5, momentum=1., affine=True,
                 track_running_stats=True, num_surrogates=1, dims=[0, 2, 3],
                 inject_noise=False, ghost_batch_size=-1):
        super(SNorm, self).__init__(num_features, eps, momentum, affine,
                                    track_running_stats)
        self.num_surrogates = num_surrogates
        self.dims = dims
        self.inject_noise = inject_noise
        self.ghost_batch_size = ghost_batch_size

    def forward(self, x):
        weight = self.weight
        bias = self.bias
        if self.training and self.inject_noise:
            with torch.no_grad():
                # var, mean = torch.var_mean(x[:self.ghost_batch_size], self.dims)
                N = self.ghost_batch_size
                noise_mul = (Chi2(torch.empty_like(weight)).sample() / N).rsqrt()
                noise_add = torch.empty_like(weight).normal_().mul(1 / N)
            weight = weight * noise_mul
            bias = bias - weight * noise_add

        return surrogate_norm(x, self.running_mean, self.running_var, weight, bias,
                              self.training, self.momentum, self.eps, self.num_surrogates, self.dims)

    @classmethod
    def convert_snorm(cls, module, replaced_layers=[torch.nn.modules.batchnorm._BatchNorm],
                      add_surrogate=False, modify_momentum=None):
        r"""Helper function to convert normalization layers in the model to
        `SNorm` layer.

        Args:
            module (nn.Module): containing module
            replaced_layers (optional): layer types to replace

        Returns:
            The original module with the converted `SNorm` layer

        Example::

            >>> # Network with nn.BatchNorm layer
            >>> module = torch.nn.Sequential(
            >>>            torch.nn.Linear(20, 100),
            >>>            torch.nn.BatchNorm1d(100)
            >>>          ).cuda()
            >>> snorm_module = convert_snorm(module)

        """
        module_output = module
        if any([isinstance(module, rep) for rep in replaced_layers]):
            if modify_momentum is None:
                momentum = module.momentum
            else:
                momentum = modify_momentum
            module_output = SNorm(module.num_features,
                                  module.eps, momentum,
                                  module.affine,
                                  module.track_running_stats
                                  )
            if module.affine:
                module_output.weight.data = module.weight.data.clone(
                    memory_format=torch.preserve_format).detach()
                module_output.bias.data = module.bias.data.clone(
                    memory_format=torch.preserve_format).detach()
                # keep requires_grad unchanged
                module_output.weight.requires_grad = module.weight.requires_grad
                module_output.bias.requires_grad = module.bias.requires_grad
            module_output.running_mean = module.running_mean
            module_output.running_var = module.running_var
            module_output.num_batches_tracked = module.num_batches_tracked
        for name, child in module.named_children():
            module_output.add_module(
                name, cls.convert_snorm(child, replaced_layers, modify_momentum))
        del module
        if add_surrogate:
            module_output = nn.Sequential(AddSurrogate(), module_output, RemoveSurrogate())
        return module_output


class SNorm2d(SNorm):

    def __init__(self, num_features, eps=1e-5, momentum=1., affine=True,
                 track_running_stats=True, num_surrogates=1, dims=[0, 2, 3]):
        super(SNorm2d, self).__init__(num_features, eps, momentum, affine,
                                      track_running_stats, dims)


class AddSurrogate(nn.Module):
    def __init__(self, dim=0, num_surrogates=1):
        super(AddSurrogate, self).__init__()
        self.dim = dim
        self.num_surrogates = num_surrogates

    def forward(self, x):
        with torch.no_grad():
            surr = torch.empty_like(x.narrow(self.dim, 0, self.num_surrogates))
            surr.normal_()
        x = torch.cat([surr, x], dim=self.dim)
        return x


class RemoveSurrogate(nn.Module):
    def __init__(self, dim=0, num_surrogates=1):
        super(RemoveSurrogate, self).__init__()
        self.dim = dim
        self.num_surrogates = num_surrogates

    def forward(self, x):
        x = x.narrow(self.dim, self.num_surrogates, x.size(self.dim) - self.num_surrogates)
        return x


if __name__ == '__main__':
    from torchvision.models import resnet18
    x = torch.randn(64, 3, 32, 32).cuda()
    model = resnet18().cuda()
    model = SNorm.convert_snorm(model ,add_surrogate=True)
    print(model)
    y = model(x)
    # rmrean = torch.zeros(256).cuda()
    # rvar = torch.ones(256).cuda()
    # out = surrogate_norm(x, rmrean, rvar)
