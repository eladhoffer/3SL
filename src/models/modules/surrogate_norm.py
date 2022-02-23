import torch
import torch.nn as nn
import logging
from torch import Tensor
from typing import List, Optional


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
                 track_running_stats=True, num_surrogates=1, dims=[0, 2, 3]):
        super(SNorm, self).__init__(num_features, eps, momentum, affine,
                                    track_running_stats)
        self.num_surrogates = num_surrogates
        self.dims = dims

    def forward(self, x):
        return surrogate_norm(x, self.running_mean, self.running_var, self.weight, self.bias,
                              self.training, self.momentum, self.eps, self.num_surrogates, self.dims)

    @classmethod
    def convert_snorm(cls, module, replaced_layers=[torch.nn.modules.batchnorm._BatchNorm], modify_momentum=None):
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
        return module_output


class SNorm2d(SNorm):

    def __init__(self, num_features, eps=1e-5, momentum=1., affine=True,
                 track_running_stats=True, num_surrogates=1, dims=[0, 2, 3]):
        super(SNorm2d, self).__init__(num_features, eps, momentum, affine,
                                      track_running_stats, dims)


if __name__ == '__main__':
    x = torch.randn(64, 256, 32, 32).cuda()
    rmrean = torch.zeros(256).cuda()
    rvar = torch.ones(256).cuda()
    out = surrogate_norm(x, rmrean, rvar)
