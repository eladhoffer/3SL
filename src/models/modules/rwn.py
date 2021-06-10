
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _single, _pair, _triple


class Conv2dRWN(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', momentum=0.99):
        super(Conv2dRWN, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        self.register_buffer('gamma', torch.ones((out_channels,)))
        self.momentum = momentum
        weight, _ = self._centered_norm()
        self.weight.data.copy_(weight)
        # self.init_gamma()

    def init_gamma(self):
        with torch.no_grad():
            _, gamma = self._centered_norm()
            self.gamma.copy_(gamma)

    def _centered_norm(self, keepdim=False):
        w_mean = self.weight.mean((1, 2, 3), keepdim=True)
        weight = self.weight - w_mean
        norm = weight.norm(2, (1, 2, 3), keepdim=keepdim)
        return weight, norm

    def remove_gamma(self):
        with torch.no_grad():
            weight, norm = self._centered_norm()
            self.weight.data.copy_(weight).mul_(self.gamma / norm)
            self.gamma = None

    def forward(self, input):
        if self.gamma is None:
            weight = self.weight
        else:
            weight, norm = self._centered_norm()
            if self.training:
                gamma = self.gamma.clone().detach()
                with torch.no_grad():
                    self.gamma.mul_(self.momentum).add_(1-self.momentum, norm)
            else:
                gamma = self.gamma
            weight = weight * (gamma / norm).view(-1, 1, 1, 1)

        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            return F.conv2d(F.pad(input, expanded_padding, mode='circular'),
                            weight, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


# class Conv2dRWN(nn.Conv2d):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1,
#                  padding=0, dilation=1, groups=1,
#                  bias=True, padding_mode='zeros', momentum=0.99, freq=100):
#         super(Conv2dRWN, self).__init__(
#             in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
#         self.register_buffer('gamma', torch.ones((out_channels,)))
#         self.register_buffer('count', torch.zeros((1,)))
#         self.momentum = momentum
#         self.freq = 10
#     #     self.init_gamma()

#     # def init_gamma(self):
#     #     with torch.no_grad():
#     #         _, gamma = self._centered_norm()
#     #         self.gamma.copy_(gamma)

#     def _centered_weight(self, keepdim=False):
#         w_mean = self.weight.mean((1, 2, 3), keepdim=True)
#         weight = self.weight - w_mean
#         norm = weight.norm(2, (1, 2, 3), keepdim=keepdim)
#         return weight / norm.view(-1, 1, 1, 1)

#     def remove_gamma(self):
#         with torch.no_grad():
#             weight, norm = self._centered_weight()
#             self.weight.data.copy_(weight).mul_(self.gamma / norm)
#             self.gamma = None

#     def forward(self, input):

#         weight = self._centered_weight()
#         # if self.gamma is not None:
#             # weight = weight * self.gamma.view(-1, 1, 1, 1)
            
#         if self.padding_mode == 'circular':
#             expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
#                                 (self.padding[0] + 1) // 2, self.padding[0] // 2)
#             out = F.conv2d(F.pad(input, expanded_padding, mode='circular'),
#                            weight, self.bias, self.stride,
#                            _pair(0), self.dilation, self.groups)
#         out = F.conv2d(input, weight, self.bias, self.stride,
#                        self.padding, self.dilation, self.groups)
#         # if self.training:
#         #     if int(self.count) == 0:
#         #         std = out.std((0, 2, 3))
#         #         # print(self.gamma)
#         #         with torch.no_grad():
#         #             self.gamma.mul_(self.momentum).add_(
#         #                 1-self.momentum, std.pow(-1))
#         #         self.count.fill_(self.freq)
#         #     else:
#         #         self.count.add(-1)

#         # if self.gamma is not None:
#         #     out = out * self.gamma.view(1, -1, 1, 1)
#         return out


class LinearRWN(nn.Linear):
    def __init__(self, in_features, out_features, bias=True,  momentum=0.9):
        super(LinearRWN, self).__init__(in_features, out_features, bias)
        self.init_gamma()
        self.momentum = momentum

    def init_gamma(self):
        with torch.no_grad():
            gamma = _norm(self.weight, dim=0, p=2)
            if not hasattr(self, 'gamma'):
                self.register_buffer('gamma', gamma)
            else:
                self.gamma.copy_(gamma)

    def forward(self, input):
        norm = _norm(self.weight, dim=0, p=2)
        gamma = self.gamma.clone()
        scale = gamma / norm
        weight = self.weight * scale
        if self.training:
            with torch.no_grad():
                self.gamma.mul_(self.momentum).add_(1-self.momentum, norm)
        return F.linear(input, weight, self.bias)
