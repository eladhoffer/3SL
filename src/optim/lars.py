"""
References:
    - https://arxiv.org/pdf/1708.03888.pdf
    - https://github.com/pytorch/pytorch/blob/1.6/torch/optim/sgd.py
    
Adapterd from https://github.com/PyTorchLightning/lightning-bolts/blob/master/pl_bolts/optimizers/lars.py
"""
import torch
from torch.optim.optimizer import Optimizer, required
try:
    from habana_frameworks.torch import core as htcore
    from habana_frameworks.torch import _hpex_C    
    from habana_frameworks.torch.hpex.optimizers import FusedLars, FusedSGD
    HPU_AVAILABLE = True
except:
    HPU_AVAILABLE = False

class LARS(Optimizer):
    """Extends SGD in PyTorch with LARS scaling from the paper
    `Large batch training of Convolutional Networks <https://arxiv.org/pdf/1708.03888.pdf>`_.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
        trust_coefficient (float, optional): trust coefficient for computing LR (default: 0.001)
        eps (float, optional): eps for division denominator (default: 1e-8)

    Example:
        >>> model = torch.nn.Linear(10, 1)
        >>> input = torch.Tensor(10)
        >>> target = torch.Tensor([1.])
        >>> loss_fn = lambda input, target: (input - target) ** 2
        >>> #
        >>> optimizer = LARS(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    .. note::
        The application of momentum in the SGD part is modified according to
        the PyTorch standards. LARS scaling fits into the equation in the
        following fashion.

        .. math::
            \begin{aligned}
                g_{t+1} & = \text{lars_lr} * (\beta * p_{t} + g_{t+1}), \\
                v_{t+1} & = \\mu * v_{t} + g_{t+1}, \\
                p_{t+1} & = p_{t} - \text{lr} * v_{t+1},
            \\end{aligned}

        where :math:`p`, :math:`g`, :math:`v`, :math:`\\mu` and :math:`\beta` denote the
        parameters, gradient, velocity, momentum, and weight decay respectively.
        The :math:`lars_lr` is defined by Eq. 6 in the paper.
        The Nesterov version is analogously modified.

    .. warning::
        Parameters with weight decay set to 0 will automatically be excluded from
        layer-wise LR scaling. This is to ensure consistency with papers like SimCLR
        and BYOL.
    """

    def __init__(
        self,
        params,
        lr=required,
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False,
        trust_coefficient=0.001,
        eps=1e-8,
        skip_scale=False
    ):
        if lr is not required and lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            trust_coefficient=trust_coefficient,
            eps=eps,
            skip_scale=skip_scale
        )
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")

        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)

        for group in self.param_groups:
            group.setdefault("nesterov", False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # exclude scaling for params with 0 weight decay
        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            nesterov = group["nesterov"]
            skip_scale = group["skip_scale"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                d_p = p.grad
                if skip_scale:
                    if weight_decay != 0:
                        d_p = d_p.add(p, alpha=weight_decay)
                else:
                    p_norm = torch.norm(p.data)
                    g_norm = torch.norm(p.grad.data)

                    # lars scaling + weight decay part
                    if p_norm != 0 and g_norm != 0:
                        lars_lr = p_norm / (g_norm + p_norm * weight_decay + group["eps"])
                        lars_lr *= group["trust_coefficient"]
                        if weight_decay != 0:
                            d_p = d_p.add(p, alpha=weight_decay)
                        d_p *= lars_lr

                # sgd part
                if momentum != 0:
                    param_state = self.state[p]
                    if "momentum_buffer" not in param_state:
                        buf = param_state["momentum_buffer"] = torch.clone(d_p).detach()
                    else:
                        buf = param_state["momentum_buffer"]
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                p.add_(d_p, alpha=-group["lr"])

        return loss

if HPU_AVAILABLE:
    class FusedLARS(FusedLars):
        def __init__(
            self,
            params,
            lr=required,
            momentum=0,
            dampening=0,
            weight_decay=0,
            nesterov=False,
            trust_coefficient=0.001,
            eps=1e-8,
            skip_scale=False
        ):
            optimizer = FusedSGD(params, lr, momentum, weight_decay, dampening, nesterov)
            for group in optimizer.param_groups:
                skip = group.get('skip_scale', False)
                group['skip_mask'] = [int(skip)] * len(group['params'])
            eeta = trust_coefficient
            skip_mask = [g['skip_mask'] for g in optimizer.param_groups]
            super().__init__(optimizer, skip_mask, eeta, eps)
        
        def step(self, closure=None):
            loss = None
            if closure is not None:
                loss = closure()

            with torch.no_grad():
                weight_decays = []
                for group in self.optim.param_groups:
                    # absorb weight decay control from optimizer
                    weight_decay = group['weight_decay'] if 'weight_decay' in group else 0
                    weight_decays.append(weight_decay)
                    group['weight_decay'] = 0
                    param_list = []
                    grad_list = []
                    skip_mask_list = group['skip_mask']
                    for idx, p in enumerate(group['params']):
                        if p.grad is None:
                            continue
                        param_list.append(p.data)
                        grad_list.append(p.grad.data)
                    htcore.mark_step()
                    _hpex_C.fused_lars(param_list, grad_list, skip_mask_list, self.eeta, weight_decay, self.eps, group['lr'])
                    htcore.mark_step()

            self.optim.step()
            # return weight decay control to optimizer
            for i, group in enumerate(self.optim.param_groups):
                group['weight_decay'] = weight_decays[i]
            return loss
