import itertools
import torch
from copy import deepcopy
from torch.optim.lr_scheduler import _LRScheduler, LambdaLR
from src.optim.scheduler import cosine_anneal
import math
from torch._six import inf

# copied and modified from clip_grad_norm_


def compute_norm(parameters, norm_type: float = 2.0,
                 error_if_nonfinite: bool = False) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(parameters)
    norm_type = float(norm_type)
    device = parameters[0].device
    if norm_type == inf:
        norms = [p.detach().abs().max().to(device) for p in parameters]
        total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
    else:
        total_norm = torch.norm(torch.stack(
            [torch.norm(p.detach(), norm_type).to(device) for p in parameters]), norm_type)
    if error_if_nonfinite and torch.logical_or(total_norm.isnan(), total_norm.isinf()):
        raise RuntimeError(
            f'The total norm of order {norm_type} from '
            '`parameters` is non-finite, so it cannot be clipped. To disable '
            'this error and scale the gradients by the non-finite norm anyway, '
            'set `error_if_nonfinite=False`')
    return total_norm


def clip_norm_(
        parameters, max_norm: float, norm_type: float = 2.0, normalize=False,
        error_if_nonfinite: bool = False) -> torch.Tensor:
    r"""Clips norm of an iterable of parameters.

    The norm is computed over all parameters together, as if they were
    concatenated into a single vector. Parameters are modified in-place.

    Args:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will be normalized
        max_norm (float or int): max norm
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
        normalize (bool): if True, the gradients are normalized to have max_norm norm
        error_if_nonfinite (bool): if True, an error is thrown if the total
            norm of the gradients from :attr:`parameters` is ``nan``,
            ``inf``, or ``-inf``. Default: False (will switch to True in the future)

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(parameters)
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    total_norm = compute_norm(parameters, norm_type)
    if error_if_nonfinite and torch.logical_or(total_norm.isnan(), total_norm.isinf()):
        raise RuntimeError(
            f'The total norm of order {norm_type} from '
            '`parameters` is non-finite, so it cannot be clipped. To disable '
            'this error and scale the gradients by the non-finite norm anyway, '
            'set `error_if_nonfinite=False`')
    clip_coef = max_norm / (total_norm + 1e-6)
    # Note: multiplying by the clamped coef is redundant when the coef is clamped to 1, but doing so
    # avoids a `if clip_coef < 1:` conditional which can require a CPU <=> device synchronization
    # when the gradients do not reside in CPU memory.
    if not normalize:
        clip_coef = torch.clamp(clip_coef, max=1.0)
    for p in parameters:
        p.detach().mul_(clip_coef.to(p.device))
    return total_norm


def clip_step_norm_(
        parameters, next_parameters=None, steps=None,
        max_norm: float = inf, norm_type: float = 2.0, normalize=False, clip_ratio=False,
        eps=1e-6, error_if_nonfinite: bool = False) -> torch.Tensor:
    assert next_parameters is not None or steps is not None,\
        "Either next_parameters or steps must be provided"
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(parameters)
    if isinstance(steps, torch.Tensor):
        steps = [steps]
    if steps is None and next_parameters is not None:
        if isinstance(next_parameters, torch.Tensor):
            next_parameters = [next_parameters]
        steps = [p_next - p for (p_next, p) in zip(next_parameters, parameters)]
    steps = list(steps)
    target_norm = 1
    if clip_ratio:
        target_norm = compute_norm(parameters, norm_type, error_if_nonfinite)
    max_norm = max_norm * target_norm
    step_norm = compute_norm(steps, norm_type, error_if_nonfinite)
    scale = max_norm / (step_norm + eps)
    if not normalize:
        scale = torch.clamp(scale, max=1.0)
    for p, step in zip(parameters, steps):
        p.copy_(p + scale * step)
    return step_norm, target_norm


def dot(xs1, xs2) -> torch.Tensor:
    if isinstance(xs1, torch.Tensor) and isinstance(xs2, torch.Tensor):
        return torch.dot(xs1.view(-1), xs2.view(-1))
    # compute sum of dot product between x1, x2
    return sum([torch.dot(x1.view(-1), x2.view(-1)) for x1, x2 in zip(xs1, xs2)])


def astensor(x):
    if isinstance(x, (list, tuple)):
        return torch.cat([v.view(-1) for v in x])
    else:
        return x


def sphere_vector_intersection(w1, g, r, w0):
    w1 = astensor(w1)
    g = astensor(g)
    if w0 is None:
        w0 = 0
    else:
        w0 = astensor(w0)

    # assumes g is normalized (norm of 1)
    a = torch.dot(g, g)
    b = 2 * torch.dot(g, (w1 - w0))
    c = torch.dot(w1 - w0, w1 - w0) - r * r

    discriminant = b * b - 4 * a * c
    print('discriminant=', discriminant)
    if discriminant < 0:
        # no intersection
        return None
    elif discriminant == 0:
        # one intersection (tangent)
        return -b / (2 * a)
    else:
        # two intersections (entering and exiting)
        sqrt_discriminant = torch.sqrt(discriminant)
        t1 = (-b + sqrt_discriminant) / (2 * a)
        t2 = (-b - sqrt_discriminant) / (2 * a)

        # choose the intersection closest to `w1`
        if t1 >= 0 and t2 >= 0:
            if t1 < t2:
                return t1
            else:
                return t2
        elif t1 >= 0:
            return t1
        elif t2 >= 0:
            return t2
        else:
            # both intersections are outside the vector
            return None


@torch.no_grad()
def clip_step_center_(
        parameters, next_parameters=None, centers=None, steps=None,
        max_norm: float = inf, norm_type: float = 2.0, normalize=False,
        eps=1e-6, error_if_nonfinite: bool = False) -> torch.Tensor:
    assert next_parameters is not None or steps is not None,\
        "Either next_parameters or steps must be provided"
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(parameters)
    if isinstance(steps, torch.Tensor):
        steps = [steps]
    if steps is None and next_parameters is not None:
        if isinstance(next_parameters, torch.Tensor):
            next_parameters = [next_parameters]
        steps = [p_next - p for (p_next, p) in zip(next_parameters, parameters)]
    steps = list(steps)
    next_parameters = list(next_parameters)
    scale = 1
    if not normalize:
        if centers is None:
            dist = compute_norm(parameters)
        else:
            dist = compute_norm((p - c for p, c in zip(next_parameters, centers)),
                                norm_type, error_if_nonfinite)
        print(dist)
        if dist > max_norm:
            normalize = True
    if normalize:
        # torch.save({
        #     'parameters': parameters,
        #     'next_parameters': next_parameters,
        #     'centers': centers
        # }, 'optim_states.pt')
        # exit()
        # dotsc = dot(steps, center_steps)
        # norm_steps = compute_norm(steps, norm_type, error_if_nonfinite)
        # normsq_center_steps = compute_norm(center_steps, norm_type, error_if_nonfinite) ** 2.
        # scale = (dotsc + math.sqrt(dotsc ** 2. - (normsq_center_steps - max_norm ** 2.))
        #          ) / (norm_steps ** 2.)
        direction = astensor(steps)
        direction = direction / (direction.norm() + eps)
        scale = sphere_vector_intersection(parameters, direction, max_norm, centers)

        print('scale = ', scale)
    for p, step in zip(parameters, steps):
        p.copy_(p - scale * step)
    dist = compute_norm(parameters, norm_type, error_if_nonfinite)
    return scale


class ClipScheduler(LambdaLR):

    def __init__(self, optimizer, clip_lambda, clip_ratio=False, clip_mean_ratio=False, normalize=False, base_value=1.0,
                 norm_type: float = 2.0, error_if_nonfinite: bool = False, eps=1e-6,
                 apply_globaly=True, momentum=0.99, last_epoch=-1, verbose=False):
        self.clip_ratio = clip_ratio
        self.clip_mean_ratio = clip_mean_ratio
        self.normalize = normalize
        self.norm_type = norm_type
        self.error_if_nonfinite = error_if_nonfinite
        self.eps = eps
        self.base_value = base_value
        self.apply_globaly = apply_globaly
        self.mean_step_norm = None
        self.momentum = momentum
        with torch.no_grad():
            self._prev_parameters = deepcopy(list(self._parameters(optimizer)))
            if self.clip_mean_ratio:
                self._mean_step = deepcopy(list(self._parameters(optimizer)))
        super().__init__(optimizer, clip_lambda, last_epoch, verbose)

    def _initial_step(self):
        """Initialize step counts and performs a step"""
        self.optimizer._step_count = 0
        self._step_count = 0
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

    def _parameters(self, optimizer=None):
        if optimizer is None:
            optimizer = self.optimizer
        param_groups = (group['params'] for group in optimizer.param_groups)

        if self.apply_globaly:  # flatten
            param_groups = [list(itertools.chain.from_iterable(param_groups))]
        return param_groups

    @torch.no_grad()
    def step(self, epoch=None):
        step_norms = []
        target_norms = []
        scales = []
        for idx, param_group in enumerate(self._parameters()):
            target_norm = 1.
            prev_params = self._prev_parameters[idx]
            lmbda = self.lr_lambdas[idx]
            step = [p - prev_p for (p, prev_p) in zip(param_group, prev_params)]
            step_norm = compute_norm(step)
            if self.clip_ratio:
                target_norm = compute_norm(param_group)
            elif self.clip_mean_ratio:
                mom = 0 if self._step_count == 0 else self.momentum
                for s, m in zip(step, self._mean_step[idx]):
                    m.copy_(mom * m + (1 - mom) * s / step_norm)
                target_norm = compute_norm(self._mean_step[idx])
            max_norm = lmbda(self.last_epoch) * target_norm
            scale = max_norm / (step_norm + self.eps)
            if not self.normalize:
                scale = torch.clamp(scale, max=1.0)
            for p, prev_p, s in zip(param_group, prev_params, step):
                p.copy_(prev_p + scale * s)
                prev_p.copy_(p)
            step_norms.append(float(step_norm))
            target_norms.append(float(target_norm))
            scales.append(float(scale))

        self._last_lr = scales
        self._last_step_norm = step_norm
        self._last_target_norm = target_norm
        self._step_count += 1
        if epoch is None:
            self.last_epoch += 1

    def get_last_metric(self):
        metrics = {}
        for name in ['step_norm', 'target_norm', 'lr']:
            metric_values = getattr(self, '_last_' + name, None)
            if metric_values is not None:
                metrics[name] = metric_values
        return metrics


class MeanNormAnnealingClip(ClipScheduler):
    def __init__(self, optimizer, base_clip, normalize=False,
                 norm_type: float = 2.0, error_if_nonfinite: bool = False, eps=1e-6, apply_globaly=True, last_epoch=-1, verbose=False):
        def _fixed(*kargs):
            return base_clip
        super().__init__(optimizer, clip_lambda=_fixed, clip_mean_ratio=True,
                         normalize=normalize, norm_type=norm_type, error_if_nonfinite=error_if_nonfinite, eps=eps,
                         apply_globaly=apply_globaly, last_epoch=last_epoch, verbose=verbose)


class CosineAnnealingClip(ClipScheduler):
    def __init__(self, optimizer, base_clip, T_max, eta_min=0, clip_ratio=False, clip_mean_ratio=False, normalize=False,
                 norm_type: float = 2.0, error_if_nonfinite: bool = False, eps=1e-6, apply_globaly=True, last_epoch=-1, verbose=False):
        clip_lambda = cosine_anneal(base_clip, eta_min, T_max)
        super().__init__(optimizer, clip_lambda=clip_lambda, clip_ratio=clip_ratio, clip_mean_ratio=clip_mean_ratio,
                         normalize=normalize, norm_type=norm_type, error_if_nonfinite=error_if_nonfinite, eps=eps,
                         apply_globaly=apply_globaly, last_epoch=last_epoch, verbose=verbose)
        self.eta_min = eta_min
        self.T_max = T_max


class ClipDistanceScheduler(ClipScheduler):

    def __init__(self, optimizer, clip_lambda, w0=None, clip_ratio=False, clip_mean_ratio=False, normalize=False,
                 norm_type: float = 2.0, error_if_nonfinite: bool = False, eps=1e-6, apply_globaly=True, last_epoch=-1, verbose=False):
        self.w0 = None if w0 is None else list(w0)

        super().__init__(optimizer, clip_lambda=clip_lambda, clip_ratio=clip_ratio, clip_mean_ratio=clip_mean_ratio,
                         normalize=normalize, norm_type=norm_type, error_if_nonfinite=error_if_nonfinite, eps=eps,
                         apply_globaly=apply_globaly, last_epoch=last_epoch, verbose=verbose)

    @torch.no_grad()
    def step(self, epoch=None):
        scales = []
        for idx, param_group in enumerate(self._parameters()):
            scale = clip_step_center_(self._prev_parameters[idx], param_group, centers=None if self.w0 is None else self.w0[idx],
                                      max_norm=self.lr_lambdas[idx](self.last_epoch))
            for p, prev_p in zip(param_group, self._prev_parameters[idx]):
                p.copy_(prev_p)
            scales.append(float(scale))

        self._last_lr = scales
        if epoch is None:
            self.last_epoch += 1


class CosineAnnealingClipDistance(ClipDistanceScheduler):
    def __init__(self, optimizer, base_clip, T_max, w0=None, eta_min=0, clip_ratio=False, clip_mean_ratio=False, normalize=False,
                 norm_type: float = 2.0, error_if_nonfinite: bool = False, eps=1e-6, apply_globaly=True, last_epoch=-1, verbose=False):
        clip_lambda = cosine_anneal(base_clip, eta_min, T_max)
        super().__init__(optimizer, clip_lambda=clip_lambda, w0=w0, clip_ratio=clip_ratio, clip_mean_ratio=clip_mean_ratio,
                         normalize=normalize, norm_type=norm_type, error_if_nonfinite=error_if_nonfinite, eps=eps,
                         apply_globaly=apply_globaly, last_epoch=last_epoch, verbose=verbose)
        self.eta_min = eta_min
        self.T_max = T_max


if __name__ == '__main__':
    ckpt = torch.load(
        "/home/ehoffer/Pytorch/3SL/results/step_clip/wresnet_cifar/w0/optim_states.pt")
