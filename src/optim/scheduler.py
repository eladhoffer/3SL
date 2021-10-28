from torch.optim.lr_scheduler import _LRScheduler, LambdaLR


def linear_warmup_and_decay(base_value, warmup_steps, total_steps, init_value=0.):
    value = float(base_value)

    def _value_t_fn(t):
        if t <= warmup_steps:
            value_t = init_value + (t / warmup_steps) * (value - init_value)
        else:
            value_t = max(((total_steps - t) / (total_steps - warmup_steps)) * value, 0.)
        return value_t
    return _value_t_fn


def linear_warmup_and_decay_lr(lr, warmup_steps, total_steps, init_lr=0., return_dict=True):
    lr = linear_warmup_and_decay(lr, warmup_steps, total_steps, init_lr, return_dict)
    if return_dict:
        return {'lr': lr}
    else:
        return lr


class LinearWarmUpDecay(LambdaLR):
    def __init__(self, optimizer, lr, warmup_steps, total_steps, init_lr=0.,
                 last_epoch=-1, verbose=False):
        lr_fn = linear_warmup_and_decay_lr(lr=lr, warmup_steps=warmup_steps,
                                           total_steps=total_steps, init_lr=init_lr,
                                           return_dict=False)
        super().__init__(optimizer, lr_fn,
                         last_epoch=last_epoch, verbose=verbose)

    def get_lr(self):
        return [lmbda(self.last_epoch) for lmbda in self.lr_lambdas]


class MultiplierLinearWarmUpDecay(LambdaLR):
    def __init__(self, optimizer, lr_multiplier, warmup_steps, total_steps, init_lr_multiplier=0.,
                 last_epoch=-1, verbose=False):
        lr_fn = linear_warmup_and_decay(base_value=lr_multiplier, warmup_steps=warmup_steps,
                                        total_steps=total_steps, init_value=init_lr_multiplier,
                                        return_dict=False)
        super().__init__(optimizer, lr_fn,
                         last_epoch=last_epoch, verbose=verbose)
