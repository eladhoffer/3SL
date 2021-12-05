from torch.optim.lr_scheduler import _LRScheduler, LambdaLR


def polynomial_warmup_and_decay(base_value, warmup_steps, total_steps,
                                init_value=0., final_value=0.,
                                warmup_power=1., decay_power=1.):
    value = float(base_value)

    def _value_t_fn(t):
        if t <= warmup_steps:
            value_t = init_value + (value - init_value) * (t / warmup_steps) ** warmup_power
        else:
            value_t = value * ((total_steps - t) / (total_steps - warmup_steps)) ** decay_power
            value_t = max(value_t, final_value)
        return value_t
    return _value_t_fn


def linear_warmup_and_decay(base_value, warmup_steps, total_steps, init_value=0., final_value=0.):
    return polynomial_warmup_and_decay(base_value, warmup_steps, total_steps, init_value, final_value)


def linear_warmup_and_decay_lr(lr, warmup_steps, total_steps, init_lr=0., final_lr=0., return_dict=True):
    lr = linear_warmup_and_decay(lr, warmup_steps, total_steps, init_lr, final_lr)
    if return_dict:
        return {'lr': lr}
    else:
        return lr


class MultiplierPolynomialWarmUpDecay(LambdaLR):
    def __init__(self, optimizer, multiplier, warmup_steps, total_steps, init_multiplier=0., final_multiplier=0.,
                 warmup_power=1., decay_power=1., last_epoch=-1, verbose=False):
        multiplier_fn = polynomial_warmup_and_decay(multiplier, warmup_steps=warmup_steps, total_steps=total_steps,
                                                    init_value=init_multiplier, final_value=final_multiplier,
                                                    warmup_power=warmup_power, decay_power=decay_power)
        super().__init__(optimizer, multiplier_fn,
                         last_epoch=last_epoch, verbose=verbose)


class PolynomialWarmUpDecay(LambdaLR):
    def __init__(self, optimizer, lr, warmup_steps, total_steps, init_lr=0., final_lr=0.,
                 warmup_power=1., decay_power=1., last_epoch=-1, verbose=False):
        lr_fn = polynomial_warmup_and_decay(lr, warmup_steps=warmup_steps, total_steps=total_steps,
                                            init_value=init_lr, final_value=final_lr,
                                            warmup_power=warmup_power, decay_power=decay_power)
        super().__init__(optimizer, lr_fn,
                         last_epoch=last_epoch, verbose=verbose)

    def get_lr(self):
        return [lmbda(self.last_epoch) for lmbda in self.lr_lambdas]


class LinearWarmUpDecay(PolynomialWarmUpDecay):
    def __init__(self, optimizer, lr, warmup_steps, total_steps,
                 init_lr=0., final_lr=0., last_epoch=-1, verbose=False):
        super().__init__(optimizer, lr, warmup_steps, total_steps, init_lr, final_lr,
                         warmup_power=1., decay_power=1., last_epoch=last_epoch, verbose=verbose)


class MultiplierLinearWarmUpDecay(MultiplierPolynomialWarmUpDecay):
    def __init__(self, optimizer, lr, warmup_steps, total_steps,
                 init_lr=0., final_lr=0., last_epoch=-1, verbose=False):
        super().__init__(optimizer, lr, warmup_steps, total_steps, init_lr, final_lr,
                         warmup_power=1., decay_power=1., last_epoch=last_epoch, verbose=verbose)
