from torch.optim.lr_scheduler import *


def linear_warmup_and_decay(lr, warmup_steps, total_steps, init_lr=0., return_dict=True):
    lr = float(lr)

    def _lr_t_fn(t):
        if t <= warmup_steps:
            lr_t = init_lr + (t / warmup_steps) * (lr - init_lr)
        else:
            lr_t = max(((total_steps - t) / (total_steps - warmup_steps)) * lr, 0.)
        if return_dict:
            return {'lr': lr_t}
        else:
            return lr_t
    return _lr_t_fn


class LinearWarmUpDecay(LambdaLR):
    def __init__(self, optimizer, lr, warmup_steps, total_steps, init_lr=0.,
                 last_epoch=-1, verbose=False):
        lr_fn = linear_warmup_and_decay(lr=lr, warmup_steps=warmup_steps,
                                        total_steps=total_steps, init_lr=init_lr,
                                        return_dict=False)
        super().__init__(optimizer, lr_fn,
                         last_epoch=last_epoch, verbose=verbose)

    def get_lr(self):
        return [lmbda(self.last_epoch) for lmbda in self.lr_lambdas]
