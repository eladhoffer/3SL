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
