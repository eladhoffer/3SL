def linear_warmup_and_decay(lr, warmup_steps, total_steps, return_dict=True):
    lr = float(lr)

    def _lr_t_fn(t):
        if t <= warmup_steps:
            lr_t = (t / warmup_steps) * lr
        else:
            lr_t = max(((total_steps - t) / (total_steps - warmup_steps)) * lr, 0.)
        if return_dict:
            return {'lr': lr_t}
        else:
            return lr_t
    return _lr_t_fn
