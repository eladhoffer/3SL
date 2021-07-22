def linear_warmup_and_decay(lr, warmup_steps, total_steps):
    lr = float(lr)

    def lr_t(t):
        if t <= warmup_steps:
            return (t / warmup_steps) * lr
        else:
            return max(((total_steps - t) / (total_steps - warmup_steps)) * lr, 0.)
    return lr_t
