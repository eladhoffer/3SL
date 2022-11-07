import pytorch_lightning as pl
import torch
from copy import deepcopy
from hydra.utils import instantiate
from torch.nn.utils.clip_grad import clip_grad_norm_
from src.utils_pt.mixup import MixUp
from src.utils_pt.misc import calibrate_bn

try:
    import habana_frameworks.torch.core as htcore
except:
    pass


class Task(pl.LightningModule):

    def __init__(self, model, optimizer,
                 use_ema=False, ema_momentum=0.99, ema_bn_momentum=None, ema_device=None,
                 use_mixup=False, mixup_alpha=1.,
                 use_sam=False, sam_rho=0.05, sam_compare_grad=False,
                 channels_last=False, jit_model=False, **kwargs):
        super().__init__(**kwargs)
        self.model = instantiate(model)
        if channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)
            self.channels_last = True
        if jit_model:
            self.model = torch.jit.script(self.model)
        self.optimizer_config = optimizer
        self.optimizer_regime = None
        self._regularizers = []
        self.ema_momentum = ema_momentum
        self.ema_bn_momentum = ema_bn_momentum or ema_momentum
        self.use_sam = use_sam
        self.sam_rho = sam_rho
        self.sam_compare_grad = sam_compare_grad
        if use_ema and ema_momentum > 0:
            self.create_ema(device=ema_device)
        if use_sam:
            self.automatic_optimization = False
        if use_mixup:
            self.mixup = MixUp(alpha=mixup_alpha)
        else:
            self.mixup = None
        self.save_hyperparameters()

    def regularizers(self):
        return self._regularizers

    def configure_regularizers(self, regularizers, append=True):
        if not append:
            self._regularizers = []
        regularizers = [reg for reg in regularizers if reg is not None]
        self._regularizers.extend(regularizers)

    def configure_optimizers(self):
        optimizer_class = self.optimizer_config.get('_target_', None)
        assert optimizer_class is not None, 'optimizer class is not defined'
        if 'OptimRegime' in optimizer_class:
            self.optimizer_regime = instantiate(self.optimizer_config,
                                                model=self.model, _convert_="all")
            return self.optimizer_regime.optimizer
        elif 'OptimConfig' in optimizer_class:  # regular optimizer
            optimizer_config = instantiate(self.optimizer_config, model=self.model,
                                           _convert_="all", _recursive_=False)
            self.configure_regularizers(optimizer_config.regularizers())
            return optimizer_config.configuration()

    def on_train_batch_start(self, *kargs, **kwargs) -> None:
        if self.optimizer_regime is not None:
            self.optimizer_regime.update(self.current_epoch, self.global_step + 1)
            self.optimizer_regime.pre_forward()
            self.optimizer_regime.pre_backward()
        for regularizer in self.regularizers():
            regularizer.pre_forward()
            regularizer.pre_backward()
        return super().on_train_batch_start(*kargs, **kwargs)

    def on_after_backward(self) -> None:
        if self.optimizer_regime is not None:
            self.optimizer_regime.pre_step()
        for regularizer in self.regularizers():
            regularizer.pre_step()
        return super().on_after_backward()

    def on_before_zero_grad(self, optimizer) -> None:
        if self.optimizer_regime is not None:
            self.optimizer_regime.post_step()
        for regularizer in self.regularizers():
            regularizer.post_step()
        return super().on_before_zero_grad(optimizer)

    def training_step_end(self, losses):
        self.update_ema()
        return losses.mean()

    def log_lr(self, **kwargs):
        lrs = []
        if self.optimizer_regime is not None:
            lrs = self.optimizer_regime.get_lr()
        else:
            lr_schedulers = self.lr_schedulers()
            if not isinstance(lr_schedulers, (list, tuple)):
                lr_schedulers = [lr_schedulers]
            for lr_scheduler in lr_schedulers:
                if lr_scheduler is not None:
                    lrs += lr_scheduler.get_last_lr()
        for idx, lr in enumerate(lrs):
            name = 'lr' if idx == 0 else f'lr_{idx}'
            self.log(name, lr, **kwargs)

    def create_ema(self, device=None):
        self._model_ema = deepcopy(self.model)
        if device is not None:
            self._model_ema.to(device)
        for p in self._model_ema.parameters():
            p.requires_grad_(False)

    def update_ema(self):
        if getattr(self, '_model_ema', None) is None:
            return
        with torch.no_grad():
            for p_ema, p in zip(self._model_ema.parameters(),
                                self.model.parameters()):
                p_ema.lerp_(p, 1. - self.ema_momentum)
            for b_ema, b in zip(self._model_ema.buffers(),
                                self.model.buffers()):
                if b_ema.dtype == torch.long or b.dtype == torch.long:
                    b_ema.copy_(b)
                else:
                    b_ema.lerp_(b, 1. - self.ema_bn_momentum)

    def set_benchmark(self, *kargs, **kwargs):
        pass

    def manual_step(self, loss, set_to_none=False):
        if self.use_sam and self.sam_compare_grad:
            grad_sam = torch.cat([p.grad.view(-1) for p in self.model.parameters()])
        opt = self.optimizers()
        opt.zero_grad(set_to_none=set_to_none)
        self.manual_backward(loss)
        if self.use_sam and self.sam_compare_grad:
            grad = torch.cat([p.grad.view(-1) for p in self.model.parameters()])
            self.log('sam/grad_eps_diff', (grad - grad_sam).norm())
        opt.step()

    def sam_step(self, loss, set_to_none=False):
        self.model.zero_grad(set_to_none=set_to_none)
        self.manual_backward(loss)
        with torch.no_grad():
            params, grads = zip(*[(p, p.grad)
                                  for p in self.model.parameters() if p.grad is not None])
            grad_norm = clip_grad_norm_(self.model.parameters(),
                                        max_norm=float('inf'), norm_type=2.0)
            # needed to revert in update
            eps_w = torch._foreach_mul(grads, self.sam_rho / grad_norm)
            torch._foreach_add_(params, eps_w)
        return eps_w

    def prepare_batch(self, batch):
        if isinstance(batch, dict):  # drop unlabled
            batch = batch['labeled']
        x, y = batch
        if getattr(self, 'channels_last', False):
            x = x.to(device=self.device,
                     memory_format=torch.channels_last)
        else:
            x = x.to(device=self.device)
        y = y.to(device=self.device)
        if self.mixup:
            self.mixup.sample(x.size(0))
            x = self.mixup(x)
        return x, y

    def calibrate(self, loader, num_steps=100):
        model = getattr(self, '_model_ema', self.model)
        with calibrate_bn(model) as model:
            for idx, batch in enumerate(loader):
                if idx > num_steps:
                    break
                x, _ = self.prepare_batch(batch)
                with torch.no_grad():
                    _ = model(x)
