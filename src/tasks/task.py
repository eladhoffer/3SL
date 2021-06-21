import pytorch_lightning as pl
import torch
from src.utils_pt.optim import OptimRegime
from copy import deepcopy
from hydra.utils import instantiate


class Task(pl.LightningModule):

    def __init__(self, model, optimizer,
                 use_ema=False, ema_momentum=0.99, ema_bn_momentum=None, ema_device=None,
                 jit_model=False, **kwargs):
        super().__init__(**kwargs)
        self.model = instantiate(model)
        if jit_model:
            self.model = torch.jit.script(self.model)
        self.optimizer_config = optimizer
        self.optimizer_regime = None
        self.ema_momentum = ema_momentum
        self.ema_bn_momentum = ema_bn_momentum or ema_momentum
        if use_ema and ema_momentum > 0:
            self.create_ema(device=ema_device)
        self.save_hyperparameters()

    def configure_optimizers(self):
        if 'OptimRegime' in self.optimizer_config.get('_target_', None):
            self.optimizer_regime = instantiate(self.optimizer_config,
                                                model=self.model, _convert_="all")
            return self.optimizer_regime.optimizer
        else:  # regular optimizer
            return instantiate(self.optimizer_config,
                               params=self.model.parameters(), _convert_="all")

    def on_train_batch_start(self, batch, batch_idx: int, dataloader_idx: int) -> None:
        if self.optimizer_regime is not None:
            self.optimizer_regime.update(self.current_epoch, self.global_step + 1)
            self.optimizer_regime.pre_forward()
            self.optimizer_regime.pre_backward()
        return super().on_train_batch_start(batch, batch_idx, dataloader_idx)

    def on_after_backward(self) -> None:
        if self.optimizer_regime is not None:
            self.optimizer_regime.pre_step()
        return super().on_after_backward()

    def on_before_zero_grad(self, optimizer) -> None:
        if self.optimizer_regime is not None:
            self.optimizer_regime.post_step()
        return super().on_before_zero_grad(optimizer)

    def training_step_end(self, losses):
        self.update_ema()
        return losses.mean()

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
