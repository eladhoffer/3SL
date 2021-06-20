import torch
from torch.nn.modules.batchnorm import _BatchNorm
from pytorch_lightning import Callback, Trainer


def calibrate_bn(model, dataloader, num_steps=None):
    with torch.no_grad():
        prev_settings = {}
        for name, m in model.named_modules():
            if isinstance(m, _BatchNorm):
                prev_settings[name] = {
                    'momentum': m.momentum,
                    'track_running_stats': m.track_running_stats
                }
                m.momentum = None
                m.track_running_stats = True
                m.reset_running_stats()
        model.train()
        data_iter = iter(dataloader)
        for _ in range(num_steps):
            try:
                x = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                x = next(data_iter)
            model(x)
        for name, m in model.named_modules():
            if isinstance(m, _BatchNorm):
                m.momentum = prev_settings[name]['momentum']
                m.track_running_stats = prev_settings[name]['track_running_stats']


class CalibrateModel(Callback):
    """Make wandb watch model at the beginning of the run."""

    def __init__(self, datamodule, split='train', num_steps=100, phase='train'):
        self.datamodule = datamodule
        self.split = split
        self.num_steps = num_steps
        self.phase = phase

    def get_loader(self):
        if self.split == 'train':
            return self.datamodule.train_dataloader()
        if self.split == 'val':
            return self.datamodule.val_dataloader()
        if self.split == 'test':
            return self.datamodule.test_dataloader()

    def on_validation_start(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        if self.phase == 'val':
            calibrate_bn(trainer.model, dataloader=self.get_loader(), num_steps=self.num_steps)
        return super().on_validation_start(trainer, pl_module)

    def on_train_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        if self.phase == 'train':
            calibrate_bn(trainer.model, dataloader=self.get_loader(), num_steps=self.num_steps)
        return super().on_train_end(trainer, pl_module)
