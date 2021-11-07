import torch
from torch.nn.modules.batchnorm import _BatchNorm
from pytorch_lightning import Callback, Trainer


class CalibrateModel(Callback):
    """Make wandb watch model at the beginning of the run."""

    def __init__(self, split='train', num_steps=100, phase='train'):
        self.split = split
        self.num_steps = num_steps
        self.phase = phase

    def get_loader(self, trainer):
        if self.split == 'train':
            return trainer.datamodule.train_dataloader()
        if self.split == 'val':
            return trainer.datamodule.val_dataloader()
        if self.split == 'test':
            return trainer.datamodule.test_dataloader()

    def on_validation_start(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        if self.phase == 'val':
            pl_module.calibrate_bn_on_eval = True
            trainer.validate(pl_module, self.get_loader(trainer))
            pl_module.calibrate_bn_on_eval = False
        return super().on_validation_start(trainer, pl_module)

    def on_train_start(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        if self.phase == 'train':
            pl_module.calibrate_bn_on_eval = True
            trainer.validate(pl_module, self.get_loader(trainer))
            pl_module.calibrate_bn_on_eval = False
        return super().on_train_end(trainer, pl_module)
