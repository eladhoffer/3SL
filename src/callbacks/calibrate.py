import torch
from torch.nn.modules.batchnorm import _BatchNorm
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.utilities.distributed import rank_zero_only

class CalibrateModel(Callback):
    """Make wandb watch model at the beginning of the run."""

    def __init__(self, split='train', num_steps=100, phase='val'):
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
            loader = self.get_loader(trainer)
            trainer.model.calibrate(loader, num_steps=self.num_steps)
        return super().on_validation_start(trainer, pl_module)

    def on_train_start(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        if self.phase == 'train':
            loader = self.get_loader(trainer)
            trainer.model.calibrate(loader, num_steps=self.num_steps)
        return super().on_train_end(trainer, pl_module)
