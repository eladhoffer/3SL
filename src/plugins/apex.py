
import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities import _APEX_AVAILABLE, AMPType
from pytorch_lightning.plugins.precision.apex_amp import ApexMixedPrecisionPlugin

if _APEX_AVAILABLE:
    from apex import amp


class ConfigurableApexMixedPrecisionPlugin(ApexMixedPrecisionPlugin):
    def __init__(self, amp_level: str, loss_scale=None,
                 min_loss_scale=None, max_loss_scale=2. ** 24.) -> None:
        super().__init__(amp_level=amp_level)
        self.loss_scale = loss_scale
        self.min_loss_scale = min_loss_scale
        self.max_loss_scale = max_loss_scale

    def dispatch(self, trainer: "pl.Trainer") -> None:
        if not self._connected:
            accelerator = trainer.accelerator
            _, accelerator.optimizers = amp.initialize(
                trainer.lightning_module, accelerator.optimizers, opt_level=self.amp_level,
                loss_scale=self.loss_scale, min_loss_scale=self.min_loss_scale,
                max_loss_scale=self.max_loss_scale
            )
            self._connected = True
        return super().dispatch(trainer)
