from .supervised import ClassificationTask
from pytorch_lightning.metrics import functional as FM
import torch.nn.functional as F
from src.utils_pt.cross_entropy import cross_entropy
import torch
import math


class MaskedLanguageModelTask(ClassificationTask):
    """
    A task for training a masked language model.
    """

    def __init__(self, model, optimizer,
                 input_keys=['input_ids', 'attention_mask'], output_key='prediction_logits',
                 **kwargs):
        super().__init__(model, optimizer, **kwargs)
        self.input_keys = input_keys
        self.output_key = output_key

    def loss(self, output, target):
        output = output.flatten(0, 1)
        target = target.flatten(0, 1)
        return cross_entropy(output, target, smooth_eps=self.label_smoothing)

    def training_step(self, batch, batch_idx):
        x = {ikey: batch[ikey] for ikey in self.input_keys}
        y = batch['labels']
        y_hat = self.model(**x)[self.output_key]
        loss = self.loss(y_hat, y)
        self.log_lr(on_step=True)
        self.log_dict({
            'loss/train': loss}, prog_bar=True, on_epoch=True, on_step=True)
        if self.use_sam:
            eps_w = self.sam_step(loss)
            y_hat = self.model(**x)[self.output_key]
            loss_w_sam = self.loss(y_hat, y)
            # revert eps_w
            torch._foreach_sub_(list(self.parameters()), eps_w)
            self.manual_step(loss_w_sam)
        return loss

    def training_step_end(self, losses):
        return super().training_step_end(losses)

    def evaluation_step(self, batch, batch_idx):
        model = getattr(self, '_model_ema', self.model)
        model.eval()
        x = {ikey: batch[ikey] for ikey in self.input_keys}
        y = batch['labels']
        breakpoint()

        y_hat = self.model(**x)[self.output_key]
        loss = self.loss(y_hat, y)
        metrics = {'loss': loss}
        return metrics

    # def evaluation_step_end(self, batch, batch_idx):
    #     metrics = {'loss': loss}
    #     return super().evaluation_step(batch, batch_idx)

    # def validation_step(self, batch, batch_idx):
    #     return super().validation_step(batch, batch_idx)

    # def validation_step_end(self, *args, **kwargs) -> Optional[STEP_OUTPUT]:
    #     return super().validation_step_end(*args, **kwargs)

    # def test_step(self, batch, batch_idx):
    #     return super().test_step(batch, batch_idx)

    # def test_step_end(self, *args, **kwargs) -> Optional[STEP_OUTPUT]:
    #     return super().test_step_end(*args, **kwargs)
