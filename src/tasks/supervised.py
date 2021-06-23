from pytorch_lightning.metrics import functional as FM
import torch.nn.functional as F
from src.utils_pt.cross_entropy import cross_entropy
from .task import Task
import torch


class ClassificationTask(Task):
    def __init__(self, model, optimizer, **kwargs):
        super().__init__(model, optimizer, **kwargs)

    def loss(self, output, target):
        if self.mixup:
            target = self.mixup.mix_target(target, output.size(-1))
        return cross_entropy(output, target)

    def training_step(self, batch, batch_idx):
        if isinstance(batch, dict):  # drop unlabled
            batch = batch['labeled']
        x, y = batch
        if self.mixup:
            self.mixup.sample(x.size(0))
            x = self.mixup(x)
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        acc = FM.accuracy(y_hat.softmax(-1), y)
        if self.optimizer_regime is not None:
            self.log('lr', self.optimizer_regime.get_lr()[0], on_step=True)
        self.log_dict({
            'loss/train': loss,
            'accuracy/train': acc}, prog_bar=True, on_epoch=True, on_step=True)
        if self.use_sam:
            eps_w = self.sam_step(loss)
            loss_w_sam = self.loss(self.model(x), y)
            # revert eps_w
            torch._foreach_sub_(list(self.parameters()), eps_w)
            self.manual_step(loss_w_sam)
        return loss

    def validation_step(self, batch, batch_idx):
        model = getattr(self, '_model_ema', self.model)
        model.eval()
        x, y = batch
        y_hat = model(x)
        loss = F.cross_entropy(y_hat, y)
        acc = FM.accuracy(y_hat.softmax(dim=-1), y)

        # loss is tensor. The Checkpoint Callback is monitoring 'checkpoint_on'
        metrics = {'accuracy/val': acc, 'loss/val': loss}
        self.log_dict(metrics)
        return loss

    def test_step(self, batch, batch_idx):
        metrics = self.validation_step(batch, batch_idx)
        metrics = {'accuracy/test': metrics['val_acc'],
                   'loss/test': metrics['val_loss']}
        self.log_dict(metrics)
