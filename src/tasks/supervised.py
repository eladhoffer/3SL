from pytorch_lightning.metrics import functional as FM
import torch.nn.functional as F
from .task import Task


class ClassificationTask(Task):
    def __init__(self, model, optimizer, **kwargs):
        super().__init__(model, optimizer, **kwargs)

    def training_step(self, batch, batch_idx):
        x, y = batch
        if isinstance(x, list):  # drop unlabled
            x, y = x
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        acc = FM.accuracy(y_hat.softmax(-1), y)
        self.log('lr', self.optim_regime.get_lr()[0], on_step=True)
        self.log_dict({
            'loss/train': loss,
            'accuracy/train': acc}, prog_bar=True, on_epoch=True, on_step=True)
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
