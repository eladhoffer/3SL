from pytorch_lightning.metrics import functional as FM
import torch.nn.functional as F
from src.utils_pt.cross_entropy import cross_entropy
from .task import Task
import torch
from hydra.utils import instantiate


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
        self.log_lr(on_step=True)
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
        loss = cross_entropy(y_hat, y)
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


class DistillationTask(ClassificationTask):
    def __init__(self, model, optimizer, teacher, T=1.0,
                 jit_teacher=False, **kwargs):
        super().__init__(model, optimizer, **kwargs)
        self.teacher = instantiate(teacher)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        if jit_teacher:
            self.teacher = torch.jit.script(self.teacher)

        self.T = T

    def loss(self, output, target):
        with torch.no_grad():
            target /= self.T
            if self.mixup:
                target = self.mixup.mix_target(target, output.size(-1))
                target = F.log_softmax(target, -1)
        output = F.log_softmax(output / self.T, -1)
        return F.kl_div(output, target, reduction='batchmean', log_target=True)

    def training_step(self, batch, batch_idx):
        if isinstance(batch, dict):  # drop unlabled
            batch = batch['labeled']
        x, y = batch
        with torch.no_grad():
            self.teacher.eval()
            target = self.teacher(x)
        dist_batch = (x, target)
        return super().training_step(dist_batch)
