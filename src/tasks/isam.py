from .supervised import ClassificationTask
from pytorch_lightning.metrics import functional as FM
import torch
from torch.nn.utils.clip_grad import clip_grad_norm_


class ClassificationTaskiSAM(ClassificationTask):
    def __init__(self, model, optimizer, isam_rho=0.1, **kwargs):
        super(ClassificationTaskiSAM, self).__init__(model, optimizer, **kwargs)
        self.isam_rho = isam_rho
        self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        if isinstance(batch, dict):  # drop unlabled
            batch = batch['labeled']
        x, y = batch
        if self.mixup:
            self.mixup.sample(x.size(0))
            x = self.mixup(x)
        x = self.isam_step(x, y)
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        acc = FM.accuracy(y_hat.softmax(-1), y)
        self.log_lr(on_step=True)
        self.log_dict({
            'loss/train': loss,
            'accuracy/train': acc}, prog_bar=True, on_epoch=True, on_step=True)
        self.manual_step(loss)            
        return loss

    def isam_step(self, x, y):
        x.requires_grad_(True)
        for p in self.model.parameters():
            p.requires_grad_(False)
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        self.manual_backward(loss)
        with torch.no_grad():
            grad_norm = x.grad.norm(2, dim=(1, 2, 3), keepdim=True)
            # needed to revert in update
            eps_w = self.sam_rho * (x.grad / grad_norm)
            out = x + eps_w
        for p in self.model.parameters():
            p.requires_grad_(True)
        x.requires_grad_(False)
        out.requires_grad_(False)
        return out
