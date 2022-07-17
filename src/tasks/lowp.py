from src.tasks.supervised import ClassificationTask
from lowp import Lowp
from lowp.functional import truncate_fp8
from lowp.measured import register_qm, QUpdater
from torchmetrics import functional as FM
import torch.nn.functional as F
from src.utils_pt.cross_entropy import cross_entropy
from src.tasks.task import Task
import torch
from hydra.utils import instantiate
from copy import deepcopy


class QMClassificationTask(ClassificationTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        register_qm(self.model)
        self.qupdater = QUpdater(self.model)

    def log_qstats(self):
        def _log_all(module_name, module,
                     names=['statistics', 'exp_bias', 'grad_statistics', 'grad_exp_bias']):
            for name in names:
                stat = getattr(module, name, None)
                if stat is not None:
                    self.log(f'{name}/{module_name}', stat.item())

        for name, module in self.model.named_modules():
            for qm_name in ['input', 'output', 'weight', 'bias']:
                qm = getattr(module, qm_name, None)
                if qm is not None:
                    _log_all(f'{name}.{qm_name}', qm)

    def metrics(self, output, target):
        output = output.tensor
        acc = FM.accuracy(output.detach().softmax(dim=-1), target)
        return {'accuracy': acc}

    def loss(self, output, target):
        return super().loss(output, target).tensor

    def on_train_batch_start(self, batch, batch_idx: int, dataloader_idx: int) -> None:
        self.log_qstats()
        return super().on_train_batch_start(batch, batch_idx, dataloader_idx)


class LowpClassificationTask(ClassificationTask):
    def __init__(self, *args, **kwargs):
        self.lowp_mode = kwargs.pop('lowp_mode', 'FP8')
        self.lowp_warn_patched = kwargs.pop('lowp_warn_patched', True)
        self.lowp_warn_not_patched = kwargs.pop('lowp_warn_not_patched', True)
        self.lowp_exclude = kwargs.pop('lowp_exclude', [])
        self.params_precision = kwargs.pop('params_precision', 32)
        self.optimizer_samples = kwargs.pop('optimizer_samples', 1)
        super().__init__(*args, **kwargs)
        if self.params_precision == 16:
            self.model = self.model.to(dtype=torch.half)

    def training_step(self, batch, batch_idx):
        if isinstance(batch, dict):  # drop unlabled
            batch = batch['labeled']
        x, y = batch
        if self.mixup:
            self.mixup.sample(x.size(0))
            x = self.mixup(x)
        with Lowp(mode=self.lowp_mode,
                  warn_patched=self.lowp_warn_patched,
                  warn_not_patched=self.lowp_warn_not_patched,
                  exclude=self.lowp_exclude):
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

    def evaluation_step(self, batch, batch_idx):
        model = getattr(self, '_model_ema', self.model)
        model.eval()
        x, y = batch
        with Lowp(mode=self.lowp_mode, warn_patched=self.lowp_warn_patched, warn_not_patched=self.lowp_warn_not_patched):
            y_hat = model(x)
        loss = self.loss(y_hat, y)
        acc = FM.accuracy(y_hat.softmax(dim=-1), y)
        metrics = {'accuracy': acc, 'loss': loss}
        return metrics
