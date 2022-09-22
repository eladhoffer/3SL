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
        self.fixed_loss_scale = kwargs.pop('fixed_loss_scale', None)
        self.log_all_qstats = kwargs.pop('log_all_qstats', False)
        self.adaptive = kwargs.pop('adaptive', True)
        qm_config = kwargs.pop('qm_config', {})
        super().__init__(*args, **kwargs)
        register_qm(self.model, **qm_config)

    def log_qstats(self):
        def _log_all(module_name, module,
                     names=['statistics', 'exp_bias', 'grad_statistics', 'grad_exp_bias']):
            for name in names:
                stat = getattr(module, name, None)
                if stat is not None:
                    if 'statistics' in name and float(stat) < 0:
                        continue
                    self.log(f'{name}/{module_name}', stat.item())

        for name, module in self.model.named_modules():
            for qm_name in ['input', 'output', 'weight', 'bias']:
                qm = getattr(module, qm_name, None)
                if qm is not None:
                    _log_all(f'{name}.{qm_name}', qm)

    def metrics(self, output, target=None, **kwargs):
        #kwargs['loss'] = kwargs['loss'].tensor
        output = output.tensor
        if self.fixed_loss_scale is not None:
            kwargs['loss'] = kwargs['loss'] / self.fixed_loss_scale
            # try:
            #     kwargs['bn1_grad'] = self.model.bn1.weight.grad.mean()
            # except AttributeError:
            #     pass
        return super().metrics(output=output, target=target, **kwargs)

    def loss(self, output, target):
        loss = super().loss(output, target)
        if self.fixed_loss_scale is not None:
            loss = loss * self.fixed_loss_scale
        return loss.tensor

    def on_train_start(self) -> None:
        self.qupdater = QUpdater(self.model, min_exp_bias=0, max_exp_bias=30)
        return super().on_train_start()

    def on_train_batch_start(self, *kargs, **kwargs) -> None:
        if self.log_all_qstats:
            self.log_qstats()
        if self.adaptive:
            self.qupdater.step()
        return super().on_train_batch_start(*kargs, **kwargs)

    def on_before_optimizer_step(self, optimizer, optimizer_idx) -> None:
        if self.fixed_loss_scale is not None:
            with torch.no_grad():
                grads = [p.grad for p in self.parameters() if p.grad is not None]
                torch._foreach_div_(grads, self.fixed_loss_scale)
        return super().on_before_optimizer_step(optimizer, optimizer_idx)


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
