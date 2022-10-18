from src.tasks.supervised import ClassificationTask
from lowp import Lowp
from lowp.measured import register_qm
from lowp.measured.modules import calibrate_qmparametrize
from torchmetrics import functional as FM
import torch
from hydra.utils import instantiate


class QMClassificationTask(ClassificationTask):
    def __init__(self, *args, **kwargs):
        self.fixed_loss_scale = kwargs.pop('fixed_loss_scale', None)
        self.log_all_qstats = kwargs.pop('log_all_qstats', False)
        self.adaptive = kwargs.pop('adaptive', True)
        self.qm_config = instantiate(kwargs.pop('qm_config'))

        # default qupdater
        self.qupdater_config = kwargs.pop('qupdater', {'_target_': 'lowp.measured.QUpdater'})
        self.qupdater_config.setdefault('min_exp_bias', 0)
        self.qupdater_config.setdefault('max_exp_bias', 30)

        self.disable_amp_loss_scaler = kwargs.pop('disable_amp_loss_scaler', False)
        super().__init__(*args, **kwargs)
        register_qm(self.model, **self.qm_config)

    def configure_optimizers(self):
        if self.disable_amp_loss_scaler:
            self.trainer.precision_plugin.scaler = None
        if self.adaptive:
            self.qupdater = instantiate(self.qupdater_config,
                                        module=self.model, _convert_="all")
        return super().configure_optimizers()

    def log_qstats(self):
        def _log_all(module_name, module,
                     names=['statistics', 'exp_bias', 'grad_statistics', 'grad_exp_bias']):
            for name in names:
                stat = getattr(module, name, None)
                if stat is not None:
                    if 'statistics' in name and float(stat[0]) < 0:
                        continue
                    self.log(f'{name}/{module_name}', stat[0].item())

        for name, module in self.model.named_modules():
            for qm_name in ['input', 'output', 'weight', 'bias']:
                qm = getattr(module, qm_name, None)
                if qm is not None:
                    _log_all(f'{name}.{qm_name}', qm)

    def metrics(self, output, target=None, **kwargs):
        for key, value in list(kwargs.items()):
            if hasattr(value, 'tensor'):  # avoid logging qmtensors
                kwargs[key] = value.tensor
        if hasattr(output, 'tensor'):
            output = output.tensor
        if self.fixed_loss_scale is not None:
            kwargs['loss'] = kwargs['loss'] / self.fixed_loss_scale
        return super().metrics(output=output, target=target, **kwargs)

    def loss(self, output, target):
        loss = super().loss(output, target)
        if hasattr(loss, 'tensor'):
            loss = loss.tensor
        if self.fixed_loss_scale is not None:
            loss = loss * self.fixed_loss_scale
        return loss

    def on_before_optimizer_step(self, optimizer, optimizer_idx) -> None:
        if self.adaptive:
            self.qupdater.step()
        if self.log_all_qstats:
            self.log_qstats()
        if self.fixed_loss_scale is not None:
            with torch.no_grad():
                grads = [p.grad for p in self.parameters() if p.grad is not None]
                torch._foreach_div_(grads, self.fixed_loss_scale)
        return super().on_before_optimizer_step(optimizer, optimizer_idx)

    def calibrate(self, loader, num_steps=100):
        super().calibrate(loader, num_steps=num_steps)
        model = getattr(self, '_model_ema', self.model)
        model.eval()
        with calibrate_qmparametrize(model):
            qupdater = instantiate(self.qupdater_config,
                                   module=model, _convert_="all")
            for idx, batch in enumerate(loader):
                if idx > num_steps:
                    break
                x, _ = self.prepare_batch(batch)
                with torch.no_grad():
                    _ = model(x)
                qupdater.step()


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
        self.model.train()
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
