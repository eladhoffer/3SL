from typing import OrderedDict
from torchmetrics import functional as FM
import torch.nn.functional as F
from src.utils_pt.misc import calibrate_bn, no_bn_update
from src.tasks.task import Task
from src.models.modules.surrogate_norm import SNorm
from src.models.modules.utils import FrozenModule, freeze_model_, remove_module, replace_module
import torch
from hydra.utils import instantiate
from math import log


class ClassificationTask(Task):
    def __init__(self, model, optimizer, label_smoothing=0.0, calibrate_bn_on_eval=False, **kwargs):
        super().__init__(model, optimizer, **kwargs)
        self.label_smoothing = label_smoothing
        self.calibrate_bn_on_eval = calibrate_bn_on_eval
        self.save_hyperparameters()

    def loss(self, output, target):
        if self.mixup:
            target = self.mixup.mix_target(target, output.size(-1))
        return F.cross_entropy(output, target, label_smoothing=self.label_smoothing)

    def metrics(self, output, target):
        acc = FM.accuracy(output.detach().softmax(dim=-1), target)
        return {'accuracy': acc}

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        if isinstance(batch, dict):  # drop unlabled
            batch = batch['labeled']
        x, y = batch
        if self.mixup:
            self.mixup.sample(x.size(0))
            x = self.mixup(x)
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        self.log_lr(on_step=True)
        metrics = self.metrics(y_hat, y)
        metrics['loss'] = loss
        self.log_dict({f'{k}/train': v for k, v in metrics.items()},
                      prog_bar=True, on_epoch=False, on_step=True)
        if self.use_sam:
            eps_w = self.sam_step(loss)
            loss_w_sam = self.loss(self.model(x), y)
            # revert eps_w
            torch._foreach_sub_(list(self.parameters()), eps_w)
            self.manual_step(loss_w_sam)
        return loss

    def evaluation_step(self, batch, batch_idx):
        model = getattr(self, '_model_ema', self.model)
        x, y = batch
        if self.calibrate_bn_on_eval:
            with calibrate_bn(model):
                y_hat = model(x)
        else:
            y_hat = model(x)
        metrics = self.metrics(y_hat, y)
        metrics['loss'] = self.loss(y_hat, y)
        return metrics

    def validation_step(self, batch, batch_idx):
        metrics = self.evaluation_step(batch, batch_idx)
        loss = metrics['loss']
        metrics = {f'{k}/val': v for k, v in metrics.items()}
        self.log_dict(metrics)
        return loss

    def test_step(self, batch, batch_idx):
        metrics = self.evaluation_step(batch, batch_idx)
        loss = metrics['loss']
        metrics = {f'{k}/test': v for k, v in metrics.items()}
        self.log_dict(metrics)
        return loss


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

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        if isinstance(batch, dict):  # drop unlabled
            batch = batch['labeled']
        x, y = batch
        with torch.no_grad():
            self.teacher.eval()
            target = self.teacher(x)
        dist_batch = (x, target)
        return super().training_step(dist_batch)


class SupervisedEmbeddingTask(ClassificationTask):
    def __init__(self, model, optimizer, criterion=None,
                 transform_output=None, transform_target=None,
                 class_embeddings=None, finetune_class_embedding=False, **kwargs):
        super().__init__(model, optimizer, **kwargs)
        self.criterion = instantiate(criterion)
        self.model.criterion = self.criterion  # in case of parametrized loss
        if transform_target is not None:
            self.model.transform_target = instantiate(transform_target)
        else:
            self.model.transform_target = None

        if class_embeddings is not None:
            class_embeddings = torch.load(class_embeddings, map_location='cpu')
            if finetune_class_embedding:
                self.model.register_parameter('class_embeddings',
                                              torch.nn.Parameter(class_embeddings))
            else:
                self.model.register_buffer('class_embeddings', class_embeddings)
        self.save_hyperparameters()

    def embed_target(self, target):
        if target.dtype == torch.long:
            target = self.model.class_embeddings.index_select(0, target)
        if self.model.transform_target is not None:
            target = self.model.transform_target(target)
        return target

    def loss(self, output, target):
        target = self.embed_target(target)
        if self.mixup:
            target = self.mixup.mix_target(target, output.size(-1))
        return self.criterion(output, target).mean()

    def metrics(self, output, target):
        metric = {}
        if target.dtype == torch.long:
            target_embedding = self.embed_target(target)
            pred = output.mm(target_embedding.t())
            metric['accuracy'] = FM.accuracy(pred.softmax(dim=-1), target)

        return metric


class FinetuneTask(ClassificationTask):
    def __init__(self, model, optimizer, classifier, checkpoint_path,
                 replace_layer='fc', finetune_all=True, freeze_bn=False, strict_load=True, **kwargs):
        super().__init__(model, optimizer, **kwargs)
        self.freeze_bn = freeze_bn
        state_dict = torch.load(checkpoint_path)['state_dict']
        state_dict = {k.replace('module', 'model'): v for k, v in state_dict.items()}
        self.load_state_dict(state_dict, strict=strict_load)
        if not finetune_all:
            freeze_model_(self.model)
        if classifier is not None:
            classifier = instantiate(classifier)
            if replace_layer is not None:
                replace_module(self.model, replace_layer, classifier)
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        if self.freeze_bn:
            for m in self.model.modules():
                if isinstance(m, torch.nn.BatchNorm2d):
                    m.eval()
        return super().training_step(batch, batch_idx, optimizer_idx=optimizer_idx)


class ClassificationWNoiseTask(ClassificationTask):
    def __init__(self, model, optimizer, label_smoothing=0, entropy_eta=1e-4, surrogate_norm=False,
                 noise_example=True, calibrate_bn_on_eval=False, **kwargs):
        super().__init__(model, optimizer, label_smoothing, calibrate_bn_on_eval, **kwargs)
        self.noise_example = noise_example
        self.entropy_eta = entropy_eta
        if surrogate_norm:
            self.noise_example = True
            self.model = SNorm.convert_snorm(self.model)

    @staticmethod
    def noise_entropy(model, input_example, noise_output=None):
        if noise_output is None:
            noise = torch.randn_like(input_example)
            with no_bn_update(model) as model:
                noise_output = model(noise)
        noise_entropy = -(F.softmax(noise_output, dim=1) *
                          F.log_softmax(noise_output, dim=1)).sum(dim=1).mean()
        return noise_entropy

    def metrics(self, output, target, input_example=None):
        _, output = self._remove_noise_output(output)
        metrics = super().metrics(output, target)
        with torch.no_grad():
            noise_entropy = self.noise_entropy(self.model, input_example)
        metrics['noise_entropy'] = noise_entropy
        return metrics

    def loss(self, output, target, input_example=None):
        noise_output, output = self._remove_noise_output(output)
        loss = super().loss(output, target)
        if self.training and self.entropy_eta > 0:
            num_classes = output.size(-1)
            # norm_margin = F.mse_loss(noise_output, output.mean(dim=0).detach())
            noise_entropy = self.noise_entropy(self.model, input_example,
                                               noise_output=noise_output)
            # mean_entropy = -(F.softmax(output, dim=1) *
            #                  F.log_softmax(output, dim=1)).sum(dim=1).mean()
            # entropy_margin = (mean_entropy.detach() - noise_entropy).abs()
            entropy_margin = log(num_classes) - noise_entropy
            loss += self.entropy_eta * entropy_margin
        return loss

    def _add_noise_example(self, x):
        if self.training and self.noise_example:
            noise = torch.randn(1, *x.shape[1:], device=x.device, dtype=x.dtype)
            x = torch.cat((noise, x), dim=0)
        return x

    def _remove_noise_output(self, output):
        noise_output = None
        if self.training and self.noise_example:
            noise_output = output[0].unsqueeze(0)
            output = output[1:]
        return noise_output, output

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        if isinstance(batch, dict):  # drop unlabled
            batch = batch['labeled']
        x, y = batch
        x = self._add_noise_example(x)
        if self.mixup:
            self.mixup.sample(x.size(0))
            x = self.mixup(x)
        y_hat = self.model(x)
        loss = self.loss(y_hat, y, input_example=x)
        self.log_lr(on_step=True)
        metrics = self.metrics(y_hat, y, input_example=x)
        metrics['loss'] = loss
        self.log_dict({f'{k}/train': v for k, v in metrics.items()},
                      prog_bar=True, on_epoch=True, on_step=True)
        if self.use_sam:
            eps_w = self.sam_step(loss)
            loss_w_sam = self.loss(self.model(x), y)
            # revert eps_w
            torch._foreach_sub_(list(self.parameters()), eps_w)
            self.manual_step(loss_w_sam)
        return loss

    def evaluation_step(self, batch, batch_idx):
        model = getattr(self, '_model_ema', self.model)
        x, y = batch
        x = self._add_noise_example(x)
        if self.calibrate_bn_on_eval:
            with calibrate_bn(model):
                y_hat = model(x)
        else:
            y_hat = model(x)
        metrics = self.metrics(y_hat, y, input_example=x)
        metrics['loss'] = self.loss(y_hat, y, input_example=x)
        return metrics
