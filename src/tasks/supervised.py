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
    def __init__(self, model, optimizer, label_smoothing=0.0, **kwargs):
        super().__init__(model, optimizer, **kwargs)
        self.label_smoothing = label_smoothing
        self.save_hyperparameters()

    def loss(self, output, target):
        if self.mixup:
            target = self.mixup.mix_target(target, output.size(-1))
        return F.cross_entropy(output, target, label_smoothing=self.label_smoothing)

    def metrics(self, output, target=None, **kwargs):
        metrics_dict = {**kwargs}
        if output is not None and target is not None:
            metrics_dict['accuracy'] = FM.accuracy(output.detach().softmax(dim=-1), target)
        return metrics_dict

    def prepare_batch(self, batch):
        if isinstance(batch, dict):  # drop unlabled
            batch = batch['labeled']
        x, y = batch
        if getattr(self, 'channels_last', False):
            x = x.to(memory_format=torch.channels_last)
        if self.mixup:
            self.mixup.sample(x.size(0))
            x = self.mixup(x)
        return x, y

    def log_phase_dict(self, logged_dict, phase='train', **kwargs):
        logged_dict = {f'{k}/{phase}': v for k, v in logged_dict.items()}
        self.log_dict(logged_dict, **kwargs)

    def sam_update(self, x, y, loss):
        eps_w = self.sam_step(loss)
        loss_w_sam = self.loss(self.model(x), y)
        # revert eps_w
        torch._foreach_sub_(list(self.parameters()), eps_w)
        self.manual_step(loss_w_sam)
        return loss_w_sam

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        x, y = self.prepare_batch(batch)
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        metrics = self.metrics(output=y_hat, target=y, loss=loss)
        if self.use_sam:
            metrics['sam_loss'] = self.sam_update(x, y, loss)
        self.log_phase_dict(metrics, prog_bar=True, on_epoch=False, on_step=True)
        self.log_lr(on_step=True)
        return loss

    def calibrate(self, loader, num_steps=100):
        model = getattr(self, '_model_ema', self.model)
        with calibrate_bn(model) as model:
            for idx, batch in enumerate(loader):
                if idx > num_steps:
                    break
                x, _ = batch
                with torch.no_grad():
                    _ = model(x)

    def evaluation_step(self, batch, batch_idx, phase='val'):
        model = getattr(self, '_model_ema', self.model)
        x, y = batch
        y_hat = model(x)
        loss = self.loss(y_hat, y)
        metrics = self.metrics(y_hat, y, loss=loss)
        self.log_phase_dict(metrics, phase=phase)
        return loss

    def validation_step(self, batch, batch_idx):
        return self.evaluation_step(batch, batch_idx, phase='val')

    def test_step(self, batch, batch_idx):
        return self.evaluation_step(batch, batch_idx, phase='test')


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
