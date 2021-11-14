from typing import OrderedDict
from pytorch_lightning.metrics import functional as FM
import torch.nn.functional as F
from src.utils_pt.cross_entropy import cross_entropy
from src.utils_pt.misc import calibrate_bn
from src.tasks.task import Task
import torch
from hydra.utils import instantiate


class ClassificationTask(Task):
    def __init__(self, model, optimizer, label_smoothing=None, calibrate_bn_on_eval=False, **kwargs):
        super().__init__(model, optimizer, **kwargs)
        self.label_smoothing = label_smoothing
        self.calibrate_bn_on_eval = calibrate_bn_on_eval
        self.save_hyperparameters()

    def loss(self, output, target):
        if self.mixup:
            target = self.mixup.mix_target(target, output.size(-1))
        return cross_entropy(output, target, smooth_eps=self.label_smoothing)

    def metrics(self, output, target):
        acc = FM.accuracy(output.softmax(dim=-1), target)
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
        metrics = {f'{k}/val': v for k, v in metrics.items()}
        self.log_dict(metrics)
        return metrics

    def test_step(self, batch, batch_idx):
        metrics = self.evaluation_step(batch, batch_idx)
        metrics = {f'{k}/test': v for k, v in metrics.items()}
        self.log_dict(metrics)
        return metrics


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
                 transform_target=None,
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


class FrozenModule(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        for p in self.module.parameters():
            p.requires_grad_(False)

    def forward(self, *args, **kwargs):
        with torch.no_grad():
            return self.module(*args, **kwargs)


def _remove_module(model, name):
    for n, m in model.named_children():
        if n == name:
            setattr(model, name, torch.nn.Identity())
        _remove_module(m, name)


class FinetuneTask(ClassificationTask):
    def __init__(self, model, optimizer, classifier, checkpoint_path,
                 remove_layer='fc', finetune_all=True, freeze_bn=False, **kwargs):
        super().__init__(model, optimizer, **kwargs)
        self.freeze_bn = freeze_bn
        state_dict = torch.load(checkpoint_path)['state_dict']
        # state_dict = {k.replace('module', 'model'): v for k, v in state_dict.items()}
        self.load_state_dict(state_dict, strict=False)
        if remove_layer is not None:
            _remove_module(self.model, remove_layer)
        if classifier is not None:
            self.classifier = instantiate(classifier)
            self.model = torch.nn.Sequential(OrderedDict([
                ('pretrained', self.model if finetune_all else FrozenModule(self.model)),
                ('classifier', self.classifier)
            ]))
        # if load_all:
        #     state_dict = torch.load(checkpoint_path)['state_dict']
        #     self.load_state_dict(state_dict, strict=False)
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        if self.freeze_bn:
            for m in self.model.modules():
                if isinstance(m, torch.nn.BatchNorm2d):
                    m.eval()
        return super().training_step(batch, batch_idx, optimizer_idx=optimizer_idx)
