from pytorch_lightning.metrics import functional as FM
import torch.nn.functional as F
from .task import Task
import torch
from hydra.utils import instantiate
import torchvision
from src.data.transforms.utils import _imagenet_stats


def normalize(x):
    return x / x.norm(p=2, dim=-1, keepdim=True)


class AdversarialTransformTask(Task):
    def __init__(self, model, optimizer, agnostic_model, attacked_model, transform_mode=None,
                 mu=1e-3, mu_max=100, normalize_logits=False, jit_eval=False, **kwargs):
        super().__init__(model, optimizer, **kwargs)
        self.mu = mu
        self.mu_max = mu_max
        self.normalize_logits = normalize_logits
        self.transform_mode = transform_mode

        def _eval_model(model_def, jit=False):
            model = instantiate(model_def)
            model.eval()
            for p in model.parameters():
                p.requires_grad_(False)
            if jit:
                model = torch.jit.script(model)
            return model
        self.attacked_model = _eval_model(attacked_model, jit_eval)
        self.agnostic_model = _eval_model(agnostic_model, jit_eval)

    def output_similarity(self, output, target):
        return F.mse_loss(output, target)

    def transform(self, x):
        out = {'T(x)': self.model(x).view_as(x)}
        if self.transform_mode == 'mask':
            out['mask'] = out['T(x)'].sigmoid()
            out['T(x)'] = out['mask'] * x
        elif self.transform_mode == 'add':
            out['T(x)'] = out['T(x)'] + x
        return out

    def measure(self, agnostic_outputs, attacked_outputs, target):
        if self.normalize_logits:
            agnostic_outputs = [normalize(x) for x in agnostic_outputs]
            attacked_outputs = [normalize(x) for x in attacked_outputs]
        agnostic_diff = F.mse_loss(*agnostic_outputs)
        attacked_diff = F.mse_loss(*attacked_outputs)
        agnostic_loss = agnostic_diff.clamp(min=self.mu)
        attacked_loss = -attacked_diff.clamp(max=self.mu_max)
        with torch.no_grad():
            agnostic_accuracy = FM.accuracy(agnostic_outputs[1].softmax(-1), target)
            agnostic_Tx_accuracy = FM.accuracy(agnostic_outputs[0].softmax(-1), target)
            attacked_accuracy = FM.accuracy(attacked_outputs[1].softmax(-1), target)
            attacked_Tx_accuracy = FM.accuracy(attacked_outputs[0].softmax(-1), target)

        return {'loss/loss': agnostic_loss + attacked_loss,
                'loss/agnostic': agnostic_loss,
                'loss/attacked': attacked_loss,
                'diff/agnostic': agnostic_diff,
                'diff/attacked': attacked_diff,
                'accuracy/agnostic': agnostic_accuracy,
                'accuracy/attacked': attacked_accuracy,
                'accuracy/agnostic-T(x)': agnostic_Tx_accuracy,
                'accuracy/attacked-T(x)': attacked_Tx_accuracy}

    def log_image(self, x, name='x', normalize=False, scale_each=False, denormalize_imagenet=False):
        if self.global_step % 100 == 0:
            mean = _imagenet_stats['mean'].to(x.device).view(1, -1, 1, 1)
            std = _imagenet_stats['std'].to(x.device).view(1, -1, 1, 1)
            x = x * std + mean
            grid = torchvision.utils.make_grid(x, normalize=normalize, scale_each=scale_each)
            for exp in self.logger.experiment:
                if isinstance(exp, torch.utils.tensorboard.writer.SummaryWriter):
                    exp.add_image(name, grid, self.global_step)

    def step(self, batch, batch_idx, phase='train'):
        x, y = batch
        self.attacked_model.eval()
        self.agnostic_model.eval()
        with torch.no_grad():
            out_attacked_x = self.attacked_model(x)
            out_agnostic_x = self.agnostic_model(x)
        transform_output = self.transform(x)
        T_x = transform_output.pop('T(x)')
        agnostic_outputs = (self.agnostic_model(T_x), out_agnostic_x)
        attacked_outputs = (self.attacked_model(T_x), out_attacked_x)
        metrics = self.measure(agnostic_outputs, attacked_outputs, target=y)

        self.log_image(x, f'images-{phase}/x', denormalize_imagenet=True)
        self.log_image(T_x, f'images-{phase}/T(x)', denormalize_imagenet=True)
        for k, v in transform_output.items():
            self.log_image(v, f'images-{phase}/{k}', normalize=True)
        metrics['diff/image'] = (x - T_x.detach()).pow(2).mean()
        return metrics

    def training_step(self, batch, batch_idx):
        self.model.train()
        self.log_lr()
        metrics = self.step(batch, batch_idx)
        self.log_dict({f'train-{k}': v for k, v in metrics.items()},
                      prog_bar=True, on_epoch=True, on_step=True)
        return metrics['loss/loss']

    def validation_step(self, batch, batch_idx):
        self.model.eval()
        metrics = self.step(batch, batch_idx, phase='val')
        self.log_dict({f'val-{k}': v for k, v in metrics.items()})
        return metrics['loss/loss']
