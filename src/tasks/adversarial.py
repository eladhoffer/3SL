from torchmetrics import functional as FM
import torch.nn.functional as F
from .task import Task
import torch
from hydra.utils import instantiate
import torchvision
from src.data.transforms.utils import _imagenet_stats
from src.utils_pt.misc import torch_dtypes
from src.utils_pt.cross_entropy import cross_entropy


def normalize(x):
    return x / x.norm(p=2, dim=-1, keepdim=True)


class AdversarialTransformTask(Task):
    def __init__(self, model, optimizer, agnostic_model, attacked_model, transform_mode=None, criterion='mse',
                 mu=0.0, mu_max=float('inf'), project_diff_eps=None, T=1.0, normalize_logits=False, jit_eval=False,
                 agnostic_dtype=None, attacked_dtype=None, image_stats=_imagenet_stats, **kwargs):
        super().__init__(model, optimizer, **kwargs)
        self.mu = mu
        self.mu_max = mu_max
        self.T = T
        self.image_stats = image_stats
        self.project_diff_eps = project_diff_eps
        self.criterion = criterion
        self.normalize_logits = normalize_logits
        self.transform_mode = transform_mode
        self.agnostic_dtype = torch_dtypes.get(agnostic_dtype, None)
        self.attacked_dtype = torch_dtypes.get(attacked_dtype, None)

        def _eval_model(model_def, dtype=torch.float, jit=False):
            model = instantiate(model_def).to(dtype=dtype)
            model.eval()
            for p in model.parameters():
                p.requires_grad_(False)
            if jit:
                model = torch.jit.script(model)
            return model
        self.attacked_model = _eval_model(attacked_model, dtype=self.attacked_dtype, jit=jit_eval)
        self.agnostic_model = _eval_model(agnostic_model, dtype=self.agnostic_dtype, jit=jit_eval)

    def output_similarity(self, output, target):
        return F.mse_loss(output, target)

    def transform(self, x):
        out = {}
        model_out = self.model(x)
        if self.transform_mode == 'mask':
            out['mask'] = model_out.view_as(x).sigmoid()
            out['T(x)'] = out['mask'] * x
        elif self.transform_mode == 'add':
            out['T(x)'] = model_out + x
        elif self.transform_mode == 'interpolate':
            if isinstance(model_out, dict):
                image = model_out['image'].sigmoid()
                mask = model_out['mask'].sigmoid()
                mean = self.image_stats['mean'].to(x.device).view(1, -1, 1, 1)
                std = self.image_stats['std'].to(x.device).view(1, -1, 1, 1)
                out['replaced'] = (image - mean) / std
                out['mask'] = mask.view(-1, 1, 1, 1)
                out['T(x)'] = out['mask'] * x + (1. - out['mask']) * out['replaced']

            else:
                model_out = model_out.sigmoid()
                model_out = model_out.view(x.size(0), 2, *x.shape[1:])
                out['mask'] = model_out[:, 0]
                out['replaced'] = model_out[:, 1]
                mean = self.image_stats['mean'].to(x.device).view(1, -1, 1, 1)
                std = self.image_stats['std'].to(x.device).view(1, -1, 1, 1)
                out['replaced'] = (out['replaced'] - mean) / std
                out['T(x)'] = out['mask'] * x + (1. - out['mask']) * out['replaced']
        if self.project_diff_eps is not None:
            diff_scale = (out['T(x)'] - x).pow(2).mean()
            proj_scale = diff_scale.clamp(max=self.project_diff_eps)
            out['T(x)'] = (out['T(x)'] - x) * (proj_scale / diff_scale) + x
        return out

    def mse_loss(self, agnostic_outputs, attacked_outputs):
        agnostic_diff = F.mse_loss(*agnostic_outputs)
        attacked_diff = F.mse_loss(*attacked_outputs)
        agnostic_loss = agnostic_diff.clamp(min=self.mu)
        attacked_loss = 1.0 - attacked_diff.clamp(max=self.mu_max)
        return {'loss/loss': agnostic_loss + attacked_loss,
                'loss/agnostic': agnostic_loss,
                'loss/attacked': attacked_loss,
                'diff/agnostic': agnostic_diff,
                'diff/attacked': attacked_diff}

    def kl_loss(self, agnostic_outputs, attacked_outputs):
        def _kl(x, y):
            x = (x / self.T).log_softmax(-1)
            y = (y / self.T).log_softmax(-1)
            return F.kl_div(x, y, log_target=True, reduction='batchmean')

        agnostic_loss = (_kl(*agnostic_outputs) - self.mu).clamp(min=0.)
        attacked_loss = (self.mu_max - _kl(*attacked_outputs)).clamp(min=0.)
        return {'loss/loss': agnostic_loss + attacked_loss,
                'loss/agnostic': agnostic_loss,
                'loss/attacked': attacked_loss}

    def labeled_ce_loss(self, agnostic_outputs, attacked_outputs, target):
        agnostic_loss = cross_entropy(agnostic_outputs[0], target).clamp(min=self.mu)
        attacked_loss = -cross_entropy(attacked_outputs[0], target).clamp(max=self.mu_max)
        return {'loss/loss': agnostic_loss + attacked_loss,
                'loss/agnostic': agnostic_loss,
                'loss/attacked': attacked_loss
                }

    def measure(self, agnostic_outputs, attacked_outputs, target):
        agnostic_outputs = [x.float() for x in agnostic_outputs]
        attacked_outputs = [x.float() for x in attacked_outputs]
        if self.normalize_logits:
            agnostic_outputs = [normalize(x) for x in agnostic_outputs]
            attacked_outputs = [normalize(x) for x in attacked_outputs]
        if self.criterion == 'mse':
            outputs = self.mse_loss(agnostic_outputs, attacked_outputs)
        elif self.criterion == 'kl':
            outputs = self.kl_loss(agnostic_outputs, attacked_outputs)
        elif self.criterion == 'labels':
            outputs = self.labeled_ce_loss(agnostic_outputs, attacked_outputs, target)
        with torch.no_grad():
            agnostic_accuracy = FM.accuracy(agnostic_outputs[1].softmax(-1), target)
            agnostic_Tx_accuracy = FM.accuracy(agnostic_outputs[0].softmax(-1), target)
            attacked_accuracy = FM.accuracy(attacked_outputs[1].softmax(-1), target)
            attacked_Tx_accuracy = FM.accuracy(attacked_outputs[0].softmax(-1), target)

        outputs.update({
            'accuracy/agnostic': agnostic_accuracy,
            'accuracy/attacked': attacked_accuracy,
            'accuracy/agnostic-T(x)': agnostic_Tx_accuracy,
            'accuracy/attacked-T(x)': attacked_Tx_accuracy,
            'accuracy/agnostic-diff': agnostic_accuracy - agnostic_Tx_accuracy,
            'accuracy/attacked-diff': attacked_accuracy - attacked_Tx_accuracy})
        return outputs

    def log_image(self, x, name='x', normalize=False, scale_each=False, denormalize=False):
        if self.global_step % 100 == 0:
            if denormalize:
                mean = self.image_stats['mean'].to(x.device).view(1, -1, 1, 1)
                std = self.image_stats['std'].to(x.device).view(1, -1, 1, 1)
                x = x * std + mean
            grid = torchvision.utils.make_grid(x, normalize=normalize, scale_each=scale_each)
            for exp in self.logger.experiment:
                if isinstance(exp, torch.utils.tensorboard.writer.SummaryWriter):
                    exp.add_image(name, grid, self.global_step)

    # def on_train_start(self) -> None:
    #     self.attacked_model = self.attacked_model.to(dtype=self.attacked_dtype)
    #     self.agnostic_model = self.agnostic_model.to(dtype=self.agnostic_dtype)
    #     return super().on_train_start()

    def step(self, batch, batch_idx, phase='train'):
        x, y = batch
        self.attacked_model.eval()
        self.agnostic_model.eval()
        with torch.no_grad():
            out_attacked_x = self.attacked_model(x.to(dtype=self.attacked_dtype))
            out_agnostic_x = self.agnostic_model(x.to(dtype=self.agnostic_dtype))
        transform_output = self.transform(x)
        T_x = transform_output.pop('T(x)')
        agnostic_outputs = (self.agnostic_model(T_x.to(dtype=self.agnostic_dtype)), out_agnostic_x)
        attacked_outputs = (self.attacked_model(T_x.to(dtype=self.attacked_dtype)), out_attacked_x)
        metrics = self.measure(agnostic_outputs, attacked_outputs, target=y)

        self.log_image(x, f'images-{phase}/x', denormalize=True)
        self.log_image(T_x, f'images-{phase}/T(x)', denormalize=True)
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
