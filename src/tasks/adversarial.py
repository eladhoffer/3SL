from pytorch_lightning.metrics import functional as FM
import torch.nn.functional as F
from .task import Task
import torch
from hydra.utils import instantiate
import torchvision
from src.data.transforms.utils import _imagenet_stats


class AdversarialTransformTask(Task):
    def __init__(self, model, optimizer, agnostic_model, attacked_model,
                 mu=1e-3, mu_max=100, jit_eval=False, **kwargs):
        super().__init__(model, optimizer, **kwargs)
        self.mu = mu
        self.mu_max = mu_max

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

    def measure(self, agnostic_outputs, attacked_outputs, target):
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

    def log_image(self, x, name='x', normalize=False, scale_each=False, denormalize_imagenet=True):
        if self.global_step % 100 == 0:
            mean = _imagenet_stats['mean'].to(x.device).view(1, -1, 1, 1)
            std = _imagenet_stats['std'].to(x.device).view(1, -1, 1, 1)
            x = x * std + mean
            grid = torchvision.utils.make_grid(x, normalize=normalize, scale_each=scale_each)
            for exp in self.logger.experiment:
                if isinstance(exp, torch.utils.tensorboard.writer.SummaryWriter):
                    exp.add_image(f'images/{name}', grid, self.global_step)

    def training_step(self, batch, batch_idx):
        x, y = batch
        self.attacked_model.eval()
        self.agnostic_model.eval()
        self.model.train()
        with torch.no_grad():
            out_attacked_x = self.attacked_model(x)
            out_agnostic_x = self.agnostic_model(x)
        T_x = self.model(x)
        agnostic_outputs = (self.agnostic_model(T_x), out_agnostic_x)
        attacked_outputs = (self.attacked_model(T_x), out_attacked_x)
        metrics = self.measure(agnostic_outputs, attacked_outputs, target=y)
        image_diff = x - T_x
        with torch.no_grad():
            metrics['diff/image'] = image_diff.pow(2).mean()
            self.log_lr()
            self.log_image(x)
            self.log_image(x, 'T(x)')
            self.log_image(image_diff, 'x-T(x)', normalize=True, denormalize_imagenet=False)
            self.log_dict(metrics, prog_bar=True, on_epoch=False, on_step=True)
        return metrics['loss/loss']

