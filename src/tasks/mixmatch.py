from .task import pack_inputs, unpack_outputs, MixUp
from .semisupervised import SemiSupTask
from pytorch_lightning.metrics import functional as FM
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils_pt.cross_entropy import cross_entropy
from src.utils_pt.misc import no_bn_update


def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets


def interleave(xy, batch):
    xy = list(xy)
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]


class MixMatchTask(SemiSupTask):

    def __init__(self, model, regime, num_views=2, lam_u=75,
                 ramp_lam_u=16000, tau=0.95, q=[1., 0.], selective_bn_update=True,
                 normalize_logits=False, soft_target=False, T=0.5, mixup=0.75, **kwargs):
        assert len(q) == num_views
        self.lam_u = lam_u
        self.ramp_lam_u = ramp_lam_u
        self.tau = tau
        self.selective_bn_update = selective_bn_update
        self.num_views = num_views
        self.q = q
        self.normalize_logits = normalize_logits
        self.soft_target = soft_target
        self.T = T
        if normalize_logits:
            model.temp_bias = nn.Parameter(torch.tensor([1.]))
            model.temp_target_bias = nn.Parameter(torch.tensor([1.0]))
        super().__init__(model, regime, **kwargs)
        self.mixup_alpha = mixup

        self.mixup = MixUp(mixup)

    def create_target(self, unlabeled):
        if self.normalize_logits:
            unlabeled = unlabeled * (self.model.temp_bias /
                                     unlabeled.norm(2, dim=-1, keepdim=True))
        B, R = unlabeled.shape[:2]
        with no_bn_update(self.model, self.selective_bn_update):
            output = self.model(unlabeled.flatten(0, 1))
        logits = output.view(B, R, -1)

        probs = F.softmax(logits, dim=-1)
        target = probs.mean(1)
        target = target.pow(1. / self.T)
        target /= target.sum(-1, keepdim=True)
        return target

    def training_step(self, batch, batch_idx):
        self.update_regime()
        (labeled, target), (unlabeled, _unlab_target) = batch
        if unlabeled is None:
            output = self.model(labeled)
            output_unsup = None
        else:
            with torch.no_grad():
                # pseudo labels
                created_target = self.create_target(unlabeled)
                all_inputs = torch.cat((labeled, *unlabeled.unbind(1)))
                onehot_target = F.one_hot(target, created_target.size(-1))
                all_targets = torch.cat([onehot_target.to(dtype=created_target.dtype)] +
                                        [created_target] * unlabeled.size(1))
                # mixup
                self.mixup.sample(all_inputs.size(0))
                mixed_inputs = self.mixup(all_inputs)
                mixed_targets = self.mixup(all_targets)

                # interleave labeled and unlabed samples between batches to get correct batchnorm calculation
                B = labeled.size(0)
                mixed_inputs = list(torch.split(mixed_inputs, B))
                mixed_targets = list(torch.split(mixed_targets, B))
                mixed_inputs = interleave(mixed_inputs, B)

            outputs = [self.model(mixed_inputs[0])]
            with no_bn_update(self.model, self.selective_bn_update):
                outputs += [self.model(x) for x in mixed_inputs[1:]]
            outputs = interleave(outputs, B)
            output, output_unsup = outputs[0], outputs[1:]
            output_unsup = torch.stack(output_unsup, dim=1)
            soft_target, soft_unlabeled_target = mixed_targets[0], mixed_targets[1:]
            soft_unlabeled_target = torch.stack(soft_unlabeled_target, dim=1)
        loss = self.semisup_loss(output, soft_target,
                                 output_unsup, soft_unlabeled_target, _unlab_target)
        if output.dim() > 2:  # batch augmented
            output = output.mean(1)
        acc = FM.accuracy(output.softmax(dim=-1), target)
        self.log('lr', self.optimizer.get_lr()[0], on_step=True)
        self.log_dict({
            'loss/train': loss,
            'accuracy/train': acc}, prog_bar=True, on_epoch=True, on_step=True)
        return loss

    def semisup_loss(self, outputs, target, unsup_outputs=None, unsup_target=None, _real_target=None):
        if self.normalize_logits:
            outputs = outputs * (self.model.temp_bias /
                                 outputs.norm(2, dim=-1, keepdim=True))
            unsup_outputs = unsup_outputs * (self.model.temp_bias /
                                             unsup_outputs.norm(2, dim=-1, keepdim=True))
        lam_u = self.lam_u * min(self.global_step / self.ramp_lam_u, 1.)
        sup_loss = cross_entropy(outputs, target)
        unsup_probs = F.softmax(unsup_outputs, dim=-1)
        unsup_loss = torch.mean((unsup_probs - unsup_target)**2)
        loss = sup_loss + lam_u * unsup_loss
        logs = {'unsup/lam_u': lam_u,
                'unsup/unsup_loss': unsup_loss,
                'unsup/sup_loss': sup_loss,
                'logits_norm/norm': outputs.norm(dim=-1).mean(),
                'logits_norm/unsup_norm': unsup_outputs.norm(dim=-1).mean()
                }
        if _real_target is not None:
            logs.update({
                'unsup/acc-weak': FM.accuracy(unsup_probs[:, 0], _real_target),
                'unsup/acc-strong': FM.accuracy(unsup_probs[:, 1], _real_target)})
        self.log_dict(logs, on_epoch=True, on_step=False)
        return loss
