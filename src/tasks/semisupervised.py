from .task import pack_inputs, unpack_outputs, MixUp
from .supervised import ClassificationTask
from pytorch_lightning.metrics import functional as FM
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.cross_entropy import cross_entropy

from utils.mixup import MixUp as _MixUp


class MixUp(_MixUp):
    def sample(self, batch_size):
        super().sample(batch_size)
        self.mix_value = max(self.mix_value, 1 - self.mix_value)


def pack_inputs(*tensors, item_num_dims=3):
    shapes = []
    tensor_list = []
    for t in tensors:
        s = t.shape[:-item_num_dims]
        if len(s) > 1:  # batch is over two dims
            t = t.view(-1, *t.shape[len(s):])
        tensor_list.append(t)
        shapes.append(s)
    return torch.cat(tensor_list), shapes


def unpack_outputs(tensor, shapes):
    idx = 0
    out = []
    for shape in shapes:
        if len(shape) == 1:
            B = shape[0]
        else:
            B = torch.tensor(shape).prod()
        out_shape = shape + tensor.shape[1:]
        out.append(tensor[idx:idx + B].view(*out_shape))
        idx += B
    return tuple(out)


class SemiSupTask(ClassificationTask):

    def __init__(self, model, regime, **kwargs):
        # self.mixup_alpha = kwargs.pop('mixup', 0)
        # if self.mixup_alpha > 0:
        #     self.mixup_labeled = MixUp()
        #     self.mixup_unlabeled = MixUp()
        # else:
        #     self.mixup = None

        super().__init__(model, regime, **kwargs)

    def semisup_loss(self, outputs, target, unsup_outputs=None, _real_target=None):
        pass

    def training_step(self, batch, batch_idx):
        self.update_regime()
        (labeled, target), (unlabeled, _unlab_target) = batch
        if unlabeled is None:
            output = self.model(labeled)
            output_unsup = None
        else:
            # if self.mixup_alpha > 0:
            #     self.mixup_labeled.sample(self.mixup_alpha, labeled.size(0))
            #     self.mixup_unlabeled.sample(self.mixup_alpha, unlabeled.size(0))
            #     labeled = self.mixup_labeled(labeled)
            #     unlabeled = self.mixup_unlabeled(unlabeled)
            inputs, shapes = pack_inputs(labeled, unlabeled)
            output = self.model(inputs)
            output, output_unsup = unpack_outputs(output, shapes)
        loss = self.semisup_loss(output, target, output_unsup, _unlab_target)
        if output.dim() > 2:  # batch augmented
            output = output.mean(1)
        acc = FM.accuracy(output, target)
        self.log('lr', self.optimizer.get_lr()[0], on_step=True)
        self.log_dict({
            'loss/train': loss,
            'accuracy/train': acc}, prog_bar=True, on_epoch=True, on_step=True)
        return loss


class FixMatchTask(SemiSupTask):

    def __init__(self, model, regime, num_views=2, lam_u=1, tau=0.95, q=[1., 0.],
                 normalize_logits=False, soft_target=False, **kwargs):
        assert len(q) == num_views
        self.lam_u = lam_u
        self.tau = tau
        self.num_views = num_views
        self.q = q
        self.normalize_logits = normalize_logits
        self.soft_target = soft_target
        if normalize_logits:
            model.temp_bias = nn.Parameter(torch.tensor([1.]))
        super().__init__(model, regime, **kwargs)

    def semisup_loss(self, outputs, target, unsup_outputs=None, _real_target=None):
        if self.normalize_logits:
            outputs = outputs * (self.model.temp_bias /
                                 outputs.norm(2, dim=-1, keepdim=True))
            if unsup_outputs is not None:
                unsup_outputs = unsup_outputs * \
                    (self.model.temp_bias / unsup_outputs.norm(2, dim=-1, keepdim=True))
        if outputs.dim() == 2:  # only one view
            sup_loss = F.cross_entropy(outputs, target)
        else:
            sup_loss = 0
            for r in range(outputs.size(1)):
                if self.q[r] > 0.:
                    sup_loss += self.q[r] * \
                        F.cross_entropy(outputs[:, r], target)
        if unsup_outputs is None:
            return sup_loss

        probs = torch.softmax(unsup_outputs, dim=-1)

        max_prob, idx_max_prob = probs.max(-1)
        if self.soft_target:
            self_target = idx_max_prob
        else:
            self_target = probs.detach()
        valid_mask = max_prob.ge(self.tau).float()
        valid_ratio = valid_mask.sum() / max_prob.nelement()

        unsup_loss = 0
        for r in range(self.num_views):
            if self.q[r] == 0.:
                continue
            for i in range(self.num_views):
                if i == r:
                    continue
                masked_ce = valid_mask[:, i] * cross_entropy(unsup_outputs[:, i], self_target[:, r],
                                                             reduction='none')
                unsup_loss += self.q[r] * masked_ce.mean()
        logs = {
            'unsup/valid_ratio': valid_ratio,
            'unsup/unsup_loss': unsup_loss,
            'unsup/sup_loss': sup_loss,
        }

        # Additional (not required logs)
        temp_weak = unsup_outputs[:, 0].norm(dim=-1).mean()
        temp_strong = unsup_outputs[:, 1].norm(dim=-1).mean()
        logs.update({'logits_norm/weak': temp_weak,
                     'logits_norm/strong': temp_strong
                     })
        if _real_target is not None:
            logs.update({
                'unsup/acc-weak': FM.accuracy(unsup_outputs[:, 0], _real_target),
                'unsup/acc-strong': FM.accuracy(unsup_outputs[:, 1], _real_target)})
        self.log_dict(logs, on_epoch=True, on_step=False)
        return sup_loss + self.lam_u * unsup_loss

    def fixup_loss(self, outputs, target, unsup_outputs=None, _real_target=None):
        if self.normalize_logits:
            outputs = outputs * (self.model.temp_bias /
                                 outputs.norm(2, dim=-1, keepdim=True))
            if unsup_outputs is not None:
                unsup_outputs = unsup_outputs * \
                    (self.model.temp_bias / unsup_outputs.norm(2, dim=-1, keepdim=True))

        sup_loss = F.cross_entropy(outputs, target)
        if unsup_outputs is None:
            return sup_loss

        unsup_probs = torch.softmax(unsup_outputs[:, 0], dim=-1)
        unsup_max_prob, unsup_self_target = unsup_probs.max(-1)
        unsup_valid = unsup_max_prob.ge(self.tau)
        num_valid = unsup_valid.int().sum()
        valid_ratio = num_valid / unsup_valid.size(0)

        if num_valid > 0:
            valid_unsup_outputs = unsup_outputs[unsup_valid]
            unsup_self_target = unsup_self_target[unsup_valid]
            unsup_loss = F.cross_entropy(valid_unsup_outputs[:, 1], unsup_self_target,
                                         reduction='sum') / unsup_valid.size(0)  # not num_valid
        else:
            unsup_loss = 0

        logs = {
            'unsup/valid_ratio': valid_ratio,
            'unsup/unsup_loss': unsup_loss,
            'unsup/sup_loss': sup_loss,
        }

        # Additional (not required logs)
        temp_weak = unsup_outputs[:, 0].norm(dim=-1).mean()
        temp_strong = unsup_outputs[:, 1].norm(dim=-1).mean()
        logs.update({'logits_norm/weak': temp_weak,
                     'logits_norm/strong': temp_strong
                     })
        if _real_target is not None:
            logs.update({
                'unsup/acc-weak': FM.accuracy(unsup_outputs[:, 0], _real_target),
                'unsup/acc-strong': FM.accuracy(unsup_outputs[:, 1], _real_target)})
        self.log_dict(logs, on_epoch=True, on_step=False)
        return sup_loss + self.lam_u * unsup_loss


class BridgeMatchTask(SemiSupTask):

    def __init__(self, model, regime, num_views=2, lam_u=1, tau=0.95, q=[1., 0.],
                 normalize_logits=False, soft_target=False, T=1., **kwargs):
        assert len(q) == num_views
        self.lam_u = lam_u
        self.tau = tau
        self.num_views = num_views
        self.q = q
        self.normalize_logits = normalize_logits
        self.soft_target = soft_target
        self.T = T
        if normalize_logits:
            model.temp_bias = nn.Parameter(torch.tensor([1.]))
            model.temp_target_bias = nn.Parameter(torch.tensor([1.0]))
        super().__init__(model, regime, **kwargs)

    # def create_avg_target(self, logits, target):
    #     B, R, C = logits.shape
    #     B_u = B-target.size(0)
    #     view0_labeled = F.one_hot(target, C)  # B_lxC
    #     fill_value = getattr(self, 'view0_target', 1./C)
    #     view0_unlabeled = logits.new_full((B_u, C), fill_value)
    #     view0 = torch.cat((view0_labeled, view0_unlabeled), dim=0)
    #     all_targets = F.softmax(logits, dim=-1)
    #     view0 = view0.unsqueeze(1).expand_as(all_targets)
    #     all_targets = all_targets.unsqueeze(
    #         2).expand(-1, -1, R, -1).contiguous()
    #     self_prob = torch.einsum('nrrc->nrc', all_targets)
    #     all_targets = (all_targets.sum(1) - self_prob + view0) / R
    #     # for r in range(R):  # todo: replace with scatter for efficiency
    #     #     all_targets[:, r, r].fill_(0.)
    #     return all_targets

    # def create_target(self, logits, target):
    #     B, R, C = logits.shape
    #     B_u = B-target.size(0)
    #     view0_labeled = F.one_hot(target, C)  # B_lxC
    #     fill_value = getattr(self, 'view0_target', 1./C)
    #     view0_unlabeled = logits.new_full((B_u, C), fill_value)
    #     view0 = torch.cat((view0_labeled, view0_unlabeled), dim=0)
    #     all_targets = F.softmax(logits, dim=-1)
    #     view0 = view0.unsqueeze(1).expand_as(all_targets)
    #     all_targets = all_targets.unsqueeze(
    #         2).expand(-1, -1, R, -1).contiguous()
    #     self_prob = torch.einsum('nrrc->nrc', all_targets)
    #     all_targets = (all_targets.sum(1) - self_prob) / (R-1)
    #     all_targets = (all_targets + view0) / 2
    #     # for r in range(R):  # todo: replace with scatter for efficiency
    #     #     all_targets[:, r, r].fill_(0.)
    #     return all_targets

    # def create_target(self, logits, target):
    #     B, R, C = logits.shape
    #     B_u = B-target.size(0)
    #     view0_labeled = F.one_hot(target, C)  # B_lxC
    #     fill_value = getattr(self, 'view0_target', 1./C)
    #     view0_unlabeled = logits.new_full((B_u, C), fill_value)
    #     view0 = torch.cat((view0_labeled, view0_unlabeled), dim=0)
    #     all_targets = F.softmax(logits, dim=-1)
    #     all_targets = torch.cat(
    #         (all_targets, view0.unsqueeze(1)), dim=1)  # Bx(R+1)xC
    #     max_probs, _ = all_targets.max(dim=-1)
    #     r_max = max_probs.argmax(dim=1)\
    #         .view(-1, 1, 1).expand(-1, 1, all_targets.size(-1))
    #     return all_targets.gather(1, r_max).squeeze(1)

    # def create_target(self, logits, target):
    #     B, R, C = logits.shape
    #     B_u = B-target.size(0)
    #     view0_labeled = F.one_hot(target, C)  # B_lxC
    #     fill_value = getattr(self, 'view0_target', 1./C)
    #     view0_unlabeled = logits.new_full((B_u, C), fill_value)
    #     view0 = torch.cat((view0_labeled, view0_unlabeled), dim=0)
    #     all_targets = F.softmax(logits, dim=-1)
    #     all_targets = torch.cat(
    #         (all_targets, view0.unsqueeze(1)), dim=1)  # Bx(R+1)xC
    #     return all_targets

    def create_target(self, logits, target):
        B, R, C = logits.shape
        B_l = target.size(0)
        probs = F.softmax(logits, dim=-1)
        view0_labeled = F.one_hot(target, C)  # B_lxC
        view0_unlabeled = probs[B_l:].mean(1).pow(1. / self.T)
        view0 = torch.cat((view0_labeled, view0_unlabeled), dim=0)
        return view0

    # def divergence(self, output, target):
    #     output = output.mean(1)
    #     return cross_entropy(output, target)

    def divergence(self, output, target):
        B, R, _ = output.shape
        target = target.view(B, 1, -1).expand(-1, R, -1).contiguous()
        target = target.flatten(0, 1)
        output = output.flatten(0, 1)
        loss = cross_entropy(output, target)
        # loss = loss.view(B, R).max(-1)[0].mean()
        return loss

    def semisup_loss(self, outputs, target, unsup_outputs=None, _real_target=None):
        logits = torch.cat((outputs, unsup_outputs), dim=0)  # (B_l+B_u)xRxC
        scale_target = 1.
        if self.normalize_logits:
            logits = logits * (self.model.temp_bias /
                               logits.norm(2, dim=-1, keepdim=True))
            scale_target = self.model.temp_target_bias / self.model.temp_bias

        # targets = self.create_target(scale_target * logits.detach(), target)
        with torch.no_grad():
            targets = self.create_target(logits.detach(), target)

        loss = self.divergence(logits, targets)
        logs = {'logits_norm/norm': logits.norm(dim=-1).mean(),
                # 'logits_norm/target': targets.norm(dim=-1).mean()
                }
        if _real_target is not None:
            logs.update({
                'unsup/acc-weak': FM.accuracy(unsup_outputs[:, 0], _real_target),
                'unsup/acc-strong': FM.accuracy(unsup_outputs[:, 1], _real_target)})
        self.log_dict(logs, on_epoch=True, on_step=False)
        return loss
