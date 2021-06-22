from .supervised import ClassificationTask
from pytorch_lightning.metrics import functional as FM
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils_pt.cross_entropy import cross_entropy
from src.utils_pt.mixup import MixUp as _MixUp


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

    def __init__(self, model, optimizer, **kwargs):
        # self.mixup_alpha = kwargs.pop('mixup', 0)
        # if self.mixup_alpha > 0:
        #     self.mixup_labeled = MixUp()
        #     self.mixup_unlabeled = MixUp()
        # else:
        #     self.mixup = None

        super().__init__(model, optimizer, **kwargs)

    def semisup_loss(self, outputs, target, unsup_outputs=None, _real_target=None):
        pass

    def training_step(self, batch, batch_idx):
        labeled, target = batch.get('labeled')
        unlabeled, _unlab_target = batch.get('unlabeled', (None, None))
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

    def __init__(self, model, optimizer, num_views=2, lam_u=1, tau=0.95, q=[1., 0.],
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
        super().__init__(model, optimizer, **kwargs)

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
