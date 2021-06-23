from .supervised import ClassificationTask
from pytorch_lightning.metrics import functional as FM
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils_pt.cross_entropy import cross_entropy
from .utils import pack_inputs, unpack_outputs


class SemiSupTask(ClassificationTask):

    def __init__(self, model, optimizer, **kwargs):
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
            inputs, shapes = pack_inputs(labeled, unlabeled)
            output = self.model(inputs)
            output, output_unsup = unpack_outputs(output, shapes)
        loss = self.semisup_loss(output, target, output_unsup, _unlab_target)
        if output.dim() > 2:  # batch augmented
            output = output.mean(1)
        acc = FM.accuracy(output.softmax(-1), target)
        if self.optimizer_regime is not None:
            self.log('lr', self.optimizer_regime.get_lr()[0], on_step=True)
        self.log_dict({
            'loss/train': loss,
            'accuracy/train': acc}, prog_bar=True, on_epoch=True, on_step=True)
        return loss


class FixMatchTask(SemiSupTask):

    def __init__(self, model, optimizer, num_views=2, lam_u=1, tau=0.95, q=[1., 0.],
                 normalize_logits=False, soft_target=False, **kwargs):
        super().__init__(model, optimizer, **kwargs)
        assert len(q) == num_views
        self.lam_u = lam_u
        self.tau = tau
        self.num_views = num_views
        self.q = q
        self.normalize_logits = normalize_logits
        self.soft_target = soft_target
        if normalize_logits:
            self.model.temp_bias = nn.Parameter(torch.tensor([1.]))

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
                        cross_entropy(outputs[:, r], target)
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
                'unsup/acc-weak': FM.accuracy(probs[:, 0], _real_target),
                'unsup/acc-strong': FM.accuracy(probs[:, 1], _real_target)})
        self.log_dict(logs, on_epoch=True, on_step=False)
        return sup_loss + self.lam_u * unsup_loss
