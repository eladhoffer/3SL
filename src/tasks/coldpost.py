from .mixmatch import interleave, MixUp, MixMatchTask
from pytorch_lightning.metrics import functional as FM
import torch
import torch.nn.functional as F
from utils.cross_entropy import cross_entropy
from utils.misc import no_bn_update


class ColdPostTask(MixMatchTask):

    def __init__(self, model, regime, q=[0.5, 0.5], anneal_T=None, min_T=0., **kwargs):
        super().__init__(model, regime, **kwargs)
        self.q = q
        self.min_T = min_T
        self.anneal_T = anneal_T

    def semisup_loss(self, outputs, target, unsup_outputs=None, unsup_target=None, _real_target=None):
        if self.normalize_logits:
            outputs = outputs * (self.model.temp_bias /
                                 outputs.norm(2, dim=-1, keepdim=True))
            if unsup_outputs is not None:
                unsup_outputs = unsup_outputs * \
                    (self.model.temp_bias / unsup_outputs.norm(2, dim=-1, keepdim=True))

        sup_loss = cross_entropy(outputs, target)
        if unsup_outputs is None:
            return sup_loss

        probs = torch.softmax(unsup_outputs, dim=-1)
        # TODO: weight by q (currently defaults to uniform)
        probs_weighted = probs.mean(1)
        # probs_weighted = torch.einsum('brk,r->bk', probs,
        #   probs.new_tensor(self.q))
        T = self.T
        if self.anneal_T is not None:
            T *= 1. - float(self.global_step) / self.anneal_T
        T = max(T, self.min_T)
        if T == 0:
            unsup_loss = cross_entropy(
                probs_weighted, probs_weighted.argmax(-1))
        else:
            probs_T = probs_weighted.pow(1. / T).sum(dim=-1)
            # print(f'probs={float(probs_T.mean())}')
            unsup_loss = -T * probs_T.log().mean()
        # print(f'loss={float(unsup_loss)}')
        logs = {
            'unsup/unsup_loss': unsup_loss,
            'unsup/sup_loss': sup_loss,
            'unsup/T': T
        }

        # Additional (not required logs)
        temp_weak = unsup_outputs[:, 0].norm(dim=-1).mean()
        temp_strong = unsup_outputs[:, 1].norm(dim=-1).mean()
        logs.update({'logits_norm/weak': temp_weak,
                     'logits_norm/strong': temp_strong
                     })
        # if _real_target is not None:
        #     logs.update({
        #         'unsup/acc-weak': FM.accuracy(probs[:, 0], _real_target),
        #         'unsup/acc-strong': FM.accuracy(probs[:, 1], _real_target)})

        self.log_dict(logs, on_epoch=True, on_step=False)

        lam_u = self.lam_u * min(self.global_step / self.ramp_lam_u, 1.)

        return sup_loss + lam_u * unsup_loss

    def create_target(self, unlabeled):
        if self.normalize_logits:
            unlabeled = unlabeled * (self.model.temp_bias /
                                     unlabeled.norm(2, dim=-1, keepdim=True))
        B, R = unlabeled.shape[:2]
        with no_bn_update(self.model):
            output = self.model(unlabeled.flatten(0, 1))
        logits = output.view(B, R, -1)

        probs = F.softmax(logits, dim=-1)
        target = probs.mean(1)
        T = 0.5
        target = target.pow(1. / T)
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
            with no_bn_update(self.model):
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
