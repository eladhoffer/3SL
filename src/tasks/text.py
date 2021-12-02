from .supervised import ClassificationTask
from torchmetrics import functional as FM
import torch.nn.functional as F
from src.utils_pt.cross_entropy import cross_entropy
import torch
import torch.nn as nn
import math


class MaskedLanguageModelTask(ClassificationTask):
    """
    A task for training a masked language model.
    """

    def __init__(self, model, optimizer, model_type="huggingface",
                 **kwargs):
        super().__init__(model, optimizer, **kwargs)
        self.model_type = model_type
        # self.automatic_optimization = False

    def step(self, batch, model=None):
        if model is None:
            model = self.model
        masked_tokens = batch['labels'].ge(0)
        num_tokens = masked_tokens.int().sum()
        y = batch['labels'][masked_tokens]
        lengths = batch['attention_mask'].int().sum(-1)
        if self.model_type == "fairseq":
            x = {
                'src_tokens': batch['input_ids'],
                'src_lengths': lengths,
                'masked_tokens': masked_tokens
            }
            y_hat = model(**x)[0]
        elif self.model_type == "huggingface":
            x = {ikey: batch[ikey] for ikey in ['input_ids', 'attention_mask']}
            optimized_model = hasattr(model, 'lm_head') \
                and hasattr(model.lm_head, 'set_masked_tokens')
            if optimized_model:
                model.lm_head.set_masked_tokens(masked_tokens)
                y_hat = model(**x)['logits']
            else:
                y_hat = model(**x)['logits'][masked_tokens, :]

        return {'loss': self.loss(y_hat, y),
                'num_tokens': num_tokens,
                'length': lengths.float().mean()}

    def loss(self, output, target, reduction='mean'):
        output = F.log_softmax(output, dim=-1, dtype=torch.float)
        loss = cross_entropy(output, target, reduction=reduction,
                             smooth_eps=self.label_smoothing, from_logits=False)
        return loss

    def log_lr(self, **kwargs):
        kwargs.setdefault('prog_bar', True)
        return super().log_lr(**kwargs)

    def log_metrics(self, output, phase='train', **kwargs):
        with torch.no_grad():
            loss = output['loss']
            nll = loss / math.log(2)
            self.log_dict({
                f'loss/{phase}': loss,
                f'nll/{phase}': nll,
                f'ppl/{phase}': 2. ** nll,
                f'num_tokens/{phase}': output['num_tokens'],
                f'length/{phase}': output['length']},
                **kwargs)

    def training_step(self, batch, batch_idx):
        self.model.train()
        output = self.step(batch)
        self.log_lr(on_step=True)
        self.log_metrics(output, phase='train',
                         prog_bar=True, on_step=True)
        if self.use_sam:
            eps_w = self.sam_step(output['loss'])
            loss_w_sam = self.step(batch)['loss']
            # revert eps_w
            torch._foreach_sub_(list(self.parameters()), eps_w)
            self.manual_step(loss_w_sam)
        return output

    def training_step_end(self, output):
        self.update_ema()
        return output

    def evaluation_step(self, batch, batch_idx, phase='val'):
        model = getattr(self, '_model_ema', self.model)
        model.eval()
        output = self.step(batch, model=model)
        self.log_metrics(output, phase=phase)
        return output

    def validation_step(self, batch, batch_idx):
        return self.evaluation_step(batch, batch_idx, phase='val')

    def test_step(self, batch, batch_idx):
        return self.evaluation_step(batch, batch_idx, phase='test')
