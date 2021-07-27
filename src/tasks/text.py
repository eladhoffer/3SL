from .supervised import ClassificationTask
from pytorch_lightning.metrics import functional as FM
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
            if not optimized_model:
                y_hat = y_hat[masked_tokens, :]

        return {'loss': self.loss(y_hat, y, reduction='sum'),
                'num_tokens': num_tokens,
                'length': lengths.mean()}

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
            num_tokens = output['num_tokens']
            length = output['length'].mean()
            avg_loss = (loss / num_tokens) / math.log(2)
            self.log_dict({
                f'loss/{phase}': avg_loss,
                f'ppl/{phase}': 2. ** avg_loss,
                f'num_tokens/{phase}': num_tokens,
                f'length/{phase}': length},
                **kwargs)

    def training_step(self, batch, batch_idx):
        output = self.step(batch)
        self.log_lr(on_step=True)
        if self.use_sam:
            eps_w = self.sam_step(output['loss'])
            loss_w_sam = self.step(batch)['loss']
            # revert eps_w
            torch._foreach_sub_(list(self.parameters()), eps_w)
            self.manual_step(loss_w_sam)
        return output

    def training_step_end(self, output):
        output['loss'] = output['loss'].sum()
        output['num_tokens'] = output['num_tokens'].sum()
        self.log_metrics(output, phase='train',
                         prog_bar=True, on_step=True)
        self.update_ema()
        return output

    def evaluation_step(self, batch, batch_idx):
        model = getattr(self, '_model_ema', self.model)
        model.eval()
        return self.step(batch, model=model)

    def validation_step(self, batch, batch_idx):
        return self.evaluation_step(batch, batch_idx)

    def validation_step_end(self, output):
        output['loss'] = output['loss'].sum()
        output['num_tokens'] = output['num_tokens'].sum()
        self.log_metrics(output, phase='val')
        return output

    def test_step(self, batch, batch_idx):
        return self.evaluation_step(batch, batch_idx)

    def test_step_end(self, output):
        output['loss'] = output['loss'].sum()
        output['num_tokens'] = output['num_tokens'].sum()
        self.log_metrics(output, phase='val')
        return output
