from .supervised import ClassificationTask
from .task import Task
from torchmetrics import functional as FM
import torch.nn.functional as F
from src.utils_pt.cross_entropy import cross_entropy
import torch
import torch.nn as nn
import math
from hydra.utils import instantiate
from time import time


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
        step_time0 = time()
        output = self.step(batch)
        self.log_lr(on_step=True)
        output = {'step_time': time() - step_time0}
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


class ImageToTextTask(ClassificationTask):
    """
    A task for training an image-to-language model.
    """

    def __init__(self, model, optimizer, **kwargs):
        super().__init__(model, optimizer, **kwargs)

    def metrics(self, output, target):
        # acc = FM.accuracy(output.softmax(dim=-1), target)
        # return {'accuracy': acc}
        return {}

    def _prepare_batch(self, batch, pad_token_id=0):
        text = batch.pop('text')
        labels = text.input_ids.clone()
        input_ids = text.input_ids
        labels[labels == pad_token_id] = -100
        input_ids[input_ids == pad_token_id] = 0
        batch['labels'] = labels
        batch['input_ids'] = input_ids
        return batch

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        batch = self._prepare_batch(batch)
        output = self.model(**batch)
        loss = output.loss
        self.log_lr(on_step=True)
        metrics = self.metrics(output, batch)
        metrics['loss'] = loss
        self.log_dict({f'{k}/train': v for k, v in metrics.items()},
                      prog_bar=True, on_epoch=True, on_step=True)
        if self.use_sam:
            eps_w = self.sam_step(loss)
            loss_w_sam = self.model(**batch).loss
            # revert eps_w
            torch._foreach_sub_(list(self.parameters()), eps_w)
            self.manual_step(loss_w_sam)
        return loss

    def evaluation_step(self, batch, batch_idx):
        batch = self._prepare_batch(batch)
        model = getattr(self, '_model_ema', self.model)
        output = model(**batch)
        metrics = self.metrics(output, batch)
        metrics['loss'] = output.loss
        return metrics


def mean_pooling(model_output, attention_mask):
    # First element of model_output contains all token embeddings
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class ImageFromTextTask(ClassificationTask):
    """
    A task for training an image-to-language model.
    """

    def __init__(self, model, text_model, optimizer, **kwargs):
        super().__init__(model, optimizer, **kwargs)
        self.criterion = nn.MSELoss()
        self.text_model = instantiate(text_model)
        for p in self.text_model.parameters():
            p.requires_grad = False

    def metrics(self, output, target):
        # acc = FM.accuracy(output.softmax(dim=-1), target)
        # return {'accuracy': acc}
        return {}

    def text_embedding(self, **text_inputs):
        # Compute token embeddings
        with torch.no_grad():
            # # encoded_input = encoded_input.to(device=device, dtype=dtype)
            model_output = self.text_model(**text_inputs)
            # # Perform pooling
            # sentence_embeddings = mean_pooling(model_output, text_inputs['attention_mask'])

            sentence_embeddings = model_output['pooler_output']
            sentence_embeddings = F.layer_norm(
                sentence_embeddings, (sentence_embeddings.size(-1),))
        return sentence_embeddings

    def loss(self, output, target):
        return self.criterion(output, target).mean()

    def training_step(self, batch):
        step_time = time()
        output = self.model(batch['image'])
        model_step_time = time() - step_time
        step_time = time()
        target = self.text_embedding(**batch['text'])
        text_step_time = time() - step_time
        loss = self.loss(output, target)
        self.log_lr(on_step=True)
        output = {'step_time': time() - step_time}
        metrics = self.metrics(output, batch)
        metrics['text_step_time'] = text_step_time
        metrics['model_step_time'] = model_step_time

        metrics['loss'] = loss
        self.log_dict({f'{k}/train': v for k, v in metrics.items()},
                      prog_bar=True, on_epoch=True, on_step=True)
        if self.use_sam:
            eps_w = self.sam_step(loss)
            loss_w_sam = self.model(**batch).loss
            # revert eps_w
            torch._foreach_sub_(list(self.parameters()), eps_w)
            self.manual_step(loss_w_sam)
        return loss

    def evaluation_step(self, batch, batch_idx):
        target = self.text_embedding(**batch['text'])
        model = getattr(self, '_model_ema', self.model)
        output = model(batch['image'])
        loss = self.loss(output, target)
        metrics = self.metrics(output, batch)
        metrics['loss'] = loss
        return metrics
