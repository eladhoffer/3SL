from copy import deepcopy
import itertools
import random
import torch
import torch.nn.functional as F
import torch.nn as nn
from hydra.utils import instantiate
from time import time
from src.tasks.text import LanguageModelTask


class CompressLMStateTask(LanguageModelTask):
    """
    A task for compressing state of a transformer based language model.
    """

    def __init__(self, model, optimizer, pretrained_model, time_slice=512, **kwargs):
        super().__init__(model, optimizer, **kwargs)
        pretrained_model = instantiate(pretrained_model)
        for p in pretrained_model.parameters():
            p.requires_grad = False
        if self.compile_model:
            self.pretrained_model = torch.compile(pretrained_model)
        elif self.jit_model:
            self.pretrained_model = torch.jit.script(pretrained_model)
        else:
            self.pretrained_model = pretrained_model
        self.time_slice = time_slice

    def state_dict(self, *kargs, **kwargs):
        return self.model.state_dict(*kargs, **kwargs)

    def load_state_dict(self, state_dict, strict: bool = True):
        return self.model.load_state_dict(state_dict, strict)

    def prepare_batch(self, batch):
        x, y = batch
        with torch.no_grad():
            output = self.pretrained_model(x[:, :self.time_slice], use_cache=True)
            state = output.past_key_values
        sliced_x = x[:, self.time_slice:]
        sliced_y = y[:, self.time_slice:]
        return sliced_x, sliced_y, state

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        self.model.train()
        x, y, state = self.prepare_batch(batch)
        compressed_state = self.model(state)
        y_hat = self.pretrained_model(x, past_key_values=compressed_state)
        loss = self.loss(y_hat, y)
        metrics = self.metrics(output=y_hat, target=y, loss=loss)
        if self.use_sam:
            metrics['sam_loss'] = self.sam_update(x, y, loss)
        self.log_phase_dict(metrics, prog_bar=True, on_epoch=False, on_step=True)
        self.log_lr(on_step=True)
        return loss

    def evaluation_step(self, batch, batch_idx, phase='val'):
        model = getattr(self, '_model_ema', self.model)
        model.eval()
        x, y, state = self.prepare_batch(batch)
        y_hat = self.pretrained_model(x, past_key_values=state)
        loss = self.loss(y_hat, y)
        metrics = self.metrics(y_hat, y, loss=loss)
        self.log_phase_dict(metrics, phase=f'{phase}(baseline)')

        y_hat = self.pretrained_model(x)
        loss = self.loss(y_hat, y)
        metrics = self.metrics(y_hat, y, loss=loss)
        self.log_phase_dict(metrics, phase=f'{phase}(no-state)')

        compressed_state = model(state)
        y_hat = self.pretrained_model(x, past_key_values=compressed_state)
        loss = self.loss(y_hat, y)
        metrics = self.metrics(y_hat, y, loss=loss)
        self.log_phase_dict(metrics, phase=phase)
        return loss


class CompressStateTask(LanguageModelTask):
    """
    A task for compressing state of a transformer based language model.
    """

    def __init__(self, model, optimizer, **kwargs):
        super().__init__(model, optimizer, **kwargs)
        self._compression_model = self.model
        self.pretrained_model = self.model.pretrained_model
        for p in self.pretrained_model.parameters():
            p.requires_grad = False
        self.model = self.model.compressors
        for p in self.model.parameters():
            p.requires_grad = True
        if self.compile_model:
            self.pretrained_model = torch.compile(self.pretrained_model)
        elif self.jit_model:
            self.pretrained_model = torch.jit.script(self.pretrained_model)
        else:
            self.pretrained_model = self.pretrained_model

    def state_dict(self, *kargs, **kwargs):
        return self.model.state_dict(*kargs, **kwargs)

    def load_state_dict(self, state_dict, strict: bool = True):
        return self.model.load_state_dict(state_dict, strict)

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        self.model.train()
        x, y = self.prepare_batch(batch)
        y_hat = self.pretrained_model(x)
        loss = self.loss(y_hat, y)
        metrics = self.metrics(output=y_hat, target=y, loss=loss)
        if self.use_sam:
            metrics['sam_loss'] = self.sam_update(x, y, loss)
        self.log_phase_dict(metrics, prog_bar=True, on_epoch=False, on_step=True)
        self.log_lr(on_step=True)
        return loss

    def evaluation_step(self, batch, batch_idx, phase='val'):
        model = getattr(self, '_model_ema', self.pretrained_model)
        x, y = batch
        model.eval()
        y_hat = model(x)
        loss = self.loss(y_hat, y)
        metrics = self.metrics(y_hat, y, loss=loss)
        self.log_phase_dict(metrics, phase=phase)
        return loss


def _recursive_mse(x, y):
    if isinstance(x, torch.Tensor):
        return F.mse_loss(x, y, reduction='mean')
    else:
        return sum(_recursive_mse(xi, yi) for xi, yi in zip(x, y)) / len(x)


class PrefixCompressStateTask(CompressStateTask):
    """
    A task for compressing state of a transformer based language model.
    """

    def __init__(self, model, optimizer, prefix_length=256, compression_steps=1,
                 use_all_steps=True, objective='original', state_type='past_key_values', **kwargs):
        super().__init__(model, optimizer, **kwargs)
        self.compress = self._compression_model.forward_fn
        del self._compression_model
        self._prefix_length_range = list(prefix_length)
        self.compression_steps = compression_steps
        self.objective = objective
        self.use_all_steps = use_all_steps
        self.state_type = state_type

    def prefix_length(self):
        if len(self._prefix_length_range) == 1:
            return self._prefix_length_range[0]
        return random.randint(*self._prefix_length_range)

    def kl_div(self, output, target):
        return F.kl_div(F.log_softmax(output.logits, dim=-1),
                        F.log_softmax(target.logits, dim=-1),
                        reduction='batchmean',
                        log_target=True)

    def loss(self, output, target, state=None, compressed_state=None):
        if self.objective == 'original':
            return super().loss(output, target)
        elif self.objective == 'distill_mse':
            return F.mse_loss(output.logits, target.logits)
        elif self.objective == 'state_mse':
            return _recursive_mse(state, compressed_state)
        elif self.objective == 'distill':
            return self.kl_div(output, target)

    def prepare_batch(self, batch):
        x, y = batch
        prefix_length = self.prefix_length()
        with torch.no_grad():
            if self.state_type == 'past_key_values':
                output = self.pretrained_model(x[:, :prefix_length], use_cache=True)
                state = output.past_key_values
            elif self.state_type == 'inputs_embeds':
                state = self.pretrained_model.transformer.wte(x[:, :prefix_length])
            elif self.state_type == 'hidden_states':
                output = self.pretrained_model(x[:, :prefix_length], output_hidden_states=True)
                state = output.hidden_states[-1]
        x = x[:, prefix_length:]
        position_ids = prefix_length + torch.arange(x.size(-1)).unsqueeze(0).to(x.device)

        if self.objective == 'original':
            y = y[:, prefix_length:]
        elif 'distill' in self.objective or self.objective == 'state_mse':
            with torch.no_grad():
                y = self.forward_state(x, state, position_ids=position_ids)
        return x, y, state, position_ids

    def forward_state(self, x, state, **kwargs):
        if self.state_type == 'past_key_values':
            return self.pretrained_model(x, past_key_values=state, **kwargs)
        elif self.state_type == 'inputs_embeds':
            output = self.pretrained_model(inputs_embeds=state, use_cache=True)
            return self.pretrained_model(x, past_key_values=output.past_key_values, **kwargs)
        elif self.state_type == 'hidden_states':
            output = self.pretrained_model(inputs_embeds=state, use_cache=True)
            return self.pretrained_model(x, past_key_values=output.past_key_values, **kwargs)
        else:
            raise ValueError(f'Unknown state type {self.state_type}')

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        self.model.train()
        x, y, state, position_ids = self.prepare_batch(batch)
        loss = 0
        for i in range(self.compression_steps):
            cstate = self.compress(self.model, state)
            if self.use_all_steps or i == self.compression_steps - 1:
                if self.objective == 'state_mse':
                    y_hat = None
                else:
                    y_hat = self.forward_state(x, cstate,
                                               position_ids=position_ids)
                loss += self.loss(y_hat, y, state, cstate)
        metrics = self.metrics(output=y_hat, target=y, loss=loss)
        if self.use_sam:
            metrics['sam_loss'] = self.sam_update(x, y, loss)
        self.log_phase_dict(metrics, prog_bar=True, on_epoch=False, on_step=True)
        self.log_lr(on_step=True)
        return loss

    def evaluation_step(self, batch, batch_idx, phase='val'):
        model = getattr(self, '_model_ema', self.model)
        model.eval()
        x, y, state, position_ids = self.prepare_batch(batch)
        # y_hat = self.pretrained_model(x, position_ids=position_ids,
        #                               past_key_values=state)
        # loss = self.loss(y_hat, y)
        # metrics = self.metrics(y_hat, y, loss=loss)
        # self.log_phase_dict(metrics, phase=f'{phase}(baseline)')

        # y_hat = self.pretrained_model(x, position_ids=position_ids)
        # loss = self.loss(y_hat, y)
        # metrics = self.metrics(y_hat, y, loss=loss)
        # self.log_phase_dict(metrics, phase=f'{phase}(no-state)')

        cstate = self.compress(model, state,
                               compression_steps=self.compression_steps)
        y_hat = self.forward_state(x, cstate, position_ids=position_ids)
        loss = self.loss(y_hat, y, state, cstate)
        metrics = self.metrics(y_hat, y, loss=loss)
        if self.objective == 'state_mse':
            metrics['kl_div'] = self.kl_div(y_hat, y)
        self.log_phase_dict(metrics, phase=phase)
        return loss
