import types
from typing import Optional, Tuple, Union
import torch
from torch import nn
import transformers
from transformers import AutoTokenizer
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel, GPT2Attention, GPT2Config


class CompressState(nn.Module):
    def __init__(self, compressed_length=16, seq_length=512):
        super().__init__()
        self.compressed_length = compressed_length
        self.seq_length = seq_length

    def modify_mask(self, mask=None):
        if mask is not None:
            T = min(mask.shape[3], self.compressed_length)
            mask = mask[:, :, :, -T:]
        return mask

    def join_states(self, *states):
        state = torch.cat([s for s in states if s is not None], dim=-2)
        return state

    def forward(self, state, past=None):
        state = self.join_states(past, state)
        return state, state


class NaiveLast(CompressState):
    def forward(self, state, past=None):
        state = self.join_states(past, state)
        T = min(state.shape[2], self.compressed_length)
        state = state[:, :, -T:, :]
        return state


class RecurrentState(CompressState):
    def __init__(self, rnn_type='LSTM', state_length=128, ratio=1, residual=False,
                 state_properties={'num_heads': 12, 'hidden_size': 64},
                 rnn_config={'num_layers': 1}):
        super().__init__()
        if rnn_type == 'LSTM':
            rnnf = nn.LSTM
        elif rnn_type == 'GRU':
            rnnf = nn.GRU
        elif rnn_type == 'RNN':
            rnnf = nn.RNN
        else:
            raise ValueError('Unknown rnn type: {}'.format(rnn_type))
        self.ratio = ratio
        self.state_length = state_length
        self.residual = residual
        state_size = state_properties['hidden_size'] * state_properties['num_heads']
        rnn_config.setdefault('batch_first', True)
        rnn_config.setdefault('bidirectional', False)
        rnn_config.setdefault('input_size', state_size)
        rnn_config.setdefault('hidden_size', state_size)

        def _proj(in_size, out_size):
            if in_size != out_size:
                module = nn.Linear(in_size, out_size)
            else:
                module = nn.Identity()
            return module

        self.input_proj = _proj(state_size, rnn_config['input_size'])
        self.output_proj = _proj(rnn_config['hidden_size'], state_size)
        self.rnn = rnnf(**rnn_config)

    def discard(self, state):
        if self.state_length is not None:
            state = state[:, :, -self.state_length:]
        if self.ratio > 1:
            state = state[:, :, ::self.ratio]
        return state

    def modify_mask(self, mask=None):
        # no masking for rnn induced compression
        return None

    @torch.no_grad()
    def modify_bias(self, bias):
        # bias is of shape [1, 1, seq_length, seq_length]
        # set it so that the last self.state_length are not masked
        T = bias.size(-1)
        t = torch.arange(T, device=bias.device).view(1, 1, -1, 1)
        mask = t - t.transpose(-1, -2) <= self.state_length - 1
        return bias.masked_fill_(~mask, False)

    def forward(self, state, past=None):
        # single state is (batch, num_heads, seq_length, hidden_size)
        (batch, num_heads, _, hidden_size) = state.shape
        state = state.permute(0, 2, 1, 3).flatten(2, 3)
        residual = state if self.residual else None
        state = self.input_proj(state)
        if past is not None:
            past, rnn_state = past
        else:
            rnn_state = None
        state, rnn_state = self.rnn(state, rnn_state)
        state = self.output_proj(state)
        if residual is not None:
            state = state + residual
        state = state.reshape(batch, -1, num_heads, hidden_size)
        state = state.permute(0, 2, 1, 3)
        state = self.join_states(past, state)
        present = (self.discard(state), rnn_state)
        return state, present


class AttentedState(RecurrentState):
    def __init__(self, config, state_length=128, ratio=1, residual=False):
        super().__init__()
        self.state_length = state_length
        self.residual = residual
        self.attention = GPT2Attention(config)
        self.ratio = ratio

    def modify_mask(self, mask=None):
        # no masking for rnn induced compression
        return None

    @torch.no_grad()
    def modify_bias(self, bias):
        # bias is of shape [1, 1, seq_length, seq_length]
        # set it so that the last self.state_length are not masked
        T = bias.size(-1)
        t = torch.arange(T, device=bias.device).view(1, 1, -1, 1)
        mask = t - t.transpose(-1, -2) <= self.state_length - 1
        return bias.masked_fill_(~mask, False)

    def forward(self, state, past=None):
        # single state is (batch, num_heads, seq_length, hidden_size)
        (batch, num_heads, _, hidden_size) = state.shape
        state = state.permute(0, 2, 1, 3).flatten(2, 3)
        residual = state if self.residual else None
        state = self.input_proj(state)
        if past is not None:
            past, rnn_state = past
        else:
            rnn_state = None
        state, rnn_state = self.rnn(state, rnn_state)
        state = self.output_proj(state)
        if residual is not None:
            state = state + residual
        state = self.join_states(past, state)
        present = (self.discard(state), rnn_state)
        state = state.reshape(batch, -1, num_heads, hidden_size)
        state = state.permute(0, 2, 1, 3)
        return state, present


class CompressedGPT2Attention(GPT2Attention):
    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )

            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            time_indicator, past_key_state, past_value_state = layer_past
        else:
            time_indicator, past_key_state, past_value_state = None, None, None
            T = 0
        key, key_state = self.compress_key(key, past_key_state)
        value, value_state = self.compress_value(value, past_value_state)
        attention_mask = self.compress_key.modify_mask(attention_mask)
        if use_cache is True:
            T = query.size(-2) + (time_indicator.size(-2) if time_indicator is not None else 0)
            time_indicator = torch.zeros(1, 1).expand(T, 1)
            present = (time_indicator, key_state, value_state)
        else:
            present = None

        if self.reorder_and_upcast_attn:
            attn_output, attn_weights = self._upcast_and_reordered_attn(
                query, key, value, attention_mask, head_mask)
        else:
            attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)
        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)


class GPT2RecurrentCompression(nn.Module):
    def __init__(self, pretrained_model, rnn_type='LSTM', state_length=128, ratio=1, residual=False,
                 rnn_config={'num_layers': 1}, wrap_pretrained=True):
        super().__init__()
        self.state_length = state_length
        self.ratio = ratio
        self.compressors = nn.ModuleList()
        self.pretrained_model = pretrained_model
        for p in self.pretrained_model.parameters():
            p.requires_grad = False

        for module in self.pretrained_model.modules():
            if isinstance(module, GPT2Attention):
                state_properties = {'num_heads': module.num_heads, 'hidden_size': module.head_dim}
                config = dict(state_length=state_length, ratio=ratio, residual=residual,
                              state_properties=state_properties,
                              rnn_type=rnn_type, rnn_config=rnn_config)
                compressors = nn.ModuleDict({
                    'key': RecurrentState(**config),
                    'value': RecurrentState(**config)
                })
                if wrap_pretrained:
                    module.compress_key = compressors['key']
                    module.compress_value = compressors['value']
                    module.forward = types.MethodType(CompressedGPT2Attention.forward, module)
                    module.compress_key.modify_bias(module.bias)
                self.compressors.append(compressors)

    def forward(self, *args, **kwargs):
        return self.pretrained_model(*args, **kwargs)

    # do not require self
    @staticmethod
    def compress_fn(compressors, past_key_values, retain_state=False):
        compressed_past_key_values = []
        for idx in range(len(past_key_values)):
            compressor = compressors[idx]
            _, c_key = compressor['key'](past_key_values[idx][0])
            _, c_value = compressor['value'](past_key_values[idx][1])
            if not retain_state:
                c_key = c_key[0]
                c_value = c_value[0]
            compressed_past_key_values.append((c_key, c_value))
        return tuple(compressed_past_key_values)

    def compress(self, past_key_values, retain_state=False):
        return self.compress_fn(self.compressors, past_key_values, retain_state)
