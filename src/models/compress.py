import types
from typing import Optional, Tuple, Union
import torch
from torch import nn
import transformers
from transformers import AutoTokenizer
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel, GPT2Attention, GPT2Config
from src.models.modules.conv_attn import ConvMultiheadAttention
from transformers.models.llama.modeling_llama import LlamaConfig, LlamaRMSNorm


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
        state_size = state_properties['hidden_size'] * state_properties.get('num_heads', 1)
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
        self.output_proj = _proj(rnn_config['hidden_size'] *
                                 (2 if rnn_config['bidirectional'] else 1), state_size)
        self.rnn = rnnf(**rnn_config)

    def discard(self, state):
        if self.state_length is not None:
            state = state[:, :, -self.state_length:]
        if self.ratio > 1:
            # make sure last state is taken
            # state = state[:, :, ::self.ratio]
            last = None
            if state.shape[2] % (self.ratio - 1) != 0:
                last = state[:, :, -1:]
            state = state[:, :, ::self.ratio]
            if last is not None:
                state = torch.cat([state, last], dim=2)

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


class RecurrentEmb(RecurrentState):
    def __init__(self, rnn_type='LSTM', state_length=128, ratio=1, residual=False,
                 state_properties={'num_heads': 1, 'hidden_size': 12 * 64}, rnn_config={'num_layers': 1}):
        super().__init__(rnn_type=rnn_type, state_length=state_length, ratio=ratio, residual=residual,
                         state_properties=state_properties, rnn_config=rnn_config)

    # def join_states(self, *states):
    #     state = torch.cat([s for s in states if s is not None], dim=-2)
    #     return state

    def discard(self, state):
        if self.state_length is not None:
            state = state[:, -self.state_length:]
        if self.ratio > 1:
            # make sure last state is taken
            # state = state[:, ::self.ratio]
            last = None
            if state.shape[1] % (self.ratio - 1) != 0:
                last = state[:, -1:]
            state = state[:, ::self.ratio]
            if last is not None:
                state = torch.cat([state, last], dim=1)

        return state

    def forward(self, state, past=None):
        # single state is (batch, seq_length, hidden_size)
        residual = state if self.residual else None
        state = self.input_proj(state)
        if past is not None:
            past, rnn_state = past
        else:
            rnn_state = None
        state, rnn_state = self.rnn(state, rnn_state)
        state = self.discard(state)
        state = self.output_proj(state)
        if residual is not None:
            residual = self.discard(residual)
            state = state + residual
        # state = self.join_states(past, state)
        present = (state, rnn_state)
        return state, present


class ConvAttnState(CompressState):
    def __init__(self, kernel_size, stride=1, padding=0, hidden_size=None, num_heads=None, residual=False,
                 input_proj=True, output_proj=True, state_properties={'num_heads': 12, 'hidden_size': 64}, **kwargs):
        super().__init__()
        self.residual = residual
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.input_proj = input_proj
        state_size = state_properties['hidden_size'] * state_properties['num_heads']
        hidden_size = hidden_size or state_size

        def _proj(in_size, out_size):
            if in_size != out_size:
                module = nn.Linear(in_size, out_size)
            else:
                module = nn.Identity()
            return module
        self.activation = nn.ReLU()
        if input_proj:
            input_size = hidden_size
            self.input_proj = nn.Linear(state_size, hidden_size)
        else:
            input_size = state_size
            self.input_proj = None
        if output_proj:
            self.output_proj = nn.Linear(hidden_size, state_size)
        else:
            self.output_proj = None
        self.cattn = ConvMultiheadAttention(input_size, num_heads or state_properties['num_heads'],
                                            kernel_size, stride=stride, padding=padding,
                                            batch_first=True, **kwargs)

    def discard(self, state):
        return state

    def forward(self, state, past=None):
        # single state is (batch, num_heads, seq_length, hidden_size)
        (batch, num_heads, _, hidden_size) = state.shape
        # (batch, seq_length, num_heads * hidden_size)
        state = state.permute(0, 2, 1, 3).flatten(2, 3)
        residual = state if self.residual else None
        if self.input_proj is not None:
            state = self.input_proj(state)
            state = self.activation(state)
        state, _ = self.cattn(state, state, state)
        if self.output_proj is not None:
            state = self.activation(state)
            state = self.output_proj(state)

        if residual is not None:
            residual = torch.nn.functional.avg_pool1d(
                residual.permute(0, 2, 1), 1, self.stride, 0).permute(0, 2, 1)
            state = state + residual
        state = state.reshape(batch, -1, num_heads, hidden_size)
        state = state.permute(0, 2, 1, 3)
        state = self.join_states(past, state)
        present = state
        return state, present


class LlamaMLP(nn.Module):
    def __init__(self, hidden_size=4096, intermediate_size=11008):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.GELU()  # ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


def sample_ratio_last(x, ratio):
    # make sure last state is taken
    last = None
    if x.shape[1] % (ratio - 1) != 0:
        last = x[:, -1:]
    x = x[:, ::ratio]
    if last is not None:
        x = torch.cat([x, last], dim=1)
    return x


class LlamaDecoderLayer(nn.Module):
    def __init__(self, hidden_size=4096, intermediate_size=11008, num_heads=16, rms_norm_eps=1e-6, ratio=1,
                 reduction_mode='avg', max_position_embeddings=1024):
        super().__init__()
        self.hidden_size = hidden_size
        self.self_attn = torch.nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.mlp = LlamaMLP(hidden_size=hidden_size, intermediate_size=intermediate_size)
        self.input_layernorm = LlamaRMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(hidden_size, eps=rms_norm_eps)
        self.ratio = ratio
        self.reduction_mode = reduction_mode
        if self.ratio > 1 and self.reduction_mode == 'emb':
            self.reduction_emb = nn.Embedding(max_position_embeddings, hidden_size)

    def reduce(self, x):
        if self.ratio == 1:
            return x
        sz = int(x.shape[1] // self.ratio)
        if self.reduction_mode == 'avg':
            x = x.permute(0, 2, 1)
            x = torch.nn.functional.adaptive_avg_pool1d(x, sz)
            return x.permute(0, 2, 1)
        elif self.reduction_mode == 'sample':
            assert int(self.ratio) == self.ratio, 'ratio must be an integer for sample'
            return sample_ratio_last(x, self.ratio)
        elif self.reduction_mode == 'emb':
            out = self.reduction_emb(torch.arange(sz, device=x.device).unsqueeze(0))
            return out

    def forward(
            self,
            hidden_states: torch.Tensor):

        if self.ratio == 1:
            residual = hidden_states
        # residual = self.reduce(hidden_states)

        hidden_states = self.input_layernorm(hidden_states)
        compressed_hidden_states = hidden_states
        compressed_hidden_states = self.reduce(hidden_states)
        # Self Attention
        hidden_states, _ = self.self_attn(compressed_hidden_states, hidden_states, hidden_states)
        # hidden_states, self_attn_weights, present_key_value = self.self_attn(
        #     hidden_states=hidden_states,
        #     attention_mask=attention_mask,
        #     position_ids=position_ids,
        #     past_key_value=past_key_value,
        #     output_attentions=output_attentions,
        #     use_cache=use_cache,
        # )
        if self.ratio == 1:
            hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        # outputs = (hidden_states,)

        # if output_attentions:
        #     outputs += (self_attn_weights,)

        # if use_cache:
        #     outputs += (present_key_value,)

        return hidden_states


class CompressAttnBlock(CompressState):
    def __init__(self, input_size, hidden_size, intermediate_size, num_heads, ratio, num_blocks,
                 positions=True, max_position_embeddings=1024,
                 add_inputs_embeds=0, add_output_embeds=0, residual=False,
                 reduction_mode='avg', reduction_layer='last', emb_heads=25):
        super().__init__()
        self.layers = nn.ModuleList()
        self.ratio = ratio
        self.emb_heads = emb_heads

        add_inputs_embeds = nn.Embedding(
            add_inputs_embeds, input_size).weight if add_inputs_embeds > 0 else None
        add_output_embeds = nn.Embedding(
            add_output_embeds, input_size).weight if add_output_embeds > 0 else None
        self.register_parameter('add_inputs_embeds', add_inputs_embeds)
        self.register_parameter('add_output_embeds', add_output_embeds)
        if residual:
            # check ratio is an integer
            assert ratio == int(ratio), 'ratio must be an integer for residual'
        self.residual = residual
        reduction_block = LlamaDecoderLayer(hidden_size=hidden_size,
                                            intermediate_size=intermediate_size,
                                            num_heads=num_heads, ratio=ratio,
                                            max_position_embeddings=1024,
                                            reduction_mode=reduction_mode)
        if reduction_layer == 'first':
            self.layers.append(reduction_block)

        for _ in range(num_blocks - 1):
            self.layers.append(LlamaDecoderLayer(hidden_size=hidden_size,
                                                 intermediate_size=intermediate_size,
                                                 num_heads=num_heads))
        if reduction_layer == 'last':
            self.layers.append(reduction_block)
        if positions:
            self.position_embedding = nn.Embedding(max_position_embeddings, hidden_size)
        else:
            self.position_embedding = None

        if input_size == hidden_size:
            self.input_proj = nn.Identity()
            self.output_proj = nn.Identity()
        else:
            self.input_proj = nn.Sequential(
                nn.Linear(input_size, hidden_size),
            )
            self.output_proj = nn.Sequential(
                # nn.LayerNorm(hidden_size),
                # nn.Linear(hidden_size, hidden_size),
                # nn.GELU(),
                nn.Linear(hidden_size, input_size)
            )

    def compress(self, state, past=None):
        if self.add_inputs_embeds is not None:
            state = torch.cat([state, self.add_inputs_embeds.repeat(state.shape[0], 1, 1)], dim=1)
        if self.emb_heads > 1:
            state = state.permute(0, 2, 1, 3).flatten(2, 3)
        state = self.input_proj(state)
        if self.position_embedding is not None:
            state = state + \
                self.position_embedding(torch.arange(
                    state.shape[1], device=state.device).unsqueeze(0))
        for layer in self.layers:
            state = layer(state)

        return state

    def decompress(self, state, past=None):
        state = self.output_proj(state)

        if self.add_output_embeds is not None:
            state = torch.cat(
                [state, self.add_output_embeds.repeat(state.shape[0], 1, 1)], dim=1)

        if self.emb_heads > 1:
            state = state.reshape(state.shape[0], state.shape[1], self.emb_heads, -1)
            state = state.permute(0, 2, 1, 3)
        return state

    def forward(self, state, past=None):
        if self.residual:
            residual = sample_ratio_last(state, self.ratio)
        else:
            residual = None
        state = self.compress(state, past)
        state = self.decompress(state, past)
        if residual is not None:
            state = state + residual
        return state


class SharedCompress(nn.Module):
    def __init__(self, num_layers, num_hidden, num_heads, ratio=100):
        super().__init__()
        size = num_hidden * num_heads * 2 * num_layers
        self.reduce = nn.Sequential(
            nn.Linear(size, 2 * size // ratio),
            nn.ReLU(inplace=True),
            nn.Linear(2 * size // ratio, size // ratio),
        )
        self.inflate = nn.Sequential(
            nn.Linear(size // ratio, 2 * size // ratio),
            nn.ReLU(inplace=True),
            nn.Linear(2 * size // ratio, size),
        )
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.num_heads = num_heads

    def compress(self, state, past=None):
        num_layers = len(state)
        all_states = []
        for k, v in state:
            all_states.append(k)
            all_states.append(v)
        all_states = torch.stack(all_states, dim=-1)
        # all_states is batch x num_heads x seq_length x num_hidden x (2 * num_layers)
        all_states = all_states.permute(0, 2, 1, 3, 4)
        # batch x seq_length x num_heads x num_hidden x (2 * num_layers)
        all_states = all_states.flatten(2, 4)
        all_states = self.reduce(all_states)
        return all_states

    def decompress(self, state, past=None):
        state = self.inflate(state)
        state = state.reshape(state.shape[0], state.shape[1],
                              self.num_heads, self.num_hidden, 2 * self.num_layers)
        state = state.permute(0, 2, 1, 3, 4)
        new_state = []
        for i in range(self.num_layers):
            new_state.append((state[:, :, :, :, i], state[:, :, :, :, i + self.num_layers]))
        return tuple(new_state)

    def forward(self, state, past=None):
        state = self.compress(state, past)
        state = self.decompress(state, past)
        return state


class SharedSeperableCompress(nn.Module):
    def __init__(self, num_layers, num_hidden, num_heads, ratio_hidden=16, ratio_layers=12, reduce_hidden_first=True):
        super().__init__()
        hidden_size = num_hidden * num_heads
        self.reduce_hidden = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size // ratio_hidden)
        )
        self.inflate_hidden = nn.Sequential(
            nn.Linear(hidden_size // ratio_hidden, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),            
            nn.Linear(hidden_size, hidden_size)
        )
        self.reduce_layers = nn.Sequential(
            nn.Linear(num_layers * 2, num_layers * 2),
            nn.ReLU(inplace=True),
            nn.Linear(num_layers * 2, num_layers * 2),
            nn.ReLU(inplace=True),            
            nn.Linear(num_layers * 2, (num_layers * 2) // ratio_layers)
        )
        self.inflate_layers = nn.Sequential(
            nn.Linear((num_layers * 2) // ratio_layers, num_layers * 2),
            nn.ReLU(inplace=True),
            nn.Linear(num_layers * 2, num_layers * 2),
            nn.ReLU(inplace=True),  
            nn.Linear(num_layers * 2, num_layers * 2)
        )
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.num_heads = num_heads
        self.reduce_hidden_first = reduce_hidden_first

    def compress(self, state, past=None, **kwargs):
        num_layers = len(state)
        all_states = []
        for k, v in state:
            all_states.append(k)
            all_states.append(v)
        all_states = torch.stack(all_states, dim=-1)
        # all_states is batch x num_heads x seq_length x num_hidden x (2 * num_layers)

        if self.reduce_hidden_first:
            all_states = all_states.permute(0, 2, 4, 1, 3)
            # batch x seq_length x (2 * num_layers) x num_heads x num_hidden
            all_states = all_states.flatten(3, 4)
            all_states = self.reduce_hidden(all_states)
            all_states = all_states.permute(0, 1, 3, 2)
            # batch x seq_length x reduced_hidden x (2 * num_layers)
            all_states = self.reduce_layers(all_states)
        else:
            all_states = self.reduce_layers(all_states)
            all_states = all_states.permute(0, 2, 4, 1, 3)
            # batch x seq_length x reduced_layers x (num_heads * num_hidden)
            all_states = all_states.flatten(3, 4)
            all_states = self.reduce_hidden(all_states)
        return all_states

    def decompress(self, state, past=None, **kwargs):
        if self.reduce_hidden_first:
            state = self.inflate_layers(state)
            state = state.permute(0, 1, 3, 2)
            state = self.inflate_hidden(state)
            state = state.reshape(state.shape[0], state.shape[1],
                                  2 * self.num_layers, self.num_heads, self.num_hidden)
            state = state.permute(0, 3, 1, 4, 2)
        else:
            state = self.inflate_hidden(state)
            state = state.reshape(state.shape[0], state.shape[1],
                                  state.shape[2], self.num_heads, self.num_hidden)
            state = state.permute(0, 3, 1, 4, 2)
            state = self.inflate_layers(state)
        new_state = []
        for i in range(self.num_layers):
            new_state.append((state[:, :, :, :, i], state[:, :, :, :, i + self.num_layers]))
        return tuple(new_state)

    def forward(self, state, past=None, **kwargs):
        state = self.compress(state, past)
        state = self.decompress(state, past)
        return state


class CompressAttnState(CompressState):
    def __init__(self, input_size, hidden_size, num_heads, ratio, num_blocks=4, residual=False):
        super().__init__()
        self.residual = residual
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.ratio = ratio
        self.num_heads = num_heads
        self.activation = nn.GELU()
        # self.input_proj = nn.Sequential(
        #     # nn.LayerNorm(input_size),
        #     nn.Linear(input_size, hidden_size),
        #     nn.ReLU(),
        # )
        # self.output_proj = nn.Sequential(
        #     # nn.LayerNorm(hidden_size),
        #     # nn.Linear(hidden_size, hidden_size),
        #     # nn.GELU(),
        #     nn.Linear(hidden_size, input_size)
        # )

        self.attn = torch.nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)

    def discard(self, x):
        sz = int(x.shape[1] // self.ratio)
        x = x.permute(0, 2, 1)
        x = torch.nn.functional.adaptive_avg_pool1d(x, sz)
        return x.permute(0, 2, 1)

    def forward(self, state, past=None):
        # single state is (batch, seq_length, hidden_size)
        residual = state if self.residual else None
        # state = self.input_proj(state)
        state, _ = self.attn(self.discard(state), state, state)
        # state = self.output_proj(state)

        if residual is not None:
            residual = self.discard(residual)
            state = state + residual
        # state = self.join_states(past, state)
        present = state
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
    def compress_fn(compressors, past_key_values, compression_steps=1, retain_state=False):
        for _ in range(compression_steps):
            compressed_past_key_values = []
            for idx in range(len(past_key_values)):
                compressor = compressors[idx]
                _, c_key = compressor['key'](past_key_values[idx][0])
                _, c_value = compressor['value'](past_key_values[idx][1])
                if not retain_state:
                    c_key = c_key[0]
                    c_value = c_value[0]
                compressed_past_key_values.append((c_key, c_value))
            past_key_values = tuple(compressed_past_key_values)
        return past_key_values

    def compress(self, past_key_values, compression_steps=1, retain_state=False):
        return self.compress_fn(self.compressors, past_key_values=past_key_values,
                                compression_steps=compression_steps, retain_state=retain_state)


class EmbCompression(nn.Module):
    def __init__(self, pretrained_model, compressor, wrap_pretrained=True):
        super().__init__()
        self.pretrained_model = pretrained_model
        for p in self.pretrained_model.parameters():
            p.requires_grad = False

        if wrap_pretrained:
            raise NotImplementedError
        self.compressors = compressor

    def forward(self, *args, **kwargs):
        return self.pretrained_model(*args, **kwargs)

    # do not require self
    @staticmethod
    def compress_fn(compressor, state, compression_steps=1, retain_state=False):
        compressed_state = state
        for _ in range(compression_steps):
            _, compressed_state = compressor(compressed_state)
        return compressed_state

    def compress(self, state, compression_steps=1, retain_state=False):
        return self.compress_fn(self.compressors, state, compression_steps, retain_state)


class EmbRecurrentCompression(EmbCompression):
    def __init__(self, pretrained_model, rnn_type='LSTM', state_length=128, ratio=1, residual=False,
                 rnn_config={'num_layers': 1}, state_properties={}, wrap_pretrained=True):
        config = dict(state_length=state_length, ratio=ratio, residual=residual,
                      rnn_type=rnn_type, rnn_config=rnn_config, state_properties=state_properties)
        compressors = RecurrentEmb(**config)
        super().__init__(pretrained_model, compressors, wrap_pretrained=wrap_pretrained)
        self.state_length = state_length
        self.ratio = ratio

    # do not require self
    @staticmethod
    def compress_fn(compressor, state, compression_steps=1, retain_state=False):
        compressed_state = state
        for _ in range(compression_steps):
            _, (compressed_state, _) = compressor(compressed_state)
        return compressed_state


class SharedStateCompression(nn.Module):
    def __init__(self, pretrained_model, compressors):
        super().__init__()
        self.compressors = compressors
        self.pretrained_model = pretrained_model
        for p in self.pretrained_model.parameters():
            p.requires_grad = False

    # do not require self
    @staticmethod
    def compress_fn(compressors, past_key_values):
        return compressors.compress(past_key_values)

    def compress(self, past_key_values):
        return self.compress_fn(self.compressors, past_key_values)

    # do not require self
    @staticmethod
    def decompress_fn(compressors, past_key_values):
        return compressors.decompress(past_key_values)

    def decompress(self, past_key_values):
        return self.decompress_fn(self.compressors, past_key_values)

    # do not require self

    @staticmethod
    def forward_fn(compressors, past_key_values, compression_steps=1, retain_state=False):
        return compressors(past_key_values)

    def forward(self, past_key_values, compression_steps=1, retain_state=False):
        return self.forward_fn(self.compressors, past_key_values, compression_steps, retain_state)


class StateCompression(nn.Module):
    def __init__(self, pretrained_model, compressors):
        super().__init__()
        self.compressors = compressors
        self.pretrained_model = pretrained_model
        for p in self.pretrained_model.parameters():
            p.requires_grad = False

    # do not require self
    @staticmethod
    def compress_fn(compressors, past_key_values, compression_steps=1, retain_state=False):
        for _ in range(compression_steps):
            compressed_past_key_values = []
            for idx in range(len(past_key_values)):
                compressor = compressors[idx]
                c_key = compressor['key'].compress(past_key_values[idx][0])
                c_value = compressor['value'].compress(past_key_values[idx][1])
                compressed_past_key_values.append((c_key, c_value))
            past_key_values = tuple(compressed_past_key_values)
        return past_key_values

    def compress(self, past_key_values, compression_steps=1, retain_state=False):
        return self.compress_fn(self.compressors, past_key_values, compression_steps, retain_state)

    # do not require self
    @staticmethod
    def decompress_fn(compressors, past_key_values, compression_steps=1, retain_state=False):
        for _ in range(compression_steps):
            compressed_past_key_values = []
            for idx in range(len(past_key_values)):
                compressor = compressors[idx]
                c_key = compressor['key'].decompress(past_key_values[idx][0])
                c_value = compressor['value'].decompress(past_key_values[idx][1])
                compressed_past_key_values.append((c_key, c_value))
            past_key_values = tuple(compressed_past_key_values)
        return past_key_values

    def decompress(self, past_key_values, compression_steps=1, retain_state=False):
        return self.decompress_fn(self.compressors, past_key_values, compression_steps, retain_state)

    # do not require self

    @staticmethod
    def forward_fn(compressors, past_key_values, compression_steps=1, retain_state=False):
        for _ in range(compression_steps):
            compressed_past_key_values = []
            for idx in range(len(past_key_values)):
                compressor = compressors[idx]
                c_key = compressor['key'](past_key_values[idx][0])
                c_value = compressor['value'](past_key_values[idx][1])
                compressed_past_key_values.append((c_key, c_value))
            past_key_values = tuple(compressed_past_key_values)
        return past_key_values

    def forward(self, past_key_values, compression_steps=1, retain_state=False):
        return self.forward_fn(self.compressors, past_key_values, compression_steps, retain_state)


class GPT2AttnBlockCompression(StateCompression):
    def __init__(self, pretrained_model, wrap_pretrained=True, **compressor_config):
        compressors = nn.ModuleList()
        for module in pretrained_model.modules():
            if isinstance(module, GPT2Attention):
                compressor = nn.ModuleDict({
                    'key': CompressAttnBlock(**compressor_config),
                    'value': CompressAttnBlock(**compressor_config)
                })
                if wrap_pretrained:
                    module.compress_key = compressor['key']
                    module.compress_value = compressor['value']
                    module.forward = types.MethodType(CompressedGPT2Attention.forward, module)
                    # module.compress_key.modify_bias(module.bias)
                compressors.append(compressor)
        super().__init__(pretrained_model, compressors)


class GPT2ConvAttnCompression(StateCompression):
    def __init__(self, pretrained_model, hidden_size=None, num_heads=None, kernel_size=3, stride=2, padding=1, residual=False,
                 wrap_pretrained=True, **cattn_config):
        super().__init__()
        compressors = nn.ModuleList()

        for module in self.pretrained_model.modules():
            if isinstance(module, GPT2Attention):
                state_properties = {'num_heads': module.num_heads, 'hidden_size': module.head_dim}
                config = dict(hidden_size=hidden_size, num_heads=num_heads,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, residual=residual,
                              state_properties=state_properties, **cattn_config)
                compressors = nn.ModuleDict({
                    'key': ConvAttnState(**config),
                    'value': ConvAttnState(**config)
                })
                if wrap_pretrained:
                    module.compress_key = compressors['key']
                    module.compress_value = compressors['value']
                    module.forward = types.MethodType(CompressedGPT2Attention.forward, module)
                    # module.compress_key.modify_bias(module.bias)
                compressors.append(compressors)
        super().__init__(pretrained_model, compressors, wrap_pretrained=wrap_pretrained)
