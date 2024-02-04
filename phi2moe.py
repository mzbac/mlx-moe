import inspect
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np


@dataclass
class ModelArgs:
    vocab_size: int = 51200
    max_position_embeddings: int = 2048
    hidden_size: int = 2560
    intermediate_size: int = 10240
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 32
    num_experts_per_tok: int = 2
    num_local_experts: int = 4
    rms_norm_eps: float = 1e-5
    vocab_size: int
    layer_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    partial_rotary_factor: float = 0.4
    rope_traditional: bool = False
    model_type: str = None

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

    @classmethod
    def from_dict(cls, params):
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


class LayerNorm(nn.LayerNorm):
    def __call__(self, x: mx.array) -> mx.array:
        return super().__call__(x.astype(mx.float32)).astype(x.dtype)


class PhiMoeAttention(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.repeats = self.num_heads // self.num_key_value_heads
        self.rope_theta = config.rope_theta
        self.partial_rotary_factor = config.partial_rotary_factor

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=True
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True
        )
        self.dense = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=True
        )

        self.rope = nn.RoPE(
            int(self.partial_rotary_factor * self.head_dim),
            traditional=False,
            base=self.rope_theta,
        )

    def __call__(self, x, mask=None, cache=None):
        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # Extract some shapes
        B, L, D = queries.shape

        # Prepare the queries, keys and values for the attention computation
        queries = queries.reshape(B, L, self.num_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )
        keys = keys.reshape(B, L, self.num_key_value_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )
        values = values.reshape(
            B, L, self.num_key_value_heads, self.head_dim
        ).transpose(0, 2, 1, 3)

        def repeat(a):
            a = mx.concatenate([mx.expand_dims(a, 2)] * self.repeats, axis=2)
            return a.reshape([B, self.num_heads, L, -1])

        if self.repeats > 1:
            keys, values = map(repeat, (keys, values))

        # Add RoPE to the queries and keys and combine them with the cache
        if cache is not None:
            key_cache, value_cache = cache
            queries = self.rope(queries, offset=key_cache.shape[2])
            keys = self.rope(keys, offset=key_cache.shape[2])
            keys = mx.concatenate([key_cache, keys], axis=2)
            values = mx.concatenate([value_cache, values], axis=2)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        queries = queries.astype(mx.float32)
        keys = keys.astype(mx.float32)

        # Finally perform the attention computation
        scale = math.sqrt(1 / queries.shape[-1])
        scores = (queries * scale) @ keys.transpose(0, 1, 3, 2)
        if mask is not None:
            scores = scores + mask

        scores = mx.softmax(scores, axis=-1).astype(values.dtype)
        values_hat = (scores @ values).transpose(0, 2, 1, 3).reshape(B, L, -1)

        return self.dense(values_hat), (keys, values)


class PhiMoeBLockSparseTop2MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.fc1 = nn.Linear(args.hidden_size, args.intermediate_size)
        self.fc2 = nn.Linear(args.intermediate_size, args.hidden_size)
        self.act = nn.GELU(approx="precise")

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(self.act(self.fc1(x)))


class PhiMoeSparseMoeBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.hidden_dim = args.hidden_size
        self.ffn_dim = args.intermediate_size
        self.num_experts = args.num_local_experts
        self.num_experts_per_tok = args.num_experts_per_tok

        # gating
        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)
        self._noise_linear = nn.Linear(self.hidden_dim, self.num_experts, bias=False)

        self.experts = [
            PhiMoeBLockSparseTop2MLP(args=args) for _ in range(self.num_experts)
        ]

    def __call__(self, x: mx.array) -> mx.array:
        ne = self.num_experts_per_tok
        orig_shape = x.shape
        x = x.reshape(-1, x.shape[-1])

        gates = self.gate(x)
        if self.training:

            def softplus(x):
                return mx.log(1 + mx.exp(x))

            noise_logits = self._noise_linear(x)
            noise = mx.random.normal(
                shape=noise_logits.shape, dtype=noise_logits.dtype
            ) * softplus(noise_logits)
            gates = gates + noise

        inds = mx.stop_gradient(mx.argpartition(-gates, kth=ne, axis=-1))[:, :ne]

        scores = mx.softmax(
            mx.take_along_axis(gates, inds, axis=-1).astype(mx.float32),
            axis=-1,
        ).astype(gates.dtype)

        if self.training:
            y = mx.zeros((x.shape[0], ne, x.shape[-1]))
            for e, expert in enumerate(self.experts):
                idx1, idx2 = map(mx.array, np.where(inds == e))
                if idx1.size == 0:
                    continue
                y[idx1, idx2] = expert(x[idx1])

            y = (y * scores[:, :, None]).sum(axis=1)
        else:
            y = []
            for xt, st, it in zip(x, scores, inds.tolist()):
                yt = mx.concatenate([self.experts[e](xt)[:, None] for e in it], axis=-1)
                yt = (yt * st).sum(axis=-1)
                y.append(yt[None, :])
            y = mx.concatenate(y)

        return y.reshape(orig_shape)


class PhiMoeDecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.hidden_size = args.hidden_size

        self.self_attn = PhiMoeAttention(args)

        self.block_sparse_moe = PhiMoeSparseMoeBlock(args)
        self.input_layernorm = LayerNorm(args.hidden_size, eps=args.layer_norm_eps)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        h = self.input_layernorm(x)
        attn_h, cache = self.self_attn(h, mask, cache)
        ff_h = self.block_sparse_moe(h)
        return attn_h + ff_h + x, cache


class PhiMoeModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers

        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            PhiMoeDecoderLayer(args=args) for _ in range(args.num_hidden_layers)
        ]
        self.final_layernorm = LayerNorm(args.hidden_size, eps=args.layer_norm_eps)

    def __call__(self, x, mask, cache):
        x = self.embed_tokens(x)
        if cache is None:
            cache = [None] * len(self.layers)

        for e, layer in enumerate(self.layers):
            x, cache[e] = layer(x, mask, cache[e])
        return self.final_layernorm(x), cache


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.model = PhiMoeModel(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=True)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array = None,
        cache: mx.array = None,
    ):
        mask = None
        if x.shape[1] > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(x.shape[1])
            mask = mask.astype(x.dtype)

        y, cache = self.model(x, mask, cache)
        return self.lm_head(y), cache