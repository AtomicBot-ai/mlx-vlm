from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from ..base import (
    LanguageModelOutput,
    cache_position_offset,
    create_attention_mask,
    project_logits,
    scaled_dot_product_attention,
)
from ..cache import KVCache
from ..mlp import SwiGLUMLP as MLP
from ..rope_utils import initialize_rope
from .config import TextConfig


class Attention(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        dim = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = dim // self.n_heads
        # Granite scales attention by attention_multiplier, not 1/sqrt(head_dim).
        self.scale = config.attention_multiplier
        bias = config.attention_bias
        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=bias)
        self.rope = initialize_rope(
            self.head_dim,
            config.rope_theta,
            False,
            config.rope_scaling,
            config.max_position_embeddings,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, _ = x.shape
        queries = self.q_proj(x).reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = self.k_proj(x).reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = self.v_proj(x).reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        return self.o_proj(output.transpose(0, 2, 1, 3).reshape(B, L, -1))


class TransformerBlock(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.self_attn = Attention(config)
        self.mlp = MLP(config.hidden_size, config.intermediate_size, config.mlp_bias)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.residual_multiplier = config.residual_multiplier

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        h = x + self.self_attn(self.input_layernorm(x), mask, cache) * (
            self.residual_multiplier
        )
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r * self.residual_multiplier


def _deepstack_window(deepstack_visual_embeds: mx.array, window: int, cache):
    """Columns of the full-prompt DeepStack array for the current window.

    Single-sequence chunked prefill passes the array for the full prompt with
    each chunk. Batched prefill slices it before the call.
    """
    if deepstack_visual_embeds.shape[1] == window:
        return deepstack_visual_embeds
    start = cache_position_offset(cache)
    part = deepstack_visual_embeds[:, start : start + window]
    if part.shape[1] == 0:
        return None  # Generated tokens have no image features.
    if part.shape[1] < window:
        pad = mx.zeros((part.shape[0], window - part.shape[1], *part.shape[2:]))
        part = mx.concatenate([part, pad.astype(part.dtype)], axis=1)
    return part


class GraniteDoclingTextModel(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            TransformerBlock(config) for _ in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.embedding_multiplier = config.embedding_multiplier

    def __call__(
        self,
        inputs: mx.array,
        inputs_embeds: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        deepstack_visual_embeds: Optional[mx.array] = None,
    ) -> mx.array:
        """deepstack_visual_embeds: (batch, length, layers, hidden), zero at text."""
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(inputs)
        # The multiplier applies to every input embedding, image tokens included.
        h = inputs_embeds * self.embedding_multiplier

        if cache is None:
            cache = [None] * len(self.layers)
        if mask is None:
            mask = create_attention_mask(h, cache[0])

        num_deepstack = 0
        if deepstack_visual_embeds is not None:
            deepstack_visual_embeds = _deepstack_window(
                deepstack_visual_embeds, h.shape[1], cache[0]
            )
        if deepstack_visual_embeds is not None:
            num_deepstack = deepstack_visual_embeds.shape[2]

        for layer_idx, (layer, c) in enumerate(zip(self.layers, cache)):
            h = layer(h, mask, c)
            if layer_idx < num_deepstack:
                h = h + deepstack_visual_embeds[:, :, layer_idx].astype(h.dtype)

        return self.norm(h)


class LanguageModel(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.model = GraniteDoclingTextModel(config)
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.logits_scaling = config.logits_scaling

    def __call__(
        self,
        inputs: mx.array,
        inputs_embeds: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        deepstack_visual_embeds: Optional[mx.array] = None,
        logits_to_keep: Optional[int] = 0,
        **kwargs,
    ):
        h = self.model(
            inputs,
            inputs_embeds=inputs_embeds,
            mask=mask,
            cache=cache,
            deepstack_visual_embeds=deepstack_visual_embeds,
        )
        projection = (
            self.model.embed_tokens.as_linear
            if self.config.tie_word_embeddings
            else self.lm_head
        )
        logits = project_logits(h, projection, logits_to_keep)
        return LanguageModelOutput(logits=logits / self.logits_scaling)

    @property
    def layers(self):
        return self.model.layers

    @property
    def head_dim(self):
        return self.config.hidden_size // self.config.num_attention_heads

    @property
    def n_kv_heads(self):
        return self.config.num_key_value_heads

    def make_cache(self):
        return [KVCache() for _ in self.layers]
