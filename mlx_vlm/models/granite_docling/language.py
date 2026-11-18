from typing import Any, List, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

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


def _add_deepstack(
    hidden_states: mx.array, visual_pos_masks: np.ndarray, visual_embeds: mx.array
) -> mx.array:
    """Add DeepStack features to the hidden states at the image token positions."""
    B, L, D = hidden_states.shape
    positions = mx.array(np.flatnonzero(visual_pos_masks), dtype=mx.uint32)
    flat = hidden_states.reshape(B * L, D)
    flat = flat.at[positions].add(visual_embeds.astype(flat.dtype))
    return flat.reshape(B, L, D)


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
        visual_pos_masks: Optional[mx.array] = None,
        deepstack_visual_embeds: Optional[List[mx.array]] = None,
    ) -> mx.array:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(inputs)
        # The multiplier applies to every input embedding, image tokens included.
        h = inputs_embeds * self.embedding_multiplier

        if cache is None:
            cache = [None] * len(self.layers)
        if mask is None:
            mask = create_attention_mask(h, cache[0])

        deepstack_masks = None
        if deepstack_visual_embeds is not None and visual_pos_masks is not None:
            visual_pos_masks, deepstack_visual_embeds = _align_to_window(
                visual_pos_masks, deepstack_visual_embeds, h.shape[1], cache[0]
            )
            if visual_pos_masks.any():
                deepstack_masks = visual_pos_masks

        for layer_idx, (layer, c) in enumerate(zip(self.layers, cache)):
            h = layer(h, mask, c)
            if deepstack_masks is not None and layer_idx < len(deepstack_visual_embeds):
                h = _add_deepstack(
                    h, deepstack_masks, deepstack_visual_embeds[layer_idx]
                )

        return self.norm(h)


def _align_to_window(visual_pos_masks, deepstack_visual_embeds, window: int, cache):
    """Select the image positions and DeepStack rows of the current window.

    get_input_embeddings returns the mask and the features for the full prompt.
    Chunked prefill calls the model with one slice of the prompt at a time, so
    keep only the columns from the cache offset to the end of this slice. The
    features are batch-major, the same order as np.nonzero on the mask.
    """
    mask = np.array(visual_pos_masks).astype(bool)
    if mask.shape[-1] == window:
        return mask, deepstack_visual_embeds
    start = cache_position_offset(cache)
    columns = np.nonzero(mask)[1]
    keep = mx.array(
        np.flatnonzero((columns >= start) & (columns < start + window)),
        dtype=mx.uint32,
    )
    return (
        mask[:, start : start + window],
        [embeds[keep] for embeds in deepstack_visual_embeds],
    )


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
        visual_pos_masks: Optional[mx.array] = None,
        deepstack_visual_embeds: Optional[List[mx.array]] = None,
        logits_to_keep: Optional[int] = 0,
        **kwargs,
    ):
        h = self.model(
            inputs,
            inputs_embeds=inputs_embeds,
            mask=mask,
            cache=cache,
            visual_pos_masks=visual_pos_masks,
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
