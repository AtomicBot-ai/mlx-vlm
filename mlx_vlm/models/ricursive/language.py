# Copyright © 2026 Neywa Labs Inc.

from functools import partial

import mlx.core as mx
import mlx.nn as nn

from ..base import (
    LanguageModelOutput,
    create_attention_mask,
    project_logits,
    scaled_dot_product_attention,
)
from ..linear import DECODE_BLOCK_SIZE, linear
from ..quantized_verifier import optimized_affine_linear
from .cache import KVCache, RotatingKVCache
from .config import ModelConfig


def _linear(layer, x):
    # The fused affine kernel matches singleton QMV. Native batches can use a
    # different reduction, so preserve their arithmetic with the shared helper.
    if (
        x.ndim == 3
        and x.shape[0] == 1
        and 1 < x.shape[1] <= DECODE_BLOCK_SIZE
        and not getattr(layer, "training", False)
    ):
        output = optimized_affine_linear(layer, x)
        if output is not None:
            return output
    return linear(layer, x)


@partial(mx.compile, shapeless=True)
def _rms_norm(x, weight, eps):
    # Preserve the upstream FP32 steps before rounding and applying the weight.
    hidden = x.astype(mx.float32)
    variance = mx.mean(hidden * hidden, axis=-1, keepdims=True)
    normalized = hidden * mx.rsqrt(variance + eps)
    return normalized.astype(x.dtype) * weight


@mx.compile
def _apply_rotary(x, angles):
    # Upstream rounds the trigonometric values before the FP32 rotation.
    cos = mx.cos(angles).astype(x.dtype).astype(mx.float32)[:, None]
    sin = mx.sin(angles).astype(x.dtype).astype(mx.float32)[:, None]
    first, second = mx.split(x.astype(mx.float32), 2, axis=-1)
    return mx.concatenate(
        (first * cos - second * sin, second * cos + first * sin), axis=-1
    ).astype(x.dtype)


@partial(mx.compile, shapeless=True)
def _interpolate_values(x, residual, weight):
    blend = mx.sigmoid(weight[0].astype(mx.float32)).astype(x.dtype)
    # Match torch.lerp's FP32 interpolation and single output rounding.
    blend = blend.astype(mx.float32)
    seed = residual.astype(x.dtype).astype(mx.float32)
    end = x.astype(mx.float32)
    delta = end - seed
    return mx.where(
        blend < 0.5, seed + blend * delta, end + (blend - 1) * delta
    ).astype(x.dtype)


@partial(mx.compile, shapeless=True)
def _gate_attention(out, gate, full_precision):
    if full_precision:
        gate = gate.astype(mx.float32)
    return out * mx.sigmoid(gate).astype(out.dtype)


@partial(mx.compile, shapeless=True)
def _clipped_swiglu(gate, up, threshold):
    if threshold:
        gate = mx.minimum(gate, threshold)
        up = mx.clip(up, -threshold, threshold)
    # SiLU rounds once, before the separate BF16/FP16 multiplication by up.
    activation = nn.silu(gate.astype(mx.float32)).astype(gate.dtype)
    return activation * up


class RMSNorm(nn.RMSNorm):
    def __call__(self, x):
        return _rms_norm(x, self.weight, self.eps)


class RotaryEmbedding(nn.Module):
    def __init__(self, dims, base):
        super().__init__()
        self._inv_freq = 1.0 / mx.array(
            [base ** (i / dims) for i in range(0, dims, 2)], dtype=mx.float32
        )

    def __call__(self, x, offset=0):
        positions = mx.arange(x.shape[-2], dtype=mx.float32)
        positions = positions + mx.array(offset, dtype=mx.float32).reshape(-1, 1)
        return _apply_rotary(x, positions[..., None] * self._inv_freq)


class Attention(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5
        self.gate_full_precision = config.gate_full_precision
        self.is_sliding = config.layer_types[layer_idx] == "sliding_attention"
        self.sliding_window = config.sliding_window if self.is_sliding else None

        self.q_proj = nn.Linear(
            config.hidden_size, self.n_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            config.hidden_size, self.n_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.n_kv_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.n_heads * self.head_dim, config.hidden_size, bias=False
        )
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        if config.use_valres:
            self.value_res_lambda = mx.zeros((1,))
        if config.use_gated_attention:
            self.g_proj = nn.Linear(
                config.hidden_size, self.n_heads * self.head_dim, bias=False
            )
        if self.is_sliding:
            self.rope = RotaryEmbedding(self.head_dim, config.rope_theta)

    def __call__(self, x, mask=None, cache=None, value_residual=None):
        queries, keys, values = self._project_qkv(x, cache, value_residual)
        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)
        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        return self._project_output(x, output)

    def _project_qkv(self, x, cache, value_residual):
        batch, length, _ = x.shape
        value_states = x
        if value_residual is not None:
            value_states = _interpolate_values(x, value_residual, self.value_res_lambda)

        queries = self.q_norm(
            _linear(self.q_proj, x).reshape(batch, length, self.n_heads, self.head_dim)
        )
        keys = self.k_norm(
            _linear(self.k_proj, x).reshape(
                batch, length, self.n_kv_heads, self.head_dim
            )
        )
        values = _linear(self.v_proj, value_states).reshape(
            batch, length, self.n_kv_heads, self.head_dim
        )
        queries, keys, values = (
            v.transpose(0, 2, 1, 3) for v in (queries, keys, values)
        )

        if self.is_sliding:
            offset = cache.offset if cache is not None else 0
            queries = self.rope(queries, offset=offset)
            keys = self.rope(keys, offset=offset)
        return queries, keys, values

    def _project_output(self, x, out):
        batch, length, _ = x.shape
        out = out.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        if hasattr(self, "g_proj"):
            out = _gate_attention(
                out, _linear(self.g_proj, x), self.gate_full_precision
            )
        return _linear(self.o_proj, out)


class MLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.clamp_threshold = config.ffn_clamp_threshold
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )
        self._compiled_forward = mx.compile(self._forward, inputs=self.state)

    def __call__(self, x):
        if self.training:
            return self._forward(x)
        return self._compiled_forward(x)

    def _forward(self, x):
        gate, up = _linear(self.gate_proj, x), _linear(self.up_proj, x)
        return _linear(self.down_proj, _clipped_swiglu(gate, up, self.clamp_threshold))


class DecoderLayer(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.self_attn = Attention(config, layer_idx)
        self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_feedforward_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_feedforward_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def __call__(self, x, mask=None, cache=None, value_residual=None):
        attn = self.self_attn(self.input_layernorm(x), mask, cache, value_residual)
        x = x + self.post_attention_layernorm(attn)
        return x + self.post_feedforward_layernorm(
            self.mlp(self.pre_feedforward_layernorm(x))
        )


class RicursiveModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        inputs,
        cache=None,
        inputs_embeds=None,
        capture_layer_ids=None,
        hidden_sink=None,
    ):
        capture = self._capture_layers(capture_layer_ids)
        h = self._embed(inputs, inputs_embeds)
        # The residual is the scaled input embedding, before the first pre-norm.
        value_residual = h if self.config.use_valres else None
        if cache is None:
            cache = [None] * len(self.layers)
        captured = {}
        masks = {}
        for index, (layer, c) in enumerate(zip(self.layers, cache)):
            window = self.config.sliding_window if layer.self_attn.is_sliding else None
            if window not in masks:
                masks[window] = create_attention_mask(h, c, window_size=window)
            h = layer(h, masks[window], c, value_residual)
            if index in capture:
                captured[index] = h
        if hidden_sink is not None:
            hidden_sink.extend(captured[index] for index in capture_layer_ids or [])
        return self.norm(h)

    def _capture_layers(self, capture_layer_ids):
        capture = set(capture_layer_ids or [])
        if len(capture) != len(capture_layer_ids or []) or any(
            type(index) is not int or not 0 <= index < len(self.layers)
            for index in capture
        ):
            raise ValueError("capture_layer_ids must be unique valid layer indices")
        return capture

    def _embed(self, inputs, inputs_embeds=None):
        h = self.embed_tokens(inputs) if inputs_embeds is None else inputs_embeds
        if self.config.embed_scale is not None:
            # Keep the Python scalar in FP32 until the scaled embedding is rounded.
            h = (h.astype(mx.float32) * self.config.embed_scale).astype(h.dtype)
        return h


class LanguageModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.model = RicursiveModel(config)
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(
        self,
        inputs=None,
        cache=None,
        input_embeddings=None,
        inputs_embeds=None,
        logits_to_keep: int = 0,
        capture_layer_ids=None,
        **kwargs,
    ) -> LanguageModelOutput:
        if inputs is None:
            inputs = kwargs.get("input_ids")
        if inputs_embeds is None:
            inputs_embeds = input_embeddings
        if kwargs.pop("speculative_verify", False):
            from .speculative_verifier import verify

            return verify(self, inputs, cache, capture_layer_ids)
        hidden_sink = [] if capture_layer_ids is not None else None
        out = self.model(inputs, cache, inputs_embeds, capture_layer_ids, hidden_sink)
        return LanguageModelOutput(
            logits=self._project_logits(out, logits_to_keep), hidden_states=hidden_sink
        )

    def _project_logits(self, hidden, logits_to_keep=0):
        projection = (
            self.model.embed_tokens.as_linear
            if self.config.tie_word_embeddings
            else self.lm_head
        )
        return project_logits(
            hidden,
            (
                projection
                if self.training
                else lambda hidden: _linear(projection, hidden)
            ),
            logits_to_keep,
        )

    def rollback_speculative_cache(self, caches, transaction, accepted, block_size):
        from ...speculative.cache_state import rollback_speculative_cache

        return rollback_speculative_cache(caches, transaction, accepted, block_size)

    def chunked_prefill_policy(
        self, *, draft_model=None, draft_kind=None, prefill_kwargs=None, **kwargs
    ):
        return draft_model is None or (
            draft_kind == "dflash"
            and (prefill_kwargs or {}).get("capture_layer_ids") is not None
        )

    def make_cache(self):
        return [
            (
                RotatingKVCache(max_size=self.config.sliding_window, keep=0)
                if kind == "sliding_attention"
                and self.config.sliding_window is not None
                else KVCache()
            )
            for kind in self.config.layer_types
        ]

    @property
    def layers(self):
        return self.model.layers

    @property
    def head_dim(self):
        return self.config.head_dim

    @property
    def n_kv_heads(self):
        return self.config.num_key_value_heads
