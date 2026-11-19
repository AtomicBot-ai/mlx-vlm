# Copyright © 2026 Neywa Labs Inc.

"""Ripple block verification with parallel attention and transactional caches."""

from functools import partial

from ...speculative.common import verify_forward
from ..base import (
    LanguageModelOutput,
    create_attention_mask,
    scaled_dot_product_attention,
)


def _attention(attention, x, cache, value_residual):
    queries, keys, values = attention._project_qkv(x, cache, value_residual)
    fetch_block = getattr(cache, "fetch_verification_block", None)
    block = callable(fetch_block) and queries.shape[2] > 1
    if block:
        keys, values, mask = fetch_block(keys, values)
        batch, heads, length, dim = queries.shape
        queries = queries.transpose(0, 2, 1, 3).reshape(batch * length, heads, 1, dim)
        keys = keys.reshape(batch * length, *keys.shape[2:])
        values = values.reshape(batch * length, *values.shape[2:])
        mask = mask.reshape(batch * length, 1, 1, mask.shape[-1])
    else:
        mask = create_attention_mask(x, cache, window_size=attention.sliding_window)
        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)
    output = scaled_dot_product_attention(
        queries, keys, values, cache=cache, scale=attention.scale, mask=mask
    )
    if block:
        output = output.reshape(batch, length, heads, dim).transpose(0, 2, 1, 3)
    return attention._project_output(x, output)


def _verify_block(language_model, inputs, cache, capture_layer_ids):
    model = language_model.model
    capture = model._capture_layers(capture_layer_ids)
    h = model._embed(inputs)
    value_residual = h if model.config.use_valres else None
    captured = {}
    for index, (layer, entry) in enumerate(zip(model.layers, cache)):
        attended = _attention(
            layer.self_attn, layer.input_layernorm(h), entry, value_residual
        )
        h = h + layer.post_attention_layernorm(attended)
        h = h + layer.post_feedforward_layernorm(
            layer.mlp(layer.pre_feedforward_layernorm(h))
        )
        if index in capture:
            captured[index] = h
    return LanguageModelOutput(
        logits=language_model._project_logits(model.norm(h)),
        hidden_states=(
            [captured[index] for index in capture_layer_ids]
            if capture_layer_ids is not None
            else None
        ),
    )


def verify(language_model, inputs, cache, capture_layer_ids=None):
    output, transaction = verify_forward(
        partial(_verify_block, language_model),
        inputs,
        cache,
        capture_layer_ids=capture_layer_ids,
    )
    output.gdn_states = transaction
    return output
