# Copyright © 2026 Neywa Labs Inc.

import mlx.core as mx
from mlx.nn.layers.distributed import sum_gradients

from ..base import create_attention_mask, create_ssm_mask
from ..fast_ops import short_conv_block
from ..linear import DECODE_BLOCK_SIZE, linear, native_batch_linear, tokenwise


@mx.compile
def route_block(x, weight, bias, top_k, n_group, topk_group, normalize, scaling):
    from .language import route_experts

    return tokenwise(
        lambda token: route_experts(
            token, weight, bias, top_k, n_group, topk_group, normalize, scaling
        ),
        x,
    )


@mx.compile
def combine_experts(experts, scores, x, sharded):
    dtype = mx.float32 if sharded else x.dtype
    return (experts.astype(mx.float32) * scores[..., None]).sum(-2).astype(dtype)


class BailingHybridSpeculativeVerifier:
    """Verify blocks with decode-equivalent projections and cache visibility."""

    def verify_hidden(self, language_model, inputs, cache):
        from ...speculative.common import verify_forward

        output, transaction = verify_forward(
            language_model, inputs, cache, skip_logits=True, return_hidden=True
        )
        return output.hidden_states[-1], {}, transaction

    def logits_from_hidden(self, language_model, hidden):
        head = (
            language_model.model.word_embeddings.as_linear
            if language_model.args.tie_word_embeddings
            else language_model.lm_head
        )
        return native_batch_linear(head, hidden)

    def _feed_forward(self, layer, x):
        # language.py owns the ordinary modules and imports this verifier.
        from .language import SparseMoE, clipped_swiglu

        if isinstance(layer, SparseMoE):
            if layer.sharding_group is not None:
                x = sum_gradients(layer.sharding_group)(x)
            gate = layer.gate
            args = gate.args
            indices, scores = route_block(
                x,
                gate.weight,
                gate.expert_bias,
                args.num_experts_per_tok,
                args.n_group,
                args.topk_group,
                args.norm_topk_prob,
                args.routed_scaling_factor,
            )
            experts = layer.switch_mlp(x, indices)
            out = combine_experts(experts, scores, x, layer.sharding_group is not None)
            if layer.shared_experts is not None:
                out = out + self._feed_forward(layer.shared_experts, x)
        else:
            out = linear(
                layer.down_proj,
                clipped_swiglu(
                    linear(layer.up_proj, x), linear(layer.gate_proj, x), layer.limit
                ),
            )
        if layer.sharding_group is not None:
            out = mx.distributed.all_sum(out, group=layer.sharding_group)
        return out.astype(x.dtype) if isinstance(layer, SparseMoE) else out

    def _gated_delta(self, attention, x, mask, cache):
        batch, length, _ = x.shape
        lengths = cache.lengths if cache is not None else None
        qkv = []
        for i, name in enumerate(("q", "k", "v")):
            conv = getattr(attention, f"{name}_conv1d")
            projected = linear(getattr(attention, f"{name}_proj"), x)
            previous = cache[i] if cache is not None else None
            fused = (
                short_conv_block(projected, previous, conv.conv.weight)
                if not conv.training and mask is None and lengths is None
                else None
            )
            if fused is not None:
                y, state = fused
            elif length > 1:
                state, steps = previous, []
                for t in range(length):
                    y, state = conv(
                        mx.contiguous(projected[:, t : t + 1]),
                        state,
                        None if mask is None else mask[:, t : t + 1],
                        None if lengths is None else lengths - t,
                    )
                    steps.append(y)
                y = mx.concatenate(steps, axis=1)
            else:
                y, state = conv(projected, previous, mask, lengths)
            if cache is not None:
                width = conv.kernel_size - 1
                if previous is None:
                    previous = mx.zeros((batch, width, projected.shape[-1]), x.dtype)
                if mask is not None:
                    projected = mx.where(mask[..., None], projected, 0)
                cache.record_speculative_window(
                    i, mx.concatenate([previous, projected], axis=1), width
                )
                cache[i] = state
            qkv.append(
                y.reshape(batch, length, attention.num_heads, attention.head_dim)
            )
        a = (
            linear(attention.f_proj, x)
            if attention.args.no_kda_lora
            else linear(attention.f_b_proj, linear(attention.f_a_proj, x))
        )
        out = attention.recurrent(*qkv, a, linear(attention.b_proj, x), cache, mask)
        gate = (
            linear(attention.g_proj, x)
            if attention.args.no_kda_lora
            else linear(attention.g_b_proj, linear(attention.g_a_proj, x))
        )
        gate = mx.sigmoid(gate.astype(mx.float32)).astype(x.dtype)
        out = linear(
            attention.o_proj, attention.o_norm(out).reshape(batch, length, -1) * gate
        )
        if attention.output_group is not None:
            out = mx.distributed.all_sum(out, group=attention.output_group)
        return out

    def _attention(self, attention, x, mask, cache):
        args = attention.args
        batch, length, _ = x.shape
        q = (
            linear(attention.q_proj, x)
            if args.q_lora_rank is None
            else linear(
                attention.q_b_proj,
                attention.q_a_layernorm(linear(attention.q_a_proj, x)),
            )
        )
        q = q.reshape(
            batch, length, attention.num_heads, attention.q_head_dim
        ).transpose(0, 2, 1, 3)
        latent, k_pe = mx.split(
            linear(attention.kv_a_proj_with_mqa, x), [args.kv_lora_rank], axis=-1
        )
        latent = attention.kv_a_layernorm(latent)[:, None]
        k_pe = k_pe[:, None]
        if length > 1:
            # Each query sees the same cache prefix as one ordinary decode step.
            prefix = cache.state[0].shape[-2] if cache.state[0] is not None else 0
            outputs = []
            for t in range(length):
                step_mask = (
                    None if mask is None else mask[..., t : t + 1, : prefix + t + 1]
                )
                outputs.append(
                    attention.attend(
                        mx.contiguous(q[:, :, t : t + 1]),
                        mx.contiguous(latent[:, :, t : t + 1]),
                        mx.contiguous(k_pe[:, :, t : t + 1]),
                        step_mask,
                        cache,
                    )
                )
            out = mx.concatenate(outputs, axis=2)
        else:
            out = attention.attend(q, latent, k_pe, mask, cache)
        out = out.transpose(0, 2, 1, 3)
        if attention.g_proj is not None:
            gate = mx.sigmoid(linear(attention.g_proj, x).astype(mx.float32)).astype(
                x.dtype
            )
            out = out * gate.reshape(batch, length, attention.num_heads, -1)
        out = linear(attention.dense, out.reshape(batch, length, -1))
        if attention.output_group is not None:
            out = mx.distributed.all_sum(out, group=attention.output_group)
        return out

    def __call__(self, model, hidden, cache):
        if hidden.shape[1] > DECODE_BLOCK_SIZE:
            return mx.concatenate(
                [
                    self(model, hidden[:, i : i + DECODE_BLOCK_SIZE], cache)
                    for i in range(0, hidden.shape[1], DECODE_BLOCK_SIZE)
                ],
                axis=1,
            )
        masks = {}
        for i, (layer, c) in enumerate(zip(model.layers, cache)):
            if layer.is_linear not in masks:
                masks[layer.is_linear] = (
                    create_ssm_mask(hidden, c)
                    if layer.is_linear
                    else create_attention_mask(hidden, c, return_array=True)
                )
            attention = self._gated_delta if layer.is_linear else self._attention
            hidden = hidden + attention(
                layer.attention,
                layer.input_layernorm(hidden),
                masks[layer.is_linear],
                c,
            )
            hidden = hidden + self._feed_forward(
                layer.mlp, layer.post_attention_layernorm(hidden)
            )
            if (
                model.sharding_group is not None
                and not model.training
                and mx.default_device() == mx.gpu
                and (i + 1) % 8 == 0
            ):
                mx.async_eval(hidden)
        return model.norm(hidden)
