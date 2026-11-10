# Copyright © 2026 Neywa Labs Inc.

import math
import re
from functools import partial

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import shard_inplace, shard_linear, sum_gradients

from ..base import (
    LanguageModelOutput,
    create_attention_mask,
    create_ssm_mask,
    scaled_dot_product_attention,
)
from ..cache import ArraysCache, KVCache
from ..gated_delta import gated_delta_update
from ..kimi_k3.language import ShortConv1d
from ..linear import DECODE_BLOCK_SIZE, linear
from ..mla import MultiLinear
from ..rope_utils import initialize_rope
from ..switch_layers import SwitchGLU
from .config import ModelConfig
from .speculative_verifier import BailingHybridSpeculativeVerifier

_SPECULATIVE_VERIFIER = BailingHybridSpeculativeVerifier()


@partial(mx.compile, shapeless=True)
def clipped_swiglu(up, gate, limit):
    gate = nn.silu(gate)
    if limit:
        gate = mx.minimum(gate, limit)
        up = mx.clip(up, -limit, limit)
    return gate * up


class MLP(nn.Module):
    def __init__(self, args, intermediate_size, limit=None):
        super().__init__()
        self.limit = limit
        self.sharding_group = None
        self.gate_proj = nn.Linear(args.hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(args.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, args.hidden_size, bias=False)

    def __call__(self, x):
        out = self.down_proj(
            clipped_swiglu(self.up_proj(x), self.gate_proj(x), self.limit)
        )
        if self.sharding_group is not None:
            out = mx.distributed.all_sum(out, group=self.sharding_group)
        return out


def route_experts(x, weight, bias, top_k, n_group, topk_group, normalize, scaling):
    scores = mx.sigmoid(x.astype(mx.float32) @ weight.astype(mx.float32).T)
    routing = scores + bias
    if n_group > 1 and topk_group < n_group:
        grouped = mx.unflatten(routing, -1, (n_group, -1))
        group_scores = mx.topk(grouped, 2, axis=-1).sum(-1)
        groups = mx.argpartition(-group_scores, topk_group - 1, axis=-1)
        keep = mx.any(
            mx.arange(n_group) == groups[..., :topk_group, None],
            axis=-2,
        )
        routing = mx.where(keep[..., None], grouped, -mx.inf).flatten(-2)
    indices = mx.argpartition(-routing, top_k - 1, axis=-1)
    indices = mx.stop_gradient(indices[..., :top_k])
    weights = mx.take_along_axis(scores, indices, axis=-1)
    if top_k > 1 and normalize:
        weights = weights / (weights.sum(-1, keepdims=True) + 1e-20)
    return indices, weights * scaling


class Gate(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.weight = mx.zeros((args.num_experts, args.hidden_size), mx.float32)
        self.expert_bias = mx.zeros((args.num_experts,), mx.float32)

    def __call__(self, x):
        args = self.args
        return route_experts(
            x,
            self.weight,
            self.expert_bias,
            args.num_experts_per_tok,
            args.n_group,
            args.topk_group,
            args.norm_topk_prob,
            args.routed_scaling_factor,
        )


class SparseMoE(nn.Module):
    def __init__(self, args, layer_idx):
        super().__init__()
        self.sharding_group = None

        def limit(values):
            return values[layer_idx] if values and layer_idx < len(values) else None

        self.gate = Gate(args)
        self.switch_mlp = SwitchGLU(
            args.hidden_size,
            args.moe_intermediate_size,
            args.num_experts,
            activation=partial(
                clipped_swiglu, limit=limit(args.expert_swiglu_limit_list)
            ),
        )
        self.shared_experts = (
            MLP(
                args,
                (args.moe_shared_expert_intermediate_size or args.moe_intermediate_size)
                * args.num_shared_experts,
                limit(args.share_expert_swiglu_limit_list),
            )
            if args.num_shared_experts
            else None
        )

    def __call__(self, x):
        if self.sharding_group is not None:
            x = sum_gradients(self.sharding_group)(x)
        indices, scores = self.gate(x)
        experts = self.switch_mlp(x, indices)
        # Shards hold partial expert sums; round only after adding the shared
        # expert and reducing across ranks.
        dtype = mx.float32 if self.sharding_group is not None else x.dtype
        out = (experts.astype(mx.float32) * scores[..., None]).sum(-2).astype(dtype)
        if self.shared_experts is not None:
            out = out + self.shared_experts(x)
        if self.sharding_group is not None:
            out = mx.distributed.all_sum(out, group=self.sharding_group).astype(x.dtype)
        return out


class KimiDeltaAttention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.output_group = None
        self.args = args
        self.num_heads = args.num_attention_heads
        self.head_dim = args.head_dim
        projection_size = self.num_heads * self.head_dim
        for name in ("q", "k", "v"):
            setattr(
                self,
                f"{name}_proj",
                nn.Linear(args.hidden_size, projection_size, bias=False),
            )
            setattr(
                self,
                f"{name}_conv1d",
                ShortConv1d(projection_size, args.short_conv_kernel_size),
            )
        self.A_log = mx.zeros((self.num_heads,), mx.float32)
        self.dt_bias = mx.zeros((projection_size,), mx.float32)
        self.b_proj = nn.Linear(args.hidden_size, self.num_heads, bias=False)
        for name in ("f", "g"):
            if args.no_kda_lora:
                setattr(
                    self,
                    f"{name}_proj",
                    nn.Linear(args.hidden_size, projection_size, bias=False),
                )
            else:
                setattr(
                    self,
                    f"{name}_a_proj",
                    nn.Linear(args.hidden_size, self.head_dim, bias=False),
                )
                setattr(
                    self,
                    f"{name}_b_proj",
                    nn.Linear(self.head_dim, projection_size, bias=False),
                )
        self.o_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.o_proj = nn.Linear(projection_size, args.hidden_size, bias=False)

    def __call__(self, x, mask=None, cache=None):
        if cache is not None and cache.is_speculating:
            return _SPECULATIVE_VERIFIER._gated_delta(self, x, mask, cache)
        B, T, _ = x.shape
        lengths = cache.lengths if cache is not None else None
        qkv = []
        for i, name in enumerate(("q", "k", "v")):
            conv = getattr(self, f"{name}_conv1d")
            projected = getattr(self, f"{name}_proj")(x)
            previous = cache[i] if cache is not None else None
            y, state = conv(projected, previous, mask, lengths)
            if cache is not None:
                cache[i] = state
            qkv.append(y.reshape(B, T, self.num_heads, self.head_dim))
        a = self.f_proj(x) if self.args.no_kda_lora else self.f_b_proj(self.f_a_proj(x))
        out = self.recurrent(*qkv, a, self.b_proj(x), cache, mask)
        gate = (
            self.g_proj(x) if self.args.no_kda_lora else self.g_b_proj(self.g_a_proj(x))
        )
        gate = mx.sigmoid(gate.astype(mx.float32)).astype(x.dtype)
        out = self.o_proj(self.o_norm(out).reshape(B, T, -1) * gate)
        if self.output_group is not None:
            out = mx.distributed.all_sum(out, group=self.output_group)
        return out

    def recurrent(self, q, k, v, a, b, cache, mask):
        B, T = q.shape[:2]
        # FLA applies L2 normalization (epsilon inside sqrt), then scales Q.
        scale = self.head_dim**-0.5
        q = scale**2 * mx.fast.rms_norm(q, None, 1e-6 / self.head_dim)
        k = scale * mx.fast.rms_norm(k, None, 1e-6 / self.head_dim)
        out, state = gated_delta_update(
            q,
            k,
            v,
            a.reshape(B, T, self.num_heads, self.head_dim).astype(mx.float32),
            b.astype(mx.float32),
            self.A_log[:, None],
            self.dt_bias.reshape(self.num_heads, self.head_dim),
            cache=cache,
            cache_index=3,
            mask=mask,
            use_kernel=not self.training,
            lower_bound=self.args.kda_lower_bound if self.args.kda_safe_gate else None,
        )
        if cache is not None:
            cache.advance(T)
        return out


class MultiLatentAttention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.output_group = None
        self.args = args
        self.num_heads = args.num_attention_heads
        self.q_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
        self.scale = self.q_head_dim**-0.5
        if args.q_lora_rank is None:
            self.q_proj = nn.Linear(
                args.hidden_size, self.num_heads * self.q_head_dim, bias=False
            )
        else:
            self.q_a_proj = nn.Linear(
                args.hidden_size, args.q_lora_rank, bias=args.use_qkv_bias
            )
            self.q_a_layernorm = nn.RMSNorm(args.q_lora_rank, eps=1e-6)
            self.q_b_proj = nn.Linear(
                args.q_lora_rank, self.num_heads * self.q_head_dim, bias=False
            )
        self.kv_a_proj_with_mqa = nn.Linear(
            args.hidden_size,
            args.kv_lora_rank + args.qk_rope_head_dim,
            bias=args.use_qkv_bias,
        )
        self.kv_a_layernorm = nn.RMSNorm(args.kv_lora_rank, eps=1e-6)
        self.embed_q = MultiLinear(
            args.qk_nope_head_dim, args.kv_lora_rank, self.num_heads
        )
        self.unembed_out = MultiLinear(
            args.kv_lora_rank, args.v_head_dim, self.num_heads
        )
        granularity = args.gated_attention_proj_granularity_type
        if granularity not in (None, "head_wise", "element_wise"):
            raise ValueError(f"Unsupported MLA gate granularity: {granularity}")
        self.g_proj = (
            None
            if granularity is None
            else nn.Linear(
                args.hidden_size,
                self.num_heads
                * (args.v_head_dim if granularity == "element_wise" else 1),
                bias=False,
            )
        )
        self.dense = nn.Linear(
            self.num_heads * args.v_head_dim, args.hidden_size, bias=args.use_qkv_bias
        )
        self.rope = initialize_rope(
            args.qk_rope_head_dim,
            args.rope_theta,
            traditional=args.rope_interleave,
            scaling_config=args.rope_scaling,
            max_position_embeddings=args.max_position_embeddings,
        )
        if args.rope_scaling and args.rope_scaling.get("mscale_all_dim", 0):
            factor = args.rope_scaling.get("factor", 1)
            if factor > 1:
                self.scale *= (
                    1 + 0.1 * args.rope_scaling["mscale_all_dim"] * math.log(factor)
                ) ** 2

    def __call__(self, x, mask=None, cache=None):
        args = self.args
        B, T, _ = x.shape
        q = (
            self.q_proj(x)
            if args.q_lora_rank is None
            else self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x)))
        )
        q = q.reshape(B, T, self.num_heads, self.q_head_dim).transpose(0, 2, 1, 3)
        latent, k_pe = mx.split(
            self.kv_a_proj_with_mqa(x), [args.kv_lora_rank], axis=-1
        )
        latent = self.kv_a_layernorm(latent)[:, None]
        k_pe = k_pe[:, None]
        out = self.attend(q, latent, k_pe, mask, cache)
        out = out.transpose(0, 2, 1, 3)
        if self.g_proj is not None:
            gate = mx.sigmoid(self.g_proj(x).astype(mx.float32)).astype(x.dtype)
            out = out * gate.reshape(B, T, self.num_heads, -1)
        out = self.dense(out.reshape(B, T, -1))
        if self.output_group is not None:
            out = mx.distributed.all_sum(out, group=self.output_group)
        return out

    def attend(self, q, latent, k_pe, mask, cache):
        T = q.shape[2]
        q_nope, q_pe = mx.split(q, [self.args.qk_nope_head_dim], axis=-1)
        offset = cache.offset if cache is not None else 0
        q_pe, k_pe = self.rope(q_pe, offset=offset), self.rope(k_pe, offset=offset)
        if cache is not None:
            latent, k_pe = cache.update_and_fetch(latent, k_pe)
        bias = (q_pe * self.scale) @ k_pe.swapaxes(-1, -2)
        if mask is not None:
            # Fully masked left-padding rows must stay finite: NaNs there can
            # contaminate the next layer's K/V projections before masking.
            bias = mx.where(mask, bias, mx.array(mx.finfo(bias.dtype).min, bias.dtype))
        if T == 1:
            q_nope = self.embed_q(q_nope)
            k = v = latent
        else:
            k, v = self.embed_q(latent, transpose=False), self.unembed_out(latent)
        out = scaled_dot_product_attention(
            q_nope, k, v, cache=cache, scale=self.scale, mask=bias
        )
        if T == 1:
            out = self.unembed_out(out)
        return out


class DecoderLayer(nn.Module):
    def __init__(self, args, layer_idx):
        super().__init__()
        self.is_linear = args.is_linear(layer_idx)
        self.attention = (
            KimiDeltaAttention(args) if self.is_linear else MultiLatentAttention(args)
        )
        self.mlp = (
            SparseMoE(args, layer_idx)
            if layer_idx >= args.first_k_dense_replace and args.num_experts
            else MLP(args, args.intermediate_size)
        )
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

    def __call__(self, x, mask=None, cache=None):
        h = x + self.attention(self.input_layernorm(x), mask, cache)
        return h + self.mlp(self.post_attention_layernorm(h))


class TextModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.sharding_group = None
        self.word_embeddings = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    @property
    def embed_tokens(self):
        return self.word_embeddings

    def __call__(self, inputs, cache=None, inputs_embeds=None):
        h = self.word_embeddings(inputs) if inputs_embeds is None else inputs_embeds
        cache = cache if cache is not None else [None] * len(self.layers)
        if any(getattr(c, "is_speculating", False) for c in cache):
            return _SPECULATIVE_VERIFIER(self, h, cache)
        masks = {}
        for i, (layer, c) in enumerate(zip(self.layers, cache)):
            if layer.is_linear not in masks:
                masks[layer.is_linear] = (
                    create_ssm_mask(h, c)
                    if layer.is_linear
                    else create_attention_mask(h, c, return_array=True)
                )
            h = layer(h, masks[layer.is_linear], c)
            if (
                self.sharding_group is not None
                and not self.training
                and mx.default_device() == mx.gpu
            ):
                if h.shape[1] > DECODE_BLOCK_SIZE:
                    # Large prefill graphs can stall Metal/JACCL.
                    mx.eval(h, c.state if c is not None else [])
                elif (i + 1) % 4 == 0:
                    # Overlap short-block graph construction and execution, as
                    # in the Mistral Large 4 verifier.
                    mx.async_eval(h)
        return self.norm(h)


def shard_decoder_layer(layer, group):
    size, rank = group.size(), group.rank()
    if size == 1:
        return
    attn = layer.attention
    heads = attn.num_heads // size
    start, end = rank * heads, (rank + 1) * heads
    if layer.is_linear:
        dim = attn.head_dim
        for name in ("q", "k", "v"):
            setattr(
                attn,
                f"{name}_proj",
                shard_linear(
                    getattr(attn, f"{name}_proj"), "all-to-sharded", group=group
                ),
            )
            conv = getattr(attn, f"{name}_conv1d").conv
            conv.weight = mx.contiguous(conv.weight[start * dim : end * dim])
            conv.groups = heads * dim
        for name in ("f", "g"):
            name = f"{name}_proj" if attn.args.no_kda_lora else f"{name}_b_proj"
            setattr(
                attn,
                name,
                shard_linear(getattr(attn, name), "all-to-sharded", group=group),
            )
        attn.b_proj = shard_linear(attn.b_proj, "all-to-sharded", group=group)
        shard_inplace(attn.o_proj, "sharded-to-all", group=group)
        attn.output_group = group
        attn.A_log = attn.A_log[start:end]
        attn.dt_bias = attn.dt_bias[start * dim : end * dim]
    else:
        name = "q_proj" if attn.args.q_lora_rank is None else "q_b_proj"
        setattr(
            attn, name, shard_linear(getattr(attn, name), "all-to-sharded", group=group)
        )
        if attn.g_proj is not None:
            attn.g_proj = shard_linear(attn.g_proj, "all-to-sharded", group=group)
        if "bias" in attn.dense:
            # The wrapper adds bias after the reduction, once on each rank.
            attn.dense = shard_linear(attn.dense, "sharded-to-all", group=group)
        else:
            shard_inplace(attn.dense, "sharded-to-all", group=group)
            attn.output_group = group
        attn.embed_q.apply(lambda w: mx.contiguous(w[start:end]))
        attn.unembed_out.apply(lambda w: mx.contiguous(w[start:end]))
    attn.num_heads = heads
    if isinstance(layer.mlp, SparseMoE):
        layer.mlp.sharding_group = group
        modules = [layer.mlp.switch_mlp]
        if layer.mlp.shared_experts is not None:
            modules.append(layer.mlp.shared_experts)
        for mlp in modules:
            for name, mode in (
                ("gate_proj", "all-to-sharded"),
                ("up_proj", "all-to-sharded"),
                ("down_proj", "sharded-to-all"),
            ):
                shard_inplace(getattr(mlp, name), mode, group=group)
    else:
        for name in ("gate_proj", "up_proj"):
            setattr(
                layer.mlp,
                name,
                shard_linear(getattr(layer.mlp, name), "all-to-sharded", group=group),
            )
        shard_inplace(layer.mlp.down_proj, "sharded-to-all", group=group)
        layer.mlp.sharding_group = group


class LanguageModel(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.args = self.config = args
        self.model_type = args.model_type
        self.model = TextModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs,
        cache=None,
        inputs_embeds=None,
        return_hidden=False,
        skip_logits=False,
        **kwargs,
    ):
        h = self.model(inputs, cache, inputs_embeds)
        logits = None if skip_logits else self.logits_from_hidden(h)
        return LanguageModelOutput(
            logits=logits, hidden_states=[h] if return_hidden else None
        )

    def logits_from_hidden(self, h):
        head = (
            self.model.word_embeddings.as_linear
            if self.args.tie_word_embeddings
            else self.lm_head
        )
        return linear(head, h)

    def speculative_logits_from_hidden(self, h):
        return _SPECULATIVE_VERIFIER.logits_from_hidden(self, h)

    def speculative_verify_hidden(self, inputs, cache):
        return _SPECULATIVE_VERIFIER.verify_hidden(self, inputs, cache)

    def shard(self, group=None):
        group = group or mx.distributed.init()
        if self.args.num_attention_heads % group.size():
            raise ValueError("Attention heads must be divisible by the group size")
        for layer in self.layers:
            shard_decoder_layer(layer, group)
            # Release each full checkpoint layer before materializing the next.
            mx.eval(layer.parameters())
        self.model.sharding_group = group if group.size() > 1 else None

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [
            ArraysCache(size=4) if layer.is_linear else KVCache()
            for layer in self.layers
        ]

    @property
    def quant_predicate(self):
        return lambda path, module: not path.endswith(
            ("mlp.gate", ".attention.b_proj", ".attention.g_proj")
        )

    @property
    def cast_predicate(self):
        return lambda key: not key.endswith(
            ("A_log", "dt_bias", "expert_bias", "mlp.gate.weight")
        )

    def sanitize(self, weights):
        weights = {k.removeprefix("language_model."): v for k, v in weights.items()}
        # NEXTN is a separate prediction head, not a target decoder layer.
        weights = {
            k: v
            for k, v in weights.items()
            if not (m := re.match(r"model\.layers\.(\d+)\.", k))
            or int(m[1]) < self.args.num_hidden_layers
        }
        weights = sanitize_weights(
            weights,
            self.args,
            [
                (f"model.layers.{i}", layer.is_linear)
                for i, layer in enumerate(self.layers)
            ],
        )
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)
        return weights


def sanitize_weights(weights, args, layers):
    for key in list(weights):
        if key.endswith(".weight_scale_inv"):
            scale = weights.pop(key)
            name = key.removesuffix("_scale_inv")
            raw = weights[name]
            raw = mx.from_fp8(raw, dtype=mx.bfloat16) if raw.dtype == mx.uint8 else raw
            rows, cols = raw.shape
            padded_rows, padded_cols = (rows + 127) // 128 * 128, (
                cols + 127
            ) // 128 * 128
            raw = mx.pad(raw, ((0, padded_rows - rows), (0, padded_cols - cols)))
            raw = raw.reshape(padded_rows // 128, 128, padded_cols // 128, 128)
            weights[name] = (
                (raw * scale[:, None, :, None])
                .reshape(padded_rows, padded_cols)[:rows, :cols]
                .astype(mx.bfloat16)
            )
    for prefix, is_linear in layers:
        for proj in ("gate_proj", "up_proj", "down_proj"):
            for suffix in ("weight", "scales", "biases"):
                if f"{prefix}.mlp.experts.0.{proj}.{suffix}" in weights:
                    weights[f"{prefix}.mlp.switch_mlp.{proj}.{suffix}"] = mx.stack(
                        [
                            weights.pop(f"{prefix}.mlp.experts.{e}.{proj}.{suffix}")
                            for e in range(args.num_experts)
                        ]
                    )
        prefix += ".attention"
        if is_linear:
            for name in ("q", "k", "v"):
                key = f"{prefix}.{name}_conv1d.weight"
                if key in weights:
                    w = weights.pop(key)
                    weights[f"{prefix}.{name}_conv1d.conv.weight"] = (
                        w.swapaxes(1, 2) if w.ndim == 3 else w[..., None]
                    )
        else:
            key = f"{prefix}.kv_b_proj.weight"
            if key in weights:
                w = weights.pop(key)
                quantized = f"{prefix}.kv_b_proj.scales" in weights
                if quantized:
                    scales = weights.pop(f"{prefix}.kv_b_proj.scales")
                    biases = weights.pop(f"{prefix}.kv_b_proj.biases", None)
                    bits = w.shape[-1] * 32 // args.kv_lora_rank
                    group_size = args.kv_lora_rank // scales.shape[-1]
                    mode = "mxfp8" if scales.dtype == mx.uint8 else "affine"
                    w = mx.dequantize(
                        w, scales, biases, bits=bits, group_size=group_size, mode=mode
                    )
                w = w.reshape(
                    args.num_attention_heads,
                    args.qk_nope_head_dim + args.v_head_dim,
                    args.kv_lora_rank,
                )
                weights[f"{prefix}.embed_q.weight"] = mx.contiguous(
                    w[:, : args.qk_nope_head_dim].swapaxes(-1, -2)
                )
                weights[f"{prefix}.unembed_out.weight"] = mx.contiguous(
                    w[:, args.qk_nope_head_dim :]
                )
                if quantized:
                    for projection in ("embed_q", "unembed_out"):
                        name = f"{prefix}.{projection}"
                        packed = mx.quantize(
                            weights[name + ".weight"],
                            bits=bits,
                            group_size=group_size,
                            mode=mode,
                        )
                        for suffix, tensor in zip(
                            ("weight", "scales", "biases"), packed
                        ):
                            weights[name + "." + suffix] = tensor
    return weights
