# Copyright © 2026 Neywa Labs Inc.

"""ERNIE 5.5 Flash: HCA/MLA, latent multi-router MoE and mHC.

Architecture reference: https://github.com/baidu/vLLM-ERNIE
"""

import re

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import shard_inplace
from mlx.utils import tree_flatten

from ...fp8 import MLX_MXFP8_QUANTIZATION, _dequantize_fp8_weight
from ...speculative.cache_state import (
    rollback_speculative_cache,
    start_speculative_cache,
)
from ..base import (
    LanguageModelOutput,
    create_attention_mask,
    scaled_dot_product_attention,
)
from ..cache import CacheList, KVCache, PoolingCache, RotatingKVCache
from ..deepseek_v4.hyper_connection import HyperHead, _hc_split_sinkhorn_ops, hc_expand
from ..deepseek_v4.language import (
    CompressedAttention,
    Compressor,
    _align_local_mask,
    _extend_mask,
)
from ..mla import MultiLinear
from ..switch_layers import SwitchGLU
from .config import ModelConfig


def _parallel_output(projection, x, group):
    if group is None:
        return projection(x)
    # Match a full dot product's accumulation before rounding back to BF16.
    out = projection(x.astype(mx.float32))
    return mx.distributed.all_sum(out, group=group).astype(x.dtype)


class SiTU(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.beta = config.activation_situ_beta
        self.linear_beta = config.activation_situ_linear_beta
        self.output_dtype = None

    def __call__(self, up, gate):
        dtype = gate.dtype
        gate, up = gate.astype(mx.float32), up.astype(mx.float32)
        gate = self.beta * mx.tanh(gate / self.beta) * mx.sigmoid(gate)
        up = self.linear_beta * mx.tanh(up / self.linear_beta)
        return (gate * up).astype(dtype).astype(self.output_dtype or dtype)


class MLP(nn.Module):
    def __init__(self, config, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.activation = SiTU(config)
        self.sharding_group = None

    def __call__(self, x):
        return _parallel_output(
            self.down_proj,
            self.activation(self.up_proj(x), self.gate_proj(x)),
            self.sharding_group,
        )


class MultiRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.weight = mx.zeros((config.n_routed_experts, config.hidden_size))
        self.weight_1 = mx.zeros_like(self.weight)
        self.routed_scaling_factor_param = mx.full(
            (config.n_routed_experts,), config.routed_scaling_factor
        )
        self.top_k = config.num_experts_per_tok
        self.normalize = config.norm_topk_prob

    def __call__(self, x, bias):
        x = x.astype(mx.float32)
        scores = mx.sigmoid(x @ self.weight.astype(mx.float32).T)
        scores += mx.sigmoid(x @ self.weight_1.astype(mx.float32).T)
        ids = mx.argpartition(-(scores + bias), kth=self.top_k - 1, axis=-1)[
            ..., : self.top_k
        ]
        ids = mx.stop_gradient(ids)
        weights = mx.take_along_axis(scores, ids, axis=-1)
        if self.normalize:
            weights /= mx.maximum(weights.sum(axis=-1, keepdims=True), 1e-20)
        return ids, weights * self.routed_scaling_factor_param[ids]


class LatentMoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate = MultiRouter(config)
        self.e_score_correction_bias = mx.zeros((config.n_routed_experts,))
        self.fc1_latent_proj = nn.Linear(
            config.hidden_size, config.moe_latent_size, bias=False
        )
        self.switch_mlp = SwitchGLU(
            config.moe_latent_size,
            config.moe_intermediate_size,
            config.n_routed_experts,
            activation=SiTU(config),
        )
        self.latent_norm = nn.RMSNorm(config.moe_latent_size, eps=config.rms_norm_eps)
        self.fc2_latent_proj = nn.Linear(
            config.moe_latent_size, config.hidden_size, bias=False
        )
        self.shared_experts = MLP(
            config, config.n_shared_experts * config.moe_intermediate_size
        )
        self.sharding_group = None

    def __call__(self, x, router_bias=None):
        bias = self.e_score_correction_bias if router_bias is None else router_bias
        ids, weights = self.gate(x, bias)
        latent = self.fc1_latent_proj(x)
        outputs = self.switch_mlp(latent, ids)
        if self.sharding_group is not None:
            # Reconstruct each expert before weighting and the latent RMSNorm.
            outputs = mx.distributed.all_sum(outputs, group=self.sharding_group)
            outputs = outputs.astype(latent.dtype)
        combined = (
            (outputs.astype(mx.float32) * weights[..., None])
            .sum(axis=-2)
            .astype(x.dtype)
        )
        return self.shared_experts(x) + self.fc2_latent_proj(self.latent_norm(combined))


class HyperConnection(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mult = config.hc_mult
        self.iters = config.hc_sinkhorn_iters
        self.eps = config.hc_eps
        self.norm_eps = config.rms_norm_eps
        count = (2 + self.mult) * self.mult
        self.fn = mx.zeros((count, self.mult * config.hidden_size), dtype=mx.float32)
        self.base = mx.zeros((count,), dtype=mx.float32)
        self.scale = mx.ones((3,), dtype=mx.float32)

    def __call__(self, x):
        flat = x.astype(mx.float32).flatten(-2)
        # ERNIE places epsilon outside sqrt in the sublayer mHC mixers.
        mixes = (flat @ self.fn.astype(mx.float32).T) / (
            mx.sqrt(mx.mean(flat * flat, axis=-1, keepdims=True)) + self.norm_eps
        )
        pre, post, comb = _hc_split_sinkhorn_ops(
            mixes, self.scale, self.base, self.mult, self.iters, self.eps
        )
        collapsed = (pre[..., None] * x.astype(mx.float32)).sum(axis=-2).astype(x.dtype)
        return collapsed, post, comb


class VirtualHeads(nn.Module):
    def __init__(self, heads, rank):
        super().__init__()
        self.U = mx.zeros((heads, rank))
        self.V = mx.zeros((heads, rank))
        self.sharding_group = None

    def __call__(self, x):
        # x: B, L, heads, value_dim. Fold before applying to match the
        # reference's rounding order; recompute to allow weight reloads.
        mix = (
            mx.eye(self.U.shape[0])
            + self.V.astype(mx.float32) @ self.U.astype(mx.float32).T
        )
        heads = x.shape[-2]
        if self.sharding_group is not None:
            x = mx.distributed.all_gather(
                x.transpose(2, 0, 1, 3), group=self.sharding_group
            ).transpose(1, 2, 0, 3)
            start = self.sharding_group.rank() * heads
            mix = mix[start : start + heads]
        return mix.astype(x.dtype) @ x


class HCACompressor(Compressor):
    def __call__(self, x, pool_cache, offset):
        B, _, _ = x.shape
        # The reference accumulates the compression projections and pooling
        # buffers in fp32, even for a bf16 backbone.
        kv = x.astype(mx.float32) @ self.wkv.weight.astype(mx.float32).T
        gate = x.astype(mx.float32) @ self.wgate.weight.astype(mx.float32).T
        if pool_cache is None:
            usable = kv.shape[1] // self.compress_ratio * self.compress_ratio
            kv, gate, base = kv[:, :usable], gate[:, :usable], offset
        else:
            kv, gate, base = pool_cache.accumulate_windows(kv, gate, offset)
        if kv.shape[1] == 0:
            pooled = mx.zeros((B, 0, self.head_dim), dtype=x.dtype)
        else:
            kv = kv.reshape(B, -1, self.compress_ratio, self.head_dim)
            gate = gate.reshape(B, -1, self.compress_ratio, self.head_dim)
            weights = mx.softmax(
                gate + self.ape.astype(mx.float32), axis=-2, precise=True
            )
            pooled = self.norm((kv * weights).sum(axis=-2))
            pooled = self.rope(pooled[:, None], offset=base).squeeze(1).astype(x.dtype)
        return pool_cache.update_and_fetch(pooled) if pool_cache is not None else pooled


class HCAAttention(CompressedAttention):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.compressor = HCACompressor(config, self.compress_ratio, self.head_dim)
        self.o_gate = nn.Linear(
            config.q_lora_rank, config.o_groups * config.o_lora_rank, bias=False
        )
        self.vha = VirtualHeads(self.n_heads, config.vha_rank)

    def __call__(self, x, cache=None):
        B, L, _ = x.shape
        local = cache[0] if cache is not None else None
        pool = cache[1] if cache is not None else None
        offset = local.offset if local is not None else 0
        offset = mx.array(offset) if isinstance(offset, mx.array) else offset
        mask = create_attention_mask(
            x, local, window_size=self.config.sliding_window, return_array=True
        )
        qr = self.q_norm(self.wq_a(x))
        q = self.wq_b(qr).reshape(B, L, self.n_heads, self.head_dim)
        q = mx.fast.rms_norm(q, None, self.config.rms_norm_eps).transpose(0, 2, 1, 3)
        q = self.rope(q, offset)
        kv = self.rope(self.kv_norm(self.wkv(x))[:, None], offset)
        if local is not None:
            kv, _ = local.update_and_fetch(kv, mx.zeros((B, 1, L, 0), dtype=x.dtype))
        mask = _align_local_mask(mask, kv.shape[2])
        pooled = self.compressor(x, pool, offset)
        if pooled.shape[1]:
            if pool is not None:
                pool_mask = pool.make_mask(L, offset)
            else:
                pool_mask = mx.arange(pooled.shape[1])[None] < (
                    mx.arange(1, L + 1)[:, None] // self.compress_ratio
                )
            if mask is None and pool_mask is not None:
                mask = mx.ones((B, 1, L, kv.shape[2]), dtype=mx.bool_)
            kv = mx.concatenate((kv, pooled[:, None]), axis=2)
            mask = _extend_mask(mask, pool_mask, kv.shape[2])
        out = scaled_dot_product_attention(
            q,
            kv,
            kv,
            cache=local,
            scale=self.scale,
            mask=mask,
            sinks=self.attn_sink.astype(q.dtype),
        )
        out = self.rope(out, offset, inverse=True).transpose(0, 2, 1, 3)
        out = self.vha(out).reshape(B, L, self.o_groups, -1).transpose(0, 2, 1, 3)
        out = self.wo_a(out).transpose(0, 2, 1, 3).flatten(-2)
        return _parallel_output(
            self.wo_b, out * mx.sigmoid(self.o_gate(qr)), self.vha.sharding_group
        )


class Indexer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.index_n_heads
        self.dim = config.index_head_dim
        self.topk = config.index_topk
        self.window = config.sliding_window
        self.wq_b = nn.Linear(
            config.hybrid_mla_q_lora_rank, self.heads * self.dim, bias=False
        )
        self.wk = nn.Linear(config.hidden_size, self.dim, bias=False)
        self.k_norm = nn.LayerNorm(self.dim, eps=1e-6)
        self.weights_proj = nn.Linear(config.hidden_size, self.heads, bias=False)
        self.rope = nn.RoPE(
            config.hybrid_mla_qk_rope_head_dim,
            traditional=False,
            base=config.rope_theta,
        )

    def __call__(self, x, qr, mask, cache=None):
        B, L, _ = x.shape
        offset = cache.offset if cache is not None else 0
        offset = mx.array(offset) if isinstance(offset, mx.array) else offset
        q = self.wq_b(qr).reshape(B, L, self.heads, self.dim).transpose(0, 2, 1, 3)
        k = self.k_norm(self.wk(x))[:, None]
        q, k = self.rope(q, offset), self.rope(k, offset)
        # The release's orthonormal Hadamard leaves full-precision dot products
        # unchanged; it is only needed when quantizing the indexer cache.
        if cache is not None:
            k, _ = cache.update_and_fetch(k, mx.zeros((B, 1, L, 0), dtype=k.dtype))
        S = k.shape[2]
        if S <= self.topk + self.window:
            return mask, None
        # Physical cache columns include batch left padding; the causal diagonal
        # is still S-L+j. Always include the recent window, rank only older keys.
        delta = mx.arange(S)[None, :] - mx.arange(S - L, S)[:, None]
        recent = (delta <= 0) & (delta > -self.window)
        older = delta <= -self.window
        valid = older if mask is None else older & mask
        scores = mx.maximum(q @ k.swapaxes(-1, -2), 0).astype(mx.float32)
        weights = (
            self.weights_proj(x).astype(mx.float32) * (self.heads * self.dim) ** -0.5
        )
        scores = (scores * weights.transpose(0, 2, 1)[..., None]).sum(
            axis=1, keepdims=True
        )
        scores = mx.where(valid, scores, -mx.inf)
        ids = mx.argpartition(-scores, kth=self.topk - 1, axis=-1)[..., : self.topk]
        sparse = mx.put_along_axis(
            mx.zeros(scores.shape, dtype=mx.bool_), ids, mx.array(True), axis=-1
        )
        sparse = (sparse & older) | recent
        sparse = sparse if mask is None else sparse & mask
        if L == 1:
            recent_ids = mx.broadcast_to(
                mx.arange(S - self.window, S), (B, 1, 1, self.window)
            )
            return sparse, mx.concatenate((ids, recent_ids), axis=-1)
        return sparse, None


class MLAAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.hybrid_mla_num_attention_heads
        self.nope = config.hybrid_mla_qk_nope_head_dim
        self.rope_dim = config.hybrid_mla_qk_rope_head_dim
        self.latent = config.hybrid_mla_kv_lora_rank
        self.value_dim = config.hybrid_mla_v_head_dim
        qr = config.hybrid_mla_q_lora_rank
        self.scale = (self.nope + self.rope_dim) ** -0.5
        self.q_a_proj = nn.Linear(config.hidden_size, qr, bias=False)
        self.q_a_layernorm = nn.RMSNorm(qr, eps=config.rms_norm_eps)
        self.q_b_proj = nn.Linear(
            qr, self.heads * (self.nope + self.rope_dim), bias=False
        )
        self.kv_a_proj_with_mqa = nn.Linear(
            config.hidden_size, self.latent + self.rope_dim, bias=False
        )
        self.kv_a_layernorm = nn.RMSNorm(self.latent, eps=config.rms_norm_eps)
        self.embed_q = MultiLinear(self.nope, self.latent, self.heads)
        self.unembed_out = MultiLinear(self.latent, self.value_dim, self.heads)
        self.wo_b = nn.Linear(
            self.heads * self.value_dim, config.hidden_size, bias=False
        )
        self.o_gate = nn.Linear(qr, self.heads * self.value_dim, bias=False)
        self.vha = VirtualHeads(self.heads, config.vha_rank)
        self.indexer = Indexer(config)
        self.attn_sink = mx.zeros((self.heads,), dtype=mx.float32)
        self.rope = nn.RoPE(self.rope_dim, traditional=True, base=config.rope_theta)

    def __call__(self, x, cache=None):
        B, L, _ = x.shape
        latent_cache = cache[0] if cache is not None else None
        index_cache = cache[1] if cache is not None else None
        offset = latent_cache.offset if latent_cache is not None else 0
        offset = mx.array(offset) if isinstance(offset, mx.array) else offset
        mask = create_attention_mask(x, latent_cache, return_array=True)
        qr = self.q_a_layernorm(self.q_a_proj(x))
        q = self.q_b_proj(qr).reshape(B, L, self.heads, -1).transpose(0, 2, 1, 3)
        q_nope, q_pe = mx.split(q, [self.nope], axis=-1)
        kv, k_pe = mx.split(self.kv_a_proj_with_mqa(x), [self.latent], axis=-1)
        kv = self.kv_a_layernorm(kv)[:, None]
        q_pe = self.rope(q_pe, offset)
        k_pe = self.rope(k_pe[:, None], offset)
        if latent_cache is not None:
            kv, k_pe = latent_cache.update_and_fetch(kv, k_pe)
        mask, indices = self.indexer(x, qr, mask, index_cache)
        if indices is not None:
            take = indices[:, :, 0, :, None]
            kv = mx.take_along_axis(kv, take, axis=2)
            k_pe = mx.take_along_axis(k_pe, take, axis=2)
            mask = mx.take_along_axis(mask, indices, axis=-1)
        pe_scores = (q_pe * self.scale) @ k_pe.swapaxes(-1, -2)
        if mask is not None:
            pe_scores = mx.where(mask, pe_scores, -mx.inf)
        if L == 1:
            q_nope = self.embed_q(q_nope)
            k = v = kv
        else:
            k = self.embed_q(kv, transpose=False)
            v = self.unembed_out(kv)
        out = scaled_dot_product_attention(
            q_nope,
            k,
            v,
            cache=latent_cache,
            scale=self.scale,
            mask=pe_scores,
            sinks=self.attn_sink.astype(q_nope.dtype),
        )
        if L == 1:
            out = self.unembed_out(out)
        out = self.vha(out.transpose(0, 2, 1, 3)).flatten(-2)
        return _parallel_output(
            self.wo_b, out * mx.sigmoid(self.o_gate(qr)), self.vha.sharding_group
        )


class DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = (
            MLAAttention(config)
            if config.compress_ratios[layer_idx] == -2
            else HCAAttention(config, layer_idx)
        )
        self.ffn = (
            MLP(config, config.intermediate_size)
            if layer_idx < config.first_k_dense_replace
            else LatentMoE(config)
        )
        self.attn_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn_hc = HyperConnection(config)
        self.ffn_hc = HyperConnection(config)

    def __call__(self, x, cache=None, router_bias=None):
        h, post, comb = self.attn_hc(x)
        h = self.attn_norm(h)
        h = self.attn(h, cache)
        x = hc_expand(h, x, post, comb)
        h, post, comb = self.ffn_hc(x)
        h = self.ffn_norm(h)
        h = self.ffn(h) if router_bias is None else self.ffn(h, router_bias)
        return hc_expand(h, x, post, comb)


class ErnieModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        self.hc_head = HyperHead(config)
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        inputs,
        cache=None,
        inputs_embeds=None,
        hidden_sink=None,
        skip_final_norm=False,
    ):
        h = self.embed_tokens(inputs) if inputs_embeds is None else inputs_embeds
        h = mx.broadcast_to(
            h[:, :, None, :], (*h.shape[:2], self.config.hc_mult, h.shape[-1])
        )
        if cache is None:
            cache = [None] * len(self.layers)
        if len(cache) != len(self.layers):
            raise ValueError("Expected one cache per decoder layer")
        for layer, layer_cache in zip(self.layers, cache):
            h = layer(h, layer_cache)
        if hidden_sink is not None:
            hidden_sink.append(h.flatten(-2))
        if skip_final_norm:
            return h
        return self.norm(self.hc_head(h))


class MultiMax(nn.Module):
    def __init__(self):
        super().__init__()
        self.multimax_ranges = mx.zeros((4,), dtype=mx.float32)
        self.multimax_ts = mx.zeros((4,), dtype=mx.float32)

    def __call__(self, logits):
        logits = logits.astype(mx.float32)
        r, t = self.multimax_ranges, self.multimax_ts
        return (
            logits
            + t[0] * mx.maximum(r[0] - logits, 0)
            + t[1] * mx.maximum(logits - r[1], 0)
            + t[2] * mx.maximum(r[2] - logits, 0) ** 2
            + t[3] * mx.maximum(logits - r[3], 0) ** 2
        )


class LanguageModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.model = ErnieModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.multimax = MultiMax()
        self.sharding_group = None

    def __call__(self, inputs=None, cache=None, inputs_embeds=None, **kwargs):
        hidden_sink = [] if kwargs.get("return_hidden", False) else None
        h = self.model(inputs, cache, inputs_embeds, hidden_sink=hidden_sink)
        # The upstream model computes the vocabulary projection and MultiMax
        # in fp32, independently of the backbone dtype.
        logits = self.lm_head(h.astype(mx.float32))
        return LanguageModelOutput(
            logits=self.multimax(logits),
            hidden_states=hidden_sink,
            shared_kv_states={} if kwargs.get("return_shared_kv", False) else None,
        )

    def speculative_logits_from_hidden(self, hidden):
        h = hidden.reshape(
            *hidden.shape[:2], self.config.hc_mult, self.config.hidden_size
        )
        h = self.model.norm(self.model.hc_head(h))
        return self.multimax(self.lm_head(h.astype(mx.float32)))

    def speculative_verify_hidden(self, inputs, cache):
        transaction = start_speculative_cache(cache, inputs.shape[1])
        try:
            h = self.model(inputs, cache=cache, skip_final_norm=True)
        except BaseException:
            transaction.abort()
            raise
        return h.flatten(-2), {}, transaction

    def speculative_verify_logits(self, inputs, cache, sampler):
        hidden, shared, rollback = self.speculative_verify_hidden(inputs, cache)
        return (
            hidden,
            shared,
            rollback,
            sampler(self.speculative_logits_from_hidden(hidden)),
        )

    def rollback_speculative_cache(self, caches, gdn_states, accepted, block_size):
        return rollback_speculative_cache(caches, gdn_states, accepted, block_size)

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [
            (
                CacheList(KVCache(), KVCache())
                if ratio == -2
                else CacheList(
                    RotatingKVCache(max_size=self.config.sliding_window),
                    PoolingCache(ratio),
                )
            )
            for ratio in self.config.compress_ratios
        ]

    def shard(self, group=None):
        """Partition attention heads and FFN intermediate dimensions."""
        group = group or mx.distributed.init()
        size = group.size()
        if size == 1:
            return
        if self.sharding_group is not None:
            raise ValueError("The language model is already sharded")

        partitions = []

        def partition(module, axis):
            for name, value in tree_flatten(module.parameters()):
                # Splitting packed columns and their scales together requires
                # each rank to own complete quantization groups.
                if value.shape[axis] % size:
                    raise ValueError(
                        f"Cannot shard {type(module).__name__}.{name} "
                        f"with shape {value.shape} on axis {axis} across {size} ranks"
                    )
            partitions.append((module, axis))

        def partition_mlp(mlp):
            partition(mlp.gate_proj, -2)
            partition(mlp.up_proj, -2)
            partition(mlp.down_proj, -1)

        # Validate every dimension before mutating any weights or head counts.
        for layer in self.layers:
            attn = layer.attn
            if isinstance(attn, HCAAttention):
                if attn.o_groups % size or attn.n_heads % size:
                    raise ValueError(
                        "Rank count must divide HCA heads and output groups"
                    )
                partition(attn.wq_b, -2)
                partition(attn.wo_a, 0)
            else:
                if attn.heads % size:
                    raise ValueError("Rank count must divide MLA heads")
                partition(attn.q_b_proj, -2)
                partition(attn.embed_q, 0)
                partition(attn.unembed_out, 0)
            partition(attn.o_gate, -2)
            partition(attn.wo_b, -1)
            ffn = layer.ffn
            if isinstance(ffn, LatentMoE):
                partition_mlp(ffn.switch_mlp)
                partition_mlp(ffn.shared_experts)
            else:
                partition_mlp(ffn)

        for module, axis in partitions:
            shard_inplace(module, lambda path, value, axis=axis: axis, group=group)
        for layer in self.layers:
            attn = layer.attn
            attn.vha.sharding_group = group
            attn.attn_sink = mx.split(attn.attn_sink, size)[group.rank()]
            if isinstance(attn, HCAAttention):
                attn.n_heads //= size
                attn.o_groups //= size
            else:
                attn.heads //= size
            layer.ffn.sharding_group = group
            if isinstance(layer.ffn, LatentMoE):
                layer.ffn.shared_experts.sharding_group = group
                # Keep down-projection partial sums in FP32 until all-reduce,
                # while retaining the activation's original BF16 rounding.
                layer.ffn.switch_mlp.activation.output_dtype = mx.float32
        self.sharding_group = group

    def sanitize(self, weights):
        weights = {k.removeprefix("language_model."): v for k, v in weights.items()}
        out = {}
        for key, value in weights.items():
            match = re.search(r"model\.layers\.(\d+)\.", key)
            if match and int(match[1]) >= self.config.num_hidden_layers:
                continue  # ECHO is not part of the next-token backbone.
            if key.endswith("_scale_inv"):
                if key.removesuffix("_scale_inv") not in weights:
                    raise ValueError(f"FP8 scale without its weight: {key}")
                continue
            scale_key = key + "_scale_inv"
            if scale_key in weights:
                value = _dequantize_fp8_weight(value, weights[scale_key]).astype(
                    mx.bfloat16
                )
            elif value.dtype == mx.uint8 and not key.endswith(".scales"):
                raise ValueError(f"FP8 weight without its scale: {key}")

            if key == "model.mhc_contract.hc_head_fn":
                value = value.T
            key = key.replace("model.lm_head.", "multimax.")
            key = key.replace("model.mhc_contract.hc_head_", "model.hc_head.")
            for src, dst in (("self_attention", "attn"), ("mlp", "ffn")):
                prefix = f".{src}_hyper_connection."
                if prefix in key:
                    key = key.replace(prefix, f".{dst}_hc.")
                    if key.endswith("mapping_proj.weight"):
                        key = key.replace("mapping_proj.weight", "fn")
                        value = value.T
                    key = key.replace(f".{dst}_hc.bias", f".{dst}_hc.base")
            remap = {
                ".input_layernorm.": ".attn_norm.",
                ".post_attention_layernorm.": ".ffn_norm.",
                ".block_sparse_moe.": ".ffn.",
                ".mlp.": ".ffn.",
                ".self_attn.": ".attn.",
                ".core_attention.": ".",
                ".linear_q_down_proj.": ".wq_a.",
                ".linear_q_up_proj.": ".wq_b.",
                ".linear_kv_proj.": ".wkv.",
                ".q_layernorm.": ".q_norm.",
                ".kv_layernorm.": ".kv_norm.",
                ".compressor.linear_wkv.": ".compressor.wkv.",
                ".compressor.linear_wgate.": ".compressor.wgate.",
                ".attn.gate_proj.": ".attn.o_gate.",
                ".attn.o_proj.": ".attn.wo_b.",
                ".attn.vha_postmix_U": ".attn.vha.U",
                ".attn.vha_postmix_V": ".attn.vha.V",
                ".attn.softmax_offset": ".attn.attn_sink",
            }
            for src, dst in remap.items():
                key = key.replace(src, dst)
            if key.endswith(".attn.linear_o_group_proj"):
                key = key.replace(".linear_o_group_proj", ".wo_a.weight")
                value = value.reshape(self.config.o_groups, self.config.o_lora_rank, -1)
            for src, dst in (
                ("w1", "gate_proj"),
                ("w2", "down_proj"),
                ("w3", "up_proj"),
            ):
                key = key.replace(f".shared_experts.{src}.", f".shared_experts.{dst}.")
            out[key] = value

        for i in range(self.config.num_hidden_layers):
            prefix = f"model.layers.{i}"
            for kind in ("attn", "ffn"):
                keys = [
                    f"{prefix}.{kind}_hc.alpha_{part}"
                    for part in ("pre", "post", "res")
                ]
                if keys[0] in out:
                    out[f"{prefix}.{kind}_hc.scale"] = mx.concatenate(
                        [out.pop(k).reshape(-1) for k in keys]
                    )
            for src, dst in (
                ("w1", "gate_proj"),
                ("w2", "down_proj"),
                ("w3", "up_proj"),
            ):
                for suffix in ("weight", "scales", "biases"):
                    ep = f"{prefix}.ffn.experts"
                    if f"{ep}.0.{src}.{suffix}" in out:
                        out[f"{prefix}.ffn.switch_mlp.{dst}.{suffix}"] = mx.stack(
                            [
                                out.pop(f"{ep}.{e}.{src}.{suffix}")
                                for e in range(self.config.n_routed_experts)
                            ]
                        )
        weights = out
        for i, ratio in enumerate(self.config.compress_ratios):
            key = f"model.layers.{i}.attn.kv_b_proj.weight"
            if ratio == -2 and key in weights:
                attn = self.layers[i].attn
                w = weights.pop(key)
                scale_key = key.removesuffix("weight") + "scales"
                if scale_key in weights:
                    # The shared loader converts released FP8 linears to MXFP8.
                    # Restore this fused projection before transposing its K part,
                    # following the MLA layout conversion used by DeepSeek.
                    scales = weights.pop(scale_key)
                    biases = weights.pop(key.removesuffix("weight") + "biases", None)
                    w = mx.dequantize(
                        w,
                        scales,
                        biases,
                        group_size=attn.latent // scales.shape[-1],
                        bits=w.shape[-1] * 32 // attn.latent,
                        mode="mxfp8" if scales.dtype == mx.uint8 else "affine",
                    )
                w = w.reshape(attn.heads, attn.nope + attn.value_dim, attn.latent)
                k, v = mx.split(w, [attn.nope], axis=1)
                prefix = f"model.layers.{i}.attn."
                weights[prefix + "embed_q.weight"] = k.swapaxes(-1, -2)
                weights[prefix + "unembed_out.weight"] = v
        return weights

    @property
    def quant_predicate(self):
        def predicate(path, module, *, quantization=None):
            if ".switch_mlp." in path:
                requested = quantization or {
                    "group_size": 64,
                    "bits": 4,
                    "mode": "affine",
                }
                if requested["bits"] == 8:
                    return dict(MLX_MXFP8_QUANTIZATION)
                return dict(requested)
            if (
                ".shared_experts." in path
                or path.endswith(("fc1_latent_proj", "fc2_latent_proj"))
                or path in ("echo.e_proj", "echo.h_proj")
            ):
                return dict(MLX_MXFP8_QUANTIZATION)
            return False

        return predicate

    @property
    def cast_predicate(self):
        return lambda path: not any(
            part in path
            for part in (
                "_hc.",
                "hc_head.",
                ".gate.",
                "e_score_correction_bias",
                "attn_sink",
                ".compressor.ape",
                "multimax.",
                "lm_head.",
            )
        )
