# Copyright © 2026 Neywa Labs Inc.

import mlx.core as mx
import mlx.nn as nn

from ....models.bailing_hybrid.language import DecoderLayer, sanitize_weights
from ....models.base import create_attention_mask
from ....models.cache import BatchKVCache, KVCache
from ..mtp_base import AutoregressiveMTPDraftModel


class NextNLayer(DecoderLayer):
    def __init__(self, args):
        super().__init__(args, args.num_hidden_layers)
        self.enorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.hnorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.eh_proj = nn.Linear(2 * args.hidden_size, args.hidden_size, bias=False)
        self.final_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def project_inputs(self, embeddings, hidden):
        return self.eh_proj(
            mx.concatenate([self.enorm(embeddings), self.hnorm(hidden)], axis=-1)
        )

    def __call__(self, embeddings, hidden, cache):
        h = self.project_inputs(embeddings, hidden)
        mask = create_attention_mask(h, cache, return_array=True)
        return self.final_layernorm(super().__call__(h, mask, cache))

    def cache_prefix(self, embeddings, hidden, cache):
        # Past positions only need K/V; their attention and expert outputs
        # do not contribute to the next proposal.
        h = self.input_layernorm(self.project_inputs(embeddings, hidden))
        attention = self.attention
        latent, k_pe = mx.split(
            attention.kv_a_proj_with_mqa(h), [attention.args.kv_lora_rank], axis=-1
        )
        latent = attention.kv_a_layernorm(latent)[:, None]
        k_pe = attention.rope(k_pe[:, None], offset=cache.offset)
        cache.update_and_fetch(latent, k_pe)


class Model(AutoregressiveMTPDraftModel):
    supports_ragged_batch_acceptance = False

    def __init__(self, config):
        super().__init__(config)
        self.mtp = NextNLayer(config.text_config)

    def validate_target_compatibility(self, target_model):
        target = getattr(target_model, "language_model", target_model)
        args = self.config.text_config
        for name in ("model_type", "hidden_size", "vocab_size", "num_hidden_layers"):
            if getattr(target.config, name) != getattr(args, name):
                raise ValueError(f"Ling MTP target {name} does not match")

    def bind(self, target_model):
        target = getattr(target_model, "language_model", target_model)
        self._input_embed = target.model.word_embeddings
        self._lm_head_fn = target.logits_from_hidden
        return self

    def make_cache(self, left_padding=None):
        return [KVCache() if left_padding is None else BatchKVCache(left_padding)]

    def reset(self, target_model, left_padding=None):
        super().reset(target_model)
        self._cache = self.make_cache(left_padding)
        return self._cache

    def _forward_hidden(self, token_embed, hidden, tokens, cache, **kwargs):
        h = self.mtp(token_embed, hidden, cache[0])
        return h, h

    def _prepare_verified_cache(
        self, verify_hidden, draft_tokens, accepted, token_dtype
    ):
        # Replace speculative K/V, which used estimated hidden states, with
        # the verified target states before producing the next proposal.
        super()._prepare_verified_cache(verify_hidden, draft_tokens, 0, token_dtype)
        if accepted:
            self.mtp.cache_prefix(
                self._input_embed(draft_tokens[:, :accepted].astype(token_dtype)),
                verify_hidden[:, :accepted],
                self._cache[0],
            )
            self._next_position += accepted
        return accepted

    def sanitize(self, weights):
        return sanitize_weights(
            dict(weights), self.config.text_config, [("mtp", False)]
        )

    @property
    def quant_predicate(self):
        return lambda path, module: not path.endswith(
            ("mlp.gate", "attention.g_proj", "eh_proj")
        )
