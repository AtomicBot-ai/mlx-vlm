# Copyright © 2026 Neywa Labs Inc.

"""ECHO autoregressive draft using the target's shared final MLA/MoE layer."""

import mlx.core as mx

from ....models.cache import BatchKVCache, KVCache
from ....models.ernie5_5_moe.echo import EchoWeights


class Ernie55MTPDraftModel(EchoWeights):
    supports_greedy_draft_argmax = True
    prefer_requested_block_size = True
    requires_uniform_batch_acceptance = True

    def __init__(self, config):
        super().__init__(config.text_config)
        self.config = config
        self._target = None
        self._cache = []
        self._seed_token = self._seed_hidden = None
        self._round_appended = 0
        self._kv_valid_len = 0
        self.accept_lens = []
        self.draft_lens = []

    def validate_target_compatibility(self, target_model):
        target = getattr(target_model, "language_model", target_model)
        config = target.config
        own = self.config.text_config
        fields = (
            "hidden_size",
            "vocab_size",
            "num_hidden_layers",
            "hc_mult",
            "n_routed_experts",
        )
        if config.model_type != "ernie5_5_moe" or any(
            getattr(config, k) != getattr(own, k) for k in fields
        ):
            raise ValueError("ECHO drafter is incompatible with the target model")
        if config.compress_ratios[-1] != -2:
            raise ValueError("ECHO requires a final MLA layer")

    def bind(self, target_model):
        self.validate_target_compatibility(target_model)
        self._target = getattr(target_model, "language_model", target_model)
        return self

    def reset(self, target_model):
        self.bind(target_model)
        self._cache = [KVCache(), KVCache()]
        self._seed_token = self._seed_hidden = None
        self._round_appended = 0
        self._kv_valid_len = 0
        self.accept_lens = []
        self.draft_lens = []
        return self._cache

    def set_shared_kv(
        self,
        shared_kv_states,
        kv_offset,
        position=None,
        kv_valid_len=None,
        left_padding=None,
    ):
        # ECHO shares decoder parameters, but maintains its own latent KV state.
        self._kv_valid_len = kv_offset if kv_valid_len is None else kv_valid_len

    def draft_eval_state(self):
        return [self._seed_token, self._seed_hidden, [c.state for c in self._cache]]

    def _forward_tokens(self, tokens, hidden):
        config = self.config.text_config
        h = hidden.reshape(*tokens.shape, config.hc_mult, config.hidden_size)
        embeds = self._target.model.embed_tokens(tokens)
        h = self.h_proj(self.hnorm(h)) + self.e_proj(self.enorm(embeds))[:, :, None, :]
        h = self._target.layers[-1](
            h, self._cache, router_bias=self.e_score_correction_bias
        )
        # Prefill and accepted-prefix replay need every KV update, but only the
        # last position seeds the next proposal. Avoid a prompt-wide vocab head.
        out = self.norm(self.hc_head(h[:, -1:]))
        logits = self._target.multimax(self._target.lm_head(out.astype(mx.float32)))
        return logits, h.flatten(-2)

    def _seed(self, logits, hidden, sampler, greedy):
        self._seed_token = (
            mx.argmax(logits[:, -1:], axis=-1) if greedy else sampler(logits[:, -1:])
        )
        self._seed_hidden = hidden[:, -1:]

    def prefill_from_target_hidden(
        self,
        input_ids,
        hidden,
        bonus_token,
        sampler,
        token_dtype=mx.int32,
        greedy=False,
    ):
        batch, length = input_ids.shape
        if length != hidden.shape[1]:
            raise ValueError(
                "ECHO prefill requires aligned prompt tokens and target hidden states"
            )
        valid = self._kv_valid_len
        lengths = [int(valid)] * batch if isinstance(valid, int) else valid.tolist()
        padding = [length - int(n) for n in lengths]
        if min(padding) < 0:
            raise ValueError(
                "ECHO requires full prompt hidden states; chunked speculative prefill is unsupported"
            )
        self._cache = (
            [BatchKVCache(padding), BatchKVCache(padding)]
            if batch > 1
            else [KVCache(), KVCache()]
        )
        bonus = mx.array(bonus_token, dtype=token_dtype).reshape(batch, 1)
        shifted = mx.concatenate([input_ids[:, 1:], bonus], axis=1)
        logits, h = self._forward_tokens(shifted, hidden)
        self._seed(logits, h, sampler, greedy)

    def draft_block(
        self,
        last_bonus,
        hidden,
        cache,
        block_size,
        sampler,
        token_dtype=mx.int32,
        greedy=False,
    ):
        if self._seed_token is None:
            raise RuntimeError(
                "ECHO must be prefilled from target hidden states before drafting"
            )
        tokens = [self._seed_token.astype(token_dtype)]
        h = self._seed_hidden
        self._seed_token = self._seed_hidden = None
        self._round_appended = 0
        while len(tokens) < block_size - 1:
            logits, h = self._forward_tokens(tokens[-1], h)
            tok = mx.argmax(logits, axis=-1) if greedy else sampler(logits)
            tokens.append(tok.astype(token_dtype))
            self._round_appended += 1
        return mx.concatenate(tokens, axis=1)

    def accept_verified_tokens_batch(
        self,
        verify_hidden,
        draft_tokens,
        accepted,
        new_tokens,
        sampler,
        token_dtype=mx.int32,
        greedy=False,
    ):
        if len(set(accepted)) != 1:
            raise ValueError("ECHO requires uniform batch acceptance")
        for cache in self._cache:
            cache.trim(self._round_appended)
        count = int(accepted[0])
        bonus = mx.array([[row[-1]] for row in new_tokens], dtype=token_dtype)
        tokens = mx.concatenate([draft_tokens[:, :count], bonus], axis=1)
        # Rebuild accepted draft KVs with verified target conditioning. Keeping
        # recursively drafted hidden states here would change later predictions.
        logits, h = self._forward_tokens(tokens, verify_hidden[:, : count + 1])
        self._seed(logits, h, sampler, greedy)
        self._round_appended = 0

    def accept_verified_tokens(
        self,
        verify_hidden,
        draft_tokens,
        accepted,
        new_tokens,
        sampler,
        token_dtype=mx.int32,
        greedy=False,
    ):
        return self.accept_verified_tokens_batch(
            verify_hidden,
            draft_tokens,
            [accepted],
            [new_tokens],
            sampler,
            token_dtype,
            greedy,
        )

    def filter_batch(self, keep):
        for cache in self._cache:
            cache.filter(keep)
        if self._seed_token is not None:
            self._seed_token = self._seed_token[keep]
            self._seed_hidden = self._seed_hidden[keep]

    @property
    def quant_predicate(self):
        return lambda path, module: False
