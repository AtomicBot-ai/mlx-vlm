from __future__ import annotations

import mlx.core as mx
from mlx import nn

from ..minimax_h3.transformer import (
    MiniMaxH3AdaLayerNormOut,
    MiniMaxH3RotaryPosEmbed,
    MiniMaxH3TokenRefiner,
    MiniMaxH3Transformer,
    MiniMaxH3TransformerBlock,
    TimestepEmbedding,
)
from .config import MiniMaxH3ImageTransformerConfig


class MiniMaxH3ImageTransformer(MiniMaxH3Transformer):
    """H3's shared block stack with image-only inputs and per-step output heads."""

    def __init__(self, config: MiniMaxH3ImageTransformerConfig):
        nn.Module.__init__(self)
        self.config = config
        self.proj_in = nn.Linear(config.video_patch_dim, config.hidden_size)
        self.context_embedder = nn.Linear(config.text_dim, config.hidden_size)
        self.time_embedder = TimestepEmbedding(
            config.freq_dim, config.time_embed_hidden_dim, config.time_embed_dim
        )
        self.rope = MiniMaxH3RotaryPosEmbed(config.rope_freq_dim, config.rope_theta)
        self.token_refiner = MiniMaxH3TokenRefiner(config)
        self.transformer_blocks = [
            MiniMaxH3TransformerBlock(config) for _ in range(config.num_layers)
        ]
        self.norm_out = MiniMaxH3AdaLayerNormOut(
            config.hidden_size, config.time_embed_dim, config.final_norm_eps
        )
        self.pdd_proj_out = [
            nn.Linear(config.hidden_size, config.video_patch_dim)
            for _ in range(config.num_pdd_heads)
        ]
        self._adaln_cache_token = object()

    def __call__(
        self,
        hidden_states,
        encoder_hidden_states,
        timestep,
        timestep_indices,
        token_tags,
        position_ids,
        image_indices,
        text_indices,
        pdd_step_index,
        adaln_cache=None,
    ):
        if not 0 <= pdd_step_index < len(self.pdd_proj_out):
            raise ValueError("pdd_step_index must select one of the six output heads")
        length = position_ids.shape[0]
        if (
            position_ids.shape != (length, 3)
            or token_tags.shape != (length,)
            or timestep_indices.shape != (length,)
        ):
            raise ValueError(
                "Packed positions, tags and timestep indices must have matching lengths"
            )
        if adaln_cache is None:
            if not self.adaln_weights_available:
                raise RuntimeError(
                    "Pass an AdaLN cache after dropping projection weights"
                )
            temb = self._embed_timesteps(timestep)
        else:
            adaln_cache.validate(
                self._adaln_cache_token, timestep, len(self.transformer_blocks)
            )
            temb = None
        rotary = self.rope(position_ids)
        image = self.proj_in(
            hidden_states.astype(
                getattr(self.proj_in, "scales", self.proj_in.weight).dtype
            )
        )
        text = self.token_refiner(
            self.context_embedder(
                encoder_hidden_states.astype(
                    getattr(
                        self.context_embedder, "scales", self.context_embedder.weight
                    ).dtype
                )
            )
        )
        packed = mx.zeros((text.shape[0], length, text.shape[-1]), dtype=text.dtype)
        packed = packed.at[:, text_indices].add(text)
        packed = packed.at[:, image_indices].add(image.astype(text.dtype))
        # The checkpoint retains all three H3 modality slots, including the unused audio slot.
        indices = (timestep_indices * 3 + token_tags).astype(mx.int32)
        for index, block in enumerate(self.transformer_blocks):
            modulation = (
                block.adaln_proj(temb)
                if adaln_cache is None
                else adaln_cache.block_modulations[index]
            )
            packed = block(packed, modulation, indices, rotary)
            # Bound the lazy graph and activation lifetime for the 33B block stack.
            mx.eval(packed)
        modulation = (
            self.norm_out.project(temb)
            if adaln_cache is None
            else adaln_cache.output_modulation
        )
        packed = self.norm_out.apply(packed, modulation, timestep_indices)
        head = self.pdd_proj_out[pdd_step_index]
        # Only image rows are projected; text rows never contribute to the output.
        return head(mx.take(packed, image_indices, axis=1).astype(head.weight.dtype))
