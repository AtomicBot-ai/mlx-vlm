from __future__ import annotations

from dataclasses import dataclass

from ..minimax_h3.config import MiniMaxH3TransformerConfig, MiniMaxH3VideoVAEConfig


@dataclass(frozen=True, slots=True)
class MiniMaxH3ImageTransformerConfig(MiniMaxH3TransformerConfig):
    num_pdd_heads: int = 6

    def __post_init__(self):
        super(MiniMaxH3ImageTransformerConfig, self).__post_init__()
        if self.patch_size[0] != 1 or self.num_pdd_heads != 6:
            raise ValueError("H3-Image requires spatial patches and six PDD heads")


@dataclass(frozen=True, slots=True)
class MiniMaxH3ImageVAEConfig(MiniMaxH3VideoVAEConfig):
    decoder_num_layers: int = 24
    decoder_num_attention_heads: int = 16
    decoder_rope_dim_ratio: float = 0.5
    clip_length: int = 1
    token_drop: int = 0

    def __post_init__(self):
        # The image decoder rotates two spatial axes, rather than three axes.
        if not self.block_out_channels:
            raise ValueError("VAE needs at least one encoder level")
        levels = len(self.block_out_channels)
        if (
            len(self.spatial_downsample_factors) != levels
            or len(self.temporal_downsample_factors) != levels
        ):
            raise ValueError("VAE downsample factors must match encoder levels")
        if (
            len(self.latents_mean) != self.latent_channels
            or len(self.latents_std) != self.latent_channels
        ):
            raise ValueError("VAE latent statistics must match latent_channels")
        rotary_dim = int(self.decoder_attention_head_dim * self.decoder_rope_dim_ratio)
        if (
            rotary_dim <= 0
            or rotary_dim % 4
            or rotary_dim > self.decoder_attention_head_dim
        ):
            raise ValueError(
                "Image decoder rotary dimension must be a positive multiple of four within the head"
            )
