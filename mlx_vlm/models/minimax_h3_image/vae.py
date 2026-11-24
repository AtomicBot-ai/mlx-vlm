from __future__ import annotations

import math

import mlx.core as mx
from mlx import nn

from ..minimax_h3.visual_vae import (
    MiniMaxH3DiagonalGaussianDistribution,
    MiniMaxH3VideoCausalConv3d,
    MiniMaxH3VideoEncoder3d,
    MiniMaxH3VideoTransformerBlock,
    MiniMaxH3VideoVAE,
    MiniMaxH3VideoVAEOutput,
)
from .config import MiniMaxH3ImageVAEConfig


class MiniMaxH3ImageViTDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        dim = config.decoder_num_attention_heads * config.decoder_attention_head_dim
        self.proj_in = nn.Linear(config.latent_channels, dim)
        self.register_tokens = mx.zeros((1, config.decoder_num_register_tokens, dim))
        self.transformer_blocks = [
            MiniMaxH3VideoTransformerBlock(
                dim,
                config.decoder_num_attention_heads,
                config.decoder_attention_head_dim,
                config.decoder_ffn_mult,
                config.decoder_norm_eps,
            )
            for _ in range(config.decoder_num_layers)
        ]
        self.norm_out = nn.LayerNorm(dim, eps=config.decoder_norm_eps)
        self.proj_out = nn.Linear(
            dim, config.out_channels * config.spatial_compression_ratio**2
        )

    def __call__(self, latents):
        batch, channels, height, width = latents.shape
        x = self.proj_in(latents.transpose(0, 2, 3, 1).reshape(batch, -1, channels))
        count = x.shape[1]
        registers = mx.broadcast_to(
            self.register_tokens.astype(x.dtype),
            (batch, *self.register_tokens.shape[1:]),
        )
        x = mx.concatenate([x, registers, mx.zeros_like(x[:, :1])], axis=1)
        grids = [
            2 * (mx.arange(size, dtype=mx.float32) + 0.5) / size - 1
            for size in (height, width)
        ]
        positions = mx.stack(mx.meshgrid(*grids, indexing="ij"), axis=-1).reshape(
            1, -1, 2
        )
        positions = mx.concatenate(
            [positions, mx.zeros((1, registers.shape[1] + 1, 2))], axis=1
        )
        dim = int(
            self.config.decoder_attention_head_dim * self.config.decoder_rope_dim_ratio
        )
        frequency = 1 / mx.power(
            self.config.decoder_rope_theta,
            mx.arange(dim // 4, dtype=mx.float32) * (4 / dim),
        )
        angles = (2 * math.pi * positions[..., None] * frequency).reshape(
            1, -1, dim // 2
        )
        angles = mx.concatenate([angles, angles], axis=-1)[:, :, None]
        rotary = mx.cos(angles), mx.sin(angles)
        for block in self.transformer_blocks:
            x = block(x, rotary)
        x = self.proj_out(self.norm_out(x))[:, :count]
        patch = self.config.spatial_compression_ratio
        return (
            x.reshape(batch, height, width, self.config.out_channels, patch, patch)
            .transpose(0, 3, 1, 4, 2, 5)
            .reshape(batch, self.config.out_channels, height * patch, width * patch)
        )


class _ImageConv2d(nn.Conv2d):
    def __call__(self, x):
        return super().__call__(x.transpose(0, 2, 3, 1)).transpose(0, 3, 1, 2)


class MiniMaxH3ImageVAE(MiniMaxH3VideoVAE):
    """Shared single-frame CNN encoder, image ViT decoder, and blended tiling."""

    def __init__(self, config: MiniMaxH3ImageVAEConfig):
        nn.Module.__init__(self)
        self.config = config
        self.encoder = MiniMaxH3VideoEncoder3d(config)
        self.quant_conv = MiniMaxH3VideoCausalConv3d(
            2 * config.latent_channels, 2 * config.latent_channels, 1
        )
        self.post_quant_conv = _ImageConv2d(
            config.latent_channels, config.latent_channels, 1
        )
        self.decoder = MiniMaxH3ImageViTDecoder(config)
        self.use_tiling = True
        self.tile_sample_min_height = self.tile_sample_min_width = 256
        self.tile_sample_min_overlap_height = self.tile_sample_min_overlap_width = 64

    def encode(self, pixels):
        return MiniMaxH3DiagonalGaussianDistribution(
            self._encode_clip(pixels[:, :, None])[:, :, -1]
        )

    def decode(self, latents):
        if not self.use_tiling:
            return MiniMaxH3VideoVAEOutput(self.decoder(self.post_quant_conv(latents)))
        ratio = self.config.spatial_compression_ratio
        ys, yl, yo = self._split_tiles(
            latents.shape[-2] * ratio,
            self.tile_sample_min_height,
            self.tile_sample_min_overlap_height,
        )
        xs, xl, xo = self._split_tiles(
            latents.shape[-1] * ratio,
            self.tile_sample_min_width,
            self.tile_sample_min_overlap_width,
        )
        tiles = []
        for y, h in zip(ys, yl):
            row = []
            for x, w in zip(xs, xl):
                tile = latents[
                    ..., y // ratio : (y + h) // ratio, x // ratio : (x + w) // ratio
                ]
                decoded = self.decoder(self.post_quant_conv(tile))[:, :, None]
                mx.eval(decoded)
                row.append(decoded)
            tiles.append(row)
        return MiniMaxH3VideoVAEOutput(self._stitch_tiles(tiles, yo, xo)[:, :, 0])

    @staticmethod
    def sanitize(weights):
        return {
            key: (
                value.transpose(0, 2, 3, 4, 1)
                if value.ndim == 5
                else value.transpose(0, 2, 3, 1) if value.ndim == 4 else value
            )
            for key, value in weights.items()
        }
