from functools import lru_cache
from typing import List, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..attention import VisionAttention as Attention
from ..base import check_array_shape
from ..mlp import GELUMLP as MLP
from .config import VisionConfig


class EncoderLayer(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.self_attn = Attention(
            config.hidden_size, config.num_attention_heads, bias=True
        )
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = MLP(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def __call__(self, x: mx.array) -> mx.array:
        h = x + self.self_attn(self.layer_norm1(x))
        return h + self.mlp(self.layer_norm2(h))


class Encoder(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.layers = [EncoderLayer(config) for _ in range(config.num_hidden_layers)]


class VisionEmbeddings(nn.Module):
    """Patch embeddings plus learned positions for square tiles of image_size."""

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )
        self.num_positions = (config.image_size // config.patch_size) ** 2
        self.position_embedding = nn.Embedding(self.num_positions, config.hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        # x: (num_tiles, height, width, channels)
        embeddings = mx.flatten(self.patch_embedding(x), start_axis=1, end_axis=2)
        return embeddings + self.position_embedding.weight[: embeddings.shape[1]]


@lru_cache(maxsize=8)
def sincos_position_embedding(embed_dim: int, grid_size: int) -> np.ndarray:
    """Fixed 2D sine-cosine embedding of a grid_size x grid_size token grid.

    The rows are in row-major order. Each row is [sin_w | cos_w | sin_h | cos_h],
    the layout the checkpoints were trained with.
    """
    if embed_dim % 4 != 0:
        raise ValueError(f"embed_dim must be divisible by 4, got {embed_dim}")
    pos_dim = embed_dim // 4
    omega = 1.0 / 10000.0 ** (np.arange(pos_dim, dtype=np.float64) / pos_dim)
    grid_h, grid_w = np.meshgrid(
        np.arange(grid_size, dtype=np.float64),
        np.arange(grid_size, dtype=np.float64),
        indexing="ij",
    )
    emb_h = np.outer(grid_h.reshape(-1), omega)
    emb_w = np.outer(grid_w.reshape(-1), omega)
    return np.concatenate(
        [np.sin(emb_w), np.cos(emb_w), np.sin(emb_h), np.cos(emb_h)], axis=1
    ).astype(np.float32)


def pixel_shuffle(x: mx.array, scale_factor: int) -> mx.array:
    bsz, seq, embed_dim = x.shape
    height = width = int(seq**0.5)
    x = x.reshape(bsz, height, width // scale_factor, embed_dim * scale_factor)
    x = x.transpose(0, 2, 1, 3)
    x = x.reshape(
        bsz,
        width // scale_factor,
        height // scale_factor,
        embed_dim * scale_factor**2,
    )
    x = x.transpose(0, 2, 1, 3)
    return x.reshape(bsz, seq // scale_factor**2, embed_dim * scale_factor**2)


class DeepStackMerger(nn.Module):
    def __init__(self, config: VisionConfig, scale_factor: int):
        super().__init__()
        merged_dim = config.hidden_size * scale_factor**2
        self.layer_norm = nn.LayerNorm(merged_dim)
        self.linear_1 = nn.Linear(merged_dim, config.out_hidden_size)
        self.linear_2 = nn.Linear(config.out_hidden_size, config.out_hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        x = nn.gelu(self.linear_1(self.layer_norm(x)))
        return self.linear_2(x)


class ConnectorPath(nn.Module):
    """Pixel shuffle, projection, fixed 2D positions and DeepStack mergers."""

    def __init__(self, config: VisionConfig, scale_factor: int):
        super().__init__()
        self.scale_factor = scale_factor
        self.grid_size = (config.image_size // config.patch_size) // scale_factor
        self.embed_dim = config.out_hidden_size
        self.projection = nn.Linear(
            config.hidden_size * scale_factor**2, config.out_hidden_size, bias=False
        )
        self.deepstack_mergers = [
            DeepStackMerger(config, scale_factor)
            for _ in config.deepstack_visual_indexes
        ]

    def __call__(
        self, hidden_states: mx.array, deepstack_hidden_states: List[mx.array]
    ) -> Tuple[mx.array, List[mx.array]]:
        x = self.projection(pixel_shuffle(hidden_states, self.scale_factor))
        pos_embed = sincos_position_embedding(self.embed_dim, self.grid_size)
        x = x + mx.array(pos_embed).astype(x.dtype)
        deepstack_features = [
            merger(pixel_shuffle(h, self.scale_factor)).reshape(-1, self.embed_dim)
            for merger, h in zip(self.deepstack_mergers, deepstack_hidden_states)
        ]
        return x, deepstack_features


class Connector(nn.Module):
    """Turns per-tile vision features into image tokens.

    The coarse path uses scale_factor. The optional fine path uses half of it
    and gives four times as many tokens per tile. The layer norms and the
    output projection are shared.
    """

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.ln_in = nn.LayerNorm(config.hidden_size)
        self.coarse = ConnectorPath(config, config.scale_factor)
        if config.use_fine_route:
            self.fine = ConnectorPath(config, config.scale_factor // 2)
        self.ln_mid = nn.LayerNorm(config.out_hidden_size)
        self.mlp_fc2 = nn.Linear(
            config.out_hidden_size, config.out_hidden_size, bias=False
        )
        self.ln_out = nn.LayerNorm(config.out_hidden_size)

    def __call__(
        self,
        hidden_states: mx.array,
        deepstack_hidden_states: List[mx.array],
        fine: bool = False,
    ) -> Tuple[mx.array, List[mx.array]]:
        if fine and "fine" not in self:
            raise ValueError("This checkpoint has no fine connector path.")
        path = self.fine if fine else self.coarse
        x, deepstack_features = path(self.ln_in(hidden_states), deepstack_hidden_states)
        x = self.ln_out(self.mlp_fc2(nn.gelu(self.ln_mid(x))))
        return x.reshape(-1, x.shape[-1]), deepstack_features


class VisionModel(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.embeddings = VisionEmbeddings(config)
        self.encoder = Encoder(config)
        self.post_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.connector = Connector(config)

    def encode(self, pixel_values: mx.array) -> Tuple[mx.array, List[mx.array]]:
        """Vision tokens and DeepStack taps, before the connector."""
        x = self.embeddings(pixel_values)
        taps = set(self.config.deepstack_visual_indexes)
        deepstack_hidden_states = []
        for layer_idx, layer in enumerate(self.encoder.layers):
            x = layer(x)
            if layer_idx in taps:
                deepstack_hidden_states.append(x)
        return self.post_layernorm(x), deepstack_hidden_states

    def __call__(
        self, pixel_values: mx.array, fine: bool = False
    ) -> Tuple[mx.array, List[mx.array]]:
        """Image tokens of every tile and one DeepStack feature per tap.

        pixel_values: (num_tiles, height, width, channels), channels last.
        """
        pixel_values = pixel_values.astype(self.embeddings.patch_embedding.weight.dtype)
        hidden_states, deepstack_hidden_states = self.encode(pixel_values)
        return self.connector(hidden_states, deepstack_hidden_states, fine=fine)

    def sanitize(self, weights):
        sanitized = {}
        for k, v in weights.items():
            if "position_ids" in k:
                continue
            if "patch_embedding.weight" in k and not check_array_shape(v):
                # PyTorch conv weights are (out, in, kH, kW); MLX uses (out, kH, kW, in).
                v = v.transpose(0, 2, 3, 1)
            sanitized[k] = v
        return sanitized
