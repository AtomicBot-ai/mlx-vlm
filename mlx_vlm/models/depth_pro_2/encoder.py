"""DINOv3 blocks with spatial, joint, and causal cross-frame attention."""

import math

import mlx.core as mx
import mlx.nn as nn

from ..sam3d_body.rope import apply_rope
from .config import BACKBONES

PATCH_SIZE = 16
PREFIX_TOKENS = 5


def rotary_embeddings(height, width, head_dim):
    # The checkpoint normalizes both coordinates by the longer patch-grid side.
    side = max(height, width)
    y, x = mx.meshgrid(
        (mx.arange(height) + 0.5) * (2.0 / side) - 1.0,
        (mx.arange(width) + 0.5) * (2.0 / side) - 1.0,
        indexing="ij",
    )
    periods = 100.0 ** (mx.arange(head_dim // 4) * (4.0 / head_dim))
    angles = 2 * math.pi * mx.stack([y, x], axis=-1).reshape(-1, 2, 1) / periods
    angles = angles.reshape(height * width, head_dim // 2)
    return mx.tile(mx.sin(angles), (1, 2)), mx.tile(mx.cos(angles), (1, 2))


class Attention(nn.Module):
    def __init__(self, dim, heads, bias):
        super().__init__()
        self.num_heads = heads
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, 3 * dim, bias=bias)
        self.proj = nn.Linear(dim, dim)

    def project(self, x, rope):
        batch, tokens, _ = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, -1)
        q, k, v = mx.split(qkv.transpose(2, 0, 3, 1, 4), 3, axis=0)
        q, k, v = q[0], k[0], v[0]
        if rope is not None:
            sin, cos = rope
            q, k = apply_rope(q, k, sin, cos, tokens - sin.shape[0])
        return q.astype(v.dtype), k.astype(v.dtype), v

    def attend(self, q, k, v):
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        batch, _, tokens, _ = out.shape
        return self.proj(out.transpose(0, 2, 1, 3).reshape(batch, tokens, -1))


class MLP(nn.Module):
    def __init__(self, dim, gated):
        super().__init__()
        self.gated = gated
        if gated:
            self.fc1_g = nn.Linear(dim, dim * 4)
            self.fc1_x = nn.Linear(dim, dim * 4)
        else:
            self.fc1 = nn.Linear(dim, dim * 4)
        self.fc2 = nn.Linear(dim * 4, dim)

    def __call__(self, x):
        if self.gated:
            x = nn.silu(self.fc1_g(x)) * self.fc1_x(x)
        else:
            x = nn.gelu(self.fc1(x))
        return self.fc2(x)


class Block(nn.Module):
    def __init__(self, dim, heads, *, gated=False, dinov3=True):
        super().__init__()
        eps = 1e-5 if dinov3 else 1e-6
        self.norm1 = nn.LayerNorm(dim, eps=eps)
        self.norm2 = nn.LayerNorm(dim, eps=eps)
        self.attn = Attention(dim, heads, bias=not dinov3)
        self.mlp = MLP(dim, gated)
        self.layer_scale = dinov3
        if dinov3:
            self.gamma_1 = mx.full((dim,), 1e-5)
            self.gamma_2 = mx.full((dim,), 1e-5)

    def __call__(self, x, rope=None):
        return self.finish(x, *self.attn.project(self.norm1(x), rope))

    def finish(self, x, q, k, v):
        attention = self.attn.attend(q, k, v)
        x = x + (self.gamma_1 * attention if self.layer_scale else attention)
        residual = self.mlp(self.norm2(x))
        return x + (self.gamma_2 * residual if self.layer_scale else residual)


class CrossFrameCache:
    """One bounded ring of prefixes and patch keys/values per cross-frame block."""

    def __init__(self, window):
        self.window = window
        self.layers = {}

    def update(self, index, block, prefix, key, value):
        if index not in self.layers:
            _, heads, patches, dim = key.shape
            shape = (1, heads, PREFIX_TOKENS + self.window * patches, dim)
            self.layers[index] = (
                0,
                mx.zeros((self.window, *prefix.shape), dtype=prefix.dtype),
                mx.zeros(shape, dtype=key.dtype),
                mx.zeros(shape, dtype=value.dtype),
            )
        seen, prefixes, keys, values = self.layers[index]
        slot = seen % self.window
        count = min(seen + 1, self.window)
        prefixes[slot] = prefix
        pooled = prefixes[:count].mean(axis=0)
        _, pooled_key, pooled_value = block.attn.project(block.norm1(pooled), None)

        patches = key.shape[2]
        start = PREFIX_TOKENS + slot * patches
        keys[:, :, start : start + patches] = key
        values[:, :, start : start + patches] = value
        keys[:, :, :PREFIX_TOKENS] = pooled_key
        values[:, :, :PREFIX_TOKENS] = pooled_value
        self.layers[index] = (seen + 1, prefixes, keys, values)
        end = PREFIX_TOKENS + count * patches
        return keys[:, :, :end], values[:, :, :end]

    def materialize(self, inverse_depth):
        # Cut lazy graph dependencies on frames that have left the ring.
        mx.eval(inverse_depth, [state[1:] for state in self.layers.values()])


class PatchEmbed(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Conv2d(3, dim, PATCH_SIZE, stride=PATCH_SIZE)

    def __call__(self, x):
        return self.proj(x)


class DINOv3Trunk(nn.Module):
    def __init__(self, preset, taps, cross_frame_blocks):
        super().__init__()
        self.dim, depth, heads, gated = BACKBONES[preset]
        self.head_dim = self.dim // heads
        self.patch_embed = PatchEmbed(self.dim)
        self.cls_token = mx.zeros((1, 1, self.dim))
        self.reg_token = mx.zeros((1, PREFIX_TOKENS - 1, self.dim))
        self.blocks = [Block(self.dim, heads, gated=gated) for _ in range(depth)]
        self.taps = frozenset(taps)
        self.cross_frame_blocks = frozenset(cross_frame_blocks)

    def encode(self, x, *, independent=False, cache=None):
        patches = self.patch_embed(x.astype(self.patch_embed.proj.weight.dtype))
        frames, height, width, dim = patches.shape
        prefix = mx.concatenate([self.cls_token, self.reg_token], axis=1)
        prefix = mx.broadcast_to(prefix, (frames, PREFIX_TOKENS, dim))
        tokens = mx.concatenate([prefix, patches.reshape(frames, -1, dim)], axis=1)
        rope = rotary_embeddings(height, width, self.head_dim)
        joint_rope = tuple(mx.tile(part, (frames, 1)) for part in rope)
        features = []
        for i, block in enumerate(self.blocks):
            if i not in self.cross_frame_blocks:
                tokens = block(tokens, rope)
            else:
                prefix, patches = tokens[:, :PREFIX_TOKENS], tokens[:, PREFIX_TOKENS:]
                if cache is not None:
                    q, k, v = block.attn.project(block.norm1(patches), rope)
                    k, v = cache.update(i, block, prefix, k, v)
                    patches = block.finish(patches, q, k, v)
                elif independent or frames == 1:
                    patches = block(tokens, rope)[:, PREFIX_TOKENS:]
                else:
                    pooled = prefix.mean(axis=0, keepdims=True)
                    joint = mx.concatenate(
                        [pooled, patches.reshape(1, -1, dim)], axis=1
                    )
                    patches = block(joint, joint_rope)[:, PREFIX_TOKENS:].reshape(
                        frames, height * width, dim
                    )
                tokens = mx.concatenate([prefix, patches], axis=1)
            if i in self.taps:
                features.append(tokens)
        return features, (height, width)


class MultiViewEncoder(DINOv3Trunk):
    def __init__(self, config):
        super().__init__(
            config.backbone_preset, config.reassembled_blocks, config.cross_frame_blocks
        )
        channels = config.dims_reassemble
        self.reassemble = [[nn.Conv2d(self.dim, dim, 1)] for dim in channels]
        for index, factor in enumerate((4, 2)):
            dim = channels[index]
            self.reassemble[index].append(
                nn.ConvTranspose2d(dim, dim, factor, stride=factor)
            )
        dim = channels[-1]
        self.reassemble[-1].append(nn.Conv2d(dim, dim, 3, stride=2, padding=1))

    def decode_features(self, tokens, grid):
        pyramid = []
        for feature, layers in zip(tokens, self.reassemble):
            feature = feature[:, PREFIX_TOKENS:].reshape(-1, *grid, self.dim)
            for layer in layers:
                feature = layer(feature)
            pyramid.append(feature)
        return pyramid
