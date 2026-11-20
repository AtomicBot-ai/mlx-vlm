"""Apple's shared focal-length estimator for the Depth Pro 2 family."""

import math

import mlx.core as mx
import mlx.nn as nn

from ..interpolate import resize_bilinear_nhwc
from .depth_pro_2 import INPUT_MULTIPLE, sanitize_convolutions
from .encoder import PREFIX_TOKENS, Block, DINOv3Trunk

FOCAL_CHECKPOINT = "apple/DepthPro2-focal"


class FOVHead(nn.Module):
    def __init__(self, trunk_dim):
        super().__init__()
        dim = 248
        self.tap_norms = [nn.LayerNorm(trunk_dim) for _ in range(4)]
        self.proj = nn.Linear(4 * trunk_dim, dim)
        self.query = mx.zeros((1, 1, dim))
        self.prefix_embed = mx.zeros((1, PREFIX_TOKENS, dim))
        self.blocks = [Block(dim, 4, dinov3=False) for _ in range(2)]
        self.norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, 2)

    def __call__(self, taps, grid):
        x = mx.concatenate([norm(t) for norm, t in zip(self.tap_norms, taps)], axis=-1)
        x = self.proj(x.astype(self.proj.weight.dtype))
        frames, _, dim = x.shape
        gh, gw = grid
        prefix = x[:, :PREFIX_TOKENS] + self.prefix_embed
        x = x[:, PREFIX_TOKENS:].reshape(frames, gh, gw, dim)
        # Focal inputs are multiples of 32, so the 16px patch grid is even.
        gh, gw = gh // 2, gw // 2
        x = x.reshape(frames, gh, 2, gw, 2, dim).mean(axis=(2, 4))
        x = x.reshape(frames, gh * gw, dim) + sincos_positions(gh, gw, dim).astype(
            x.dtype
        )
        query = mx.broadcast_to(self.query, (frames, 1, dim))
        x = mx.concatenate([query, prefix, x], axis=1)
        for block in self.blocks:
            x = block(x)
        return self.out(self.norm(x[:, 0])).astype(mx.float32)


def sincos_positions(height, width, dim):
    side = max(height, width)
    y, x = mx.meshgrid(
        (mx.arange(height) + 0.5 - height / 2) / side,
        (mx.arange(width) + 0.5 - width / 2) / side,
        indexing="ij",
    )
    frequencies = math.pi * 2.0 ** mx.linspace(0.0, 7.0, dim // 4)
    y = y.reshape(-1, 1) * frequencies
    x = x.reshape(-1, 1) * frequencies
    return mx.concatenate([mx.sin(y), mx.cos(y), mx.sin(x), mx.cos(x)], axis=-1)


def focal_length_from_fov(angles, height, width):
    # torch.median uses the lower of the two middle elements for even frame counts.
    axis = 0 if width < height else 1
    angle = mx.sort(angles[:, axis])[(angles.shape[0] - 1) // 2]
    return (0.5 * min(width, height)) / mx.tan(angle * (math.pi / 360))


class FocalLengthEstimator(nn.Module):
    """Predict horizontal and vertical field of view in degrees for each RGB frame."""

    def __init__(self):
        super().__init__()
        self.trunk = DINOv3Trunk(
            "vit_small_patch16_dinov3", (2, 5, 8, 11), (5, 7, 9, 11)
        )
        self.head = FOVHead(self.trunk.dim)

    @classmethod
    def from_pretrained(cls, path_or_repo=FOCAL_CHECKPOINT):
        from ...utils import get_model_path

        path = get_model_path(path_or_repo, allow_patterns=["model.safetensors"])
        model = cls()
        weights = mx.load(str(path / "model.safetensors"))
        model.load_weights(list(sanitize_convolutions(model, weights).items()))
        return model.eval()

    def __call__(self, pixel_values):
        _, height, width, _ = pixel_values.shape
        multiple = INPUT_MULTIPLE
        short = max(multiple, min(512, min(height, width) // multiple * multiple))
        scale = short / min(height, width)
        full_h = max(multiple, round(height * scale / multiple) * multiple)
        full_w = max(multiple, round(width * scale / multiple) * multiple)
        resized = resize_bilinear_nhwc(pixel_values, (full_h, full_w), antialias=True)

        short, long = min(full_h, full_w), max(full_h, full_w)
        if long > 1.7778 * short:
            keep = max(multiple, int(1.7778 * short) // multiple * multiple)
            start = (long - keep) // 2
            if full_w > full_h:
                resized = resized[:, :, start : start + keep]
            else:
                resized = resized[:, start : start + keep, :]

        features, grid = self.trunk.encode(resized, independent=True)
        angles = 1.0 + mx.sigmoid(self.head(features, grid)) * 178.0
        crop_h, crop_w = resized.shape[1:3]
        if (crop_h, crop_w) == (full_h, full_w):
            return angles
        # Restore the full frame's FoV at the same focal length after the center crop.
        scale = mx.array([full_w / crop_w, full_h / crop_h])
        return mx.arctan(scale * mx.tan(angles * (math.pi / 360))) * (360 / math.pi)

    def focal_length_px(self, pixel_values):
        return focal_length_from_fov(self(pixel_values), *pixel_values.shape[1:3])
