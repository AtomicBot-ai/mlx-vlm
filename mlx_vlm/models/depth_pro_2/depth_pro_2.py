"""Depth Pro 2 forward pass and Apple checkpoint layout conversion."""

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from .encoder import PATCH_SIZE, MultiViewEncoder

INPUT_MULTIPLE = PATCH_SIZE * 2


def sanitize_convolutions(model, weights):
    """Relayout PyTorch convolution kernels, preserving weights saved by MLX."""
    weights = dict(weights)
    for path, module in tree_flatten(model.leaf_modules(), is_leaf=nn.Module.is_module):
        if not isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            continue
        name = f"{path}.weight"
        kernel = weights.get(name)
        if kernel is None or kernel.ndim != 4 or kernel.shape == module.weight.shape:
            continue
        axes = (1, 2, 3, 0) if isinstance(module, nn.ConvTranspose2d) else (0, 2, 3, 1)
        weights[name] = kernel.transpose(*axes)
    return weights


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.residual = [
            nn.ReLU(),
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 3, padding=1),
        ]

    def __call__(self, x):
        residual = x
        for layer in self.residual:
            residual = layer(residual)
        return x + residual


class FusionBlock(nn.Module):
    def __init__(self, dim, with_skip):
        super().__init__()
        self.resnet1 = ResidualBlock(dim) if with_skip else None
        self.resnet2 = ResidualBlock(dim)
        self.deconv = nn.ConvTranspose2d(dim, dim, 2, stride=2, bias=False)
        self.out_conv = nn.Conv2d(dim, dim, 1)

    def __call__(self, x, skip=None):
        if skip is not None:
            x = x + self.resnet1(skip)
        return self.out_conv(self.deconv(self.resnet2(x)))


class DPTDecoder(nn.Module):
    def __init__(self, widths, dim):
        super().__init__()
        self.convs = [nn.Conv2d(w, dim, 3, padding=1, bias=False) for w in widths]
        self.fusions = [FusionBlock(dim, with_skip=i < 3) for i in range(4)]

    def __call__(self, features):
        x = self.fusions[-1](self.convs[-1](features[-1]))
        for i in (2, 1, 0):
            x = self.fusions[i](x, self.convs[i](features[i]))
        return x


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = MultiViewEncoder(config)
        dim = config.dim_decoder
        self.decoder = DPTDecoder(config.dims_reassemble, dim)
        self.head = [
            nn.Conv2d(dim, dim // 2, 3, padding=1),
            nn.ConvTranspose2d(dim // 2, dim // 2, 2, stride=2),
            nn.Conv2d(dim // 2, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),
            nn.ReLU(),
        ]
        self.fov = None
        if config.fov:
            from .focal import FocalLengthEstimator

            self.fov = FocalLengthEstimator()

    def sanitize(self, weights):
        return sanitize_convolutions(self, weights)

    def __call__(self, pixel_values, *, cache=None):
        """Return focal-normalized inverse depth for a normalized (T, H, W, 3) clip."""
        if pixel_values.ndim != 4 or pixel_values.shape[-1] != 3:
            raise ValueError("Depth Pro 2 expects a (T, H, W, 3) RGB clip")
        frames, height, width, _ = pixel_values.shape
        if min(frames, height, width) < 1:
            raise ValueError("Depth Pro 2 needs at least one non-empty frame")
        if cache is not None and frames != 1:
            raise ValueError("Cached Depth Pro 2 inference takes one frame at a time")
        pad_h, pad_w = -height % INPUT_MULTIPLE, -width % INPUT_MULTIPLE
        if pad_h or pad_w:
            padding = ((0, 0), (0, pad_h), (0, pad_w), (0, 0))
            pixel_values = mx.pad(pixel_values, padding, mode="edge")
        features, grid = self.encoder.encode(pixel_values, cache=cache)
        depths = []
        # At native video resolution the pyramid can dominate memory. Decode one frame at a time.
        for i in range(frames):
            pyramid = self.encoder.decode_features(
                [f[i : i + 1] for f in features], grid
            )
            x = self.decoder(pyramid).astype(self.head[0].weight.dtype)
            for layer in self.head:
                x = layer(x)
            depth = x[:, :height, :width, 0].astype(mx.float32)
            if frames > 1:
                mx.eval(depth)
            depths.append(depth)
        return mx.concatenate(depths, axis=0) if frames > 1 else depths[0]
