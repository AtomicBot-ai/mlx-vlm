from __future__ import annotations

import math

import mlx.core as mx
import numpy as np

SAMPLE_SIGMAS = (
    1.0,
    0.9675236940383911,
    0.8875485062599182,
    0.75,
    0.5327739119529724,
    0.23200887441635132,
    0.0,
)


def sampling_schedule():
    # Preserve the released float32 round-trip through the time convention t=1-sigma.
    timesteps = mx.array([1 - sigma for sigma in SAMPLE_SIGMAS[:-1]], mx.float32)
    return timesteps, mx.concatenate([1 - timesteps, mx.zeros((1,))])


def canvas_size(width, height, min_generation_area=4194304):
    if width <= 0 or height <= 0:
        raise ValueError("Image width and height must be positive")
    if not 1024 <= min_generation_area <= 16777216:
        raise ValueError("min_generation_area must be between 1024 and 16777216")
    scale = max(1, math.sqrt(min_generation_area / (width * height)))
    return tuple(max(32, round(size * scale / 32) * 32) for size in (width, height))


def bound_short_edge(width, height, minimum, maximum):
    short = min(width, height)
    if short < 32:
        raise ValueError("Reference images must have at least 32 pixels on each side")
    if minimum <= short <= maximum:
        return width, height
    scale = (minimum if short < minimum else maximum) / short
    return tuple(max(32, round(size * scale / 32) * 32) for size in (width, height))


def build_image_layout(text_tags, reference_grids, height, width, patch_size=(1, 2, 2)):
    _, ph, pw = patch_size
    grids = [*reference_grids, (height, width)]
    if any(h % ph or w % pw for h, w in grids):
        raise ValueError("Image latent grids must be divisible by the patch size")
    text_count = text_tags.shape[0]
    counts = [(h // ph) * (w // pw) for h, w in grids]
    length = text_count + sum(counts)
    positions = np.zeros((length, 3), np.float64)
    positions[:text_count, 0] = np.arange(text_count)
    cursor = text_count
    for index, ((h, w), count) in enumerate(zip(grids, counts)):
        axes = []
        for size, patch in ((h, ph), (w, pw)):
            ratio = size / math.sqrt(h * w)
            left = (1 - ratio) / 2
            axes.append(
                np.linspace(left, left + ratio, size // patch, endpoint=False) * 32
            )
        positions[cursor : cursor + count, 0] = text_count + index
        positions[cursor : cursor + count, 1:] = np.stack(
            np.meshgrid(*axes, indexing="ij"), axis=-1
        ).reshape(-1, 2)
        cursor += count
    return (
        mx.array(positions.astype(np.float32)),
        mx.concatenate(
            [text_tags.astype(mx.int32), mx.zeros((sum(counts),), mx.int32)]
        ),
        mx.arange(text_count, length),
        mx.arange(text_count),
        sum(counts[:-1]),
    )
