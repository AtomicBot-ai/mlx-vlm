"""Inkling Metal kernels are only used when Metal exists.

A GPU default device is not enough: a GPU backend without Metal must take the
pure-MLX paths instead of calling ``mx.fast.metal_kernel``.
"""

import mlx.core as mx
import mlx.nn as nn
import pytest

import mlx_vlm.models.inkling.language as lang
from mlx_vlm.models.inkling.config import TextConfig


def _forbid(name):
    def kernel(*args, **kwargs):
        raise AssertionError(f"{name} called without Metal")

    return kernel


@pytest.fixture
def gpu_without_metal(monkeypatch):
    monkeypatch.setattr(mx, "default_device", lambda: mx.gpu)
    monkeypatch.setattr(mx.metal, "is_available", lambda: False)
    for name in (
        "_mask_kernel",
        "_mask_v2_kernel",
        "_sconv_kernel",
        "_route_kernel",
        "_down_combine_kernel",
    ):
        monkeypatch.setattr(lang, name, _forbid(name))


@pytest.mark.parametrize("with_shape_ref", [False, True])
def test_banded_mask_skips_kernels(gpu_without_metal, with_shape_ref):
    B, LQ, H, D_REL, S = 1, 2, 2, 4, 6
    rel = mx.random.normal((B, LQ, H, D_REL))
    proj = mx.random.normal((D_REL, 8))
    shape_ref = mx.zeros((B, 1, S)) if with_shape_ref else None
    mask = lang.banded_additive_mask(rel, proj, S - LQ, S, 0, 8, shape_ref=shape_ref)
    assert mask.shape == (B, H, LQ, S)


def test_short_conv_decode_skips_kernel(gpu_without_metal):
    channels = 8
    conv = lang.InklingShortConvolution(channels, 4, 0)
    cache = [None]
    out = conv(mx.random.normal((1, 1, channels)), cache=cache)
    assert out.shape == (1, 1, channels)


@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize(
    "metal_available", [False, True], ids=["gpu-no-metal", "cpu-on-metal"]
)
def test_inkling2_sconv_skips_metal_kernels(monkeypatch, metal_available, masked):
    from mlx_vlm.models.cache import ArraysCache
    from mlx_vlm.models.inkling2 import kernels, language

    monkeypatch.setattr(kernels, "_short_convolution", _forbid("short convolution"))
    monkeypatch.setattr(
        language, "_conv_history_kernel", _forbid("convolution history")
    )
    monkeypatch.setattr(mx.metal, "is_available", lambda: metal_available)
    with mx.stream(mx.cpu):
        if not metal_available:
            # Emulate a non-Metal GPU backend while evaluating the fallback on CPU.
            monkeypatch.setattr(mx, "default_device", lambda: mx.gpu)
        channels = 128
        conv = language.ShortConvolution(channels, 4, 0)
        conv.conv.weight = mx.zeros((channels, 4, 1), mx.bfloat16)
        conv.eval()
        x = mx.broadcast_to(mx.arange(1, 5)[None, :, None], (1, 4, channels)).astype(
            mx.bfloat16
        )
        mask = mx.array([[False, True, True, True]]) if masked else None
        cache = ArraysCache(1)
        state = mx.zeros((1, 3, channels))
        cache[0] = state
        assert kernels.short_convolution(state, conv.conv.weight, x, mask, 4) is None
        cache.start_speculation(4)
        residual = mx.ones(x.shape)
        out = conv(x, cache, mask, residual=residual, output_dtype=mx.float32)
        cache.commit_speculation(2)
        expected = (x.astype(mx.float32) / 5**0.5).astype(mx.bfloat16).astype(
            mx.float32
        ) + residual
        assert mx.array_equal(out, expected).item()
        kept = mx.array([0, 0 if masked else 1, 2], mx.float32)
        assert mx.array_equal(
            cache[0], mx.broadcast_to(kept[None, :, None], state.shape)
        ).item()


def test_moe_route_and_down_combine_skip_kernels(gpu_without_metal):
    config = TextConfig(
        hidden_size=64,
        intermediate_size=2048,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
    )
    moe = lang.InklingSparseMoE(config)
    nn.quantize(moe.switch_mlp, group_size=64, bits=4)
    out = moe(mx.random.normal((1, 1, 64)))
    mx.eval(out)
    assert out.shape == (1, 1, 64)
