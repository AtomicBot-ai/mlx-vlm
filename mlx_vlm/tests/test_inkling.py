"""Kernel parity and fallback tests for Inkling and Inkling 2."""

import mlx.core as mx
import mlx.nn as nn
import pytest

import mlx_vlm.models.inkling.language as lang
from mlx_vlm.models.inkling2 import grouped_experts
from mlx_vlm.models.inkling2.config import TextConfig as Inkling2TextConfig
from mlx_vlm.models.inkling2.language import SparseMoE
from mlx_vlm.models.inkling.config import TextConfig as InklingTextConfig
from mlx_vlm.models.switch_layers import SwitchGLU
from mlx_vlm.quantization.nvfp4 import ScaledQuantizedSwitchLinear


def _forbid(name):
    def kernel(*args, **kwargs):
        raise AssertionError(f"Unexpected kernel dispatch for {name}")

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
    config = InklingTextConfig(
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


@pytest.mark.parametrize("rows,scale", [(1, 0.25), (4, 5.0), (8, 1.0)])
def test_inkling2_attention_head_kernel_matches_mlx_bits(rows, scale):
    from mlx_vlm.models.inkling2.kernels import prepare_attention_heads
    from mlx_vlm.models.inkling2.language import l2_clip

    if not mx.metal.is_available() or mx.default_device() != mx.gpu:
        pytest.skip("Requires Metal")

    # Strided projections and per-head scales exercise the kernel's input layout.
    def sample(shape, seed):
        wide = (*shape[:-1], 2 * shape[-1])
        return (scale * mx.random.normal(wide, key=mx.random.key(seed))).astype(
            mx.bfloat16
        )[..., ::2]

    q = sample((1, rows, 6, 128), 0)
    k, v = (sample((1, rows, 2, 128), seed) for seed in (1, 2))
    r = sample((1, rows, 6, 16), 3)
    if rows == 8:
        # A NaN in a single lane must propagate through the whole head.
        for projection in (q, k, v, r):
            projection[0, -1, 0, 0] = float("nan")
    qs, rs = sample((6, 128), 4), sample((6, 16), 5)
    got = prepare_attention_heads(q, k, v, r, qs, rs, 22.627416997969522, 8.0, 1e-6)
    expected = (
        l2_clip(q, 22.627416997969522, qs),
        l2_clip(k, 22.627416997969522),
        mx.fast.rms_norm(v.astype(mx.float32), None, 1e-6).astype(mx.bfloat16),
        l2_clip(r, 8.0, rs),
    )
    assert got is not None
    for actual, reference in zip(got, expected):
        assert actual.shape == reference.shape and actual.dtype == reference.dtype
        assert mx.array_equal(actual.view(mx.uint16), reference.view(mx.uint16)).item()


@pytest.mark.parametrize(
    "device,metal,batch,rows,dim,rel,dtype",
    [
        (mx.cpu, True, 1, 1, 128, 16, mx.bfloat16),
        (mx.gpu, False, 1, 1, 128, 16, mx.bfloat16),
        (mx.gpu, True, 2, 4, 128, 16, mx.bfloat16),
        (mx.gpu, True, 1, 9, 128, 16, mx.bfloat16),
        (mx.gpu, True, 1, 4, 64, 16, mx.bfloat16),
        (mx.gpu, True, 1, 4, 128, 8, mx.bfloat16),
        (mx.gpu, True, 1, 4, 128, 16, mx.float16),
    ],
)
def test_inkling2_attention_head_kernel_fallback(
    monkeypatch, device, metal, batch, rows, dim, rel, dtype
):
    from mlx_vlm.models.inkling2 import kernels

    monkeypatch.setattr(
        kernels, "_attention_head_kernel", _forbid("attention head preparation")
    )
    monkeypatch.setattr(mx.metal, "is_available", lambda: metal)
    monkeypatch.setattr(mx, "default_device", lambda: device)
    q = mx.zeros((batch, rows, 1, dim), dtype)
    r = mx.zeros((batch, rows, 1, rel), dtype)
    assert (
        kernels.prepare_attention_heads(q, q, q, r, q[0, 0], r[0, 0], 8.0, 4.0, 1e-6)
        is None
    )


def test_inkling2_mini_attention_uses_head_kernel_only_in_inference(monkeypatch):
    from mlx_vlm.models.inkling2 import kernels
    from mlx_vlm.models.inkling2.config import LoopConfig
    from mlx_vlm.models.inkling2.language import Attention

    cfg = Inkling2TextConfig(
        hidden_size=32,
        swa_num_attention_heads=2,
        swa_num_key_value_heads=1,
        local_layer_ids=[0],
        n_layers_per_attn_res_block=1,
        loop_config=LoopConfig(1, 2, 2, 4),
    )
    attn = Attention(cfg, 0)
    attn.set_dtype(mx.bfloat16)
    x = mx.zeros((1, 1, 32), mx.bfloat16)
    k = mx.zeros((1, 1, 128), mx.bfloat16)
    calls = []
    factory = kernels._attention_head_kernel

    def counted():
        calls.append(True)
        return factory()

    monkeypatch.setattr(kernels, "_attention_head_kernel", counted)
    attn.eval()
    expected = attn._prepare_heads(x, k, k)
    assert len(calls) == int(mx.metal.is_available() and mx.default_device() == mx.gpu)
    monkeypatch.setattr(
        kernels, "prepare_attention_heads", _forbid("inference-only head preparation")
    )
    attn.train()
    for actual, ref in zip(attn._prepare_heads(x, k, k), expected):
        assert mx.array_equal(actual, ref).item()


def _experts(mode):
    # The up/gate widths select MLX's fast dot, while down has a partial tile.
    mlp = SwitchGLU(512, 96, 8)
    for name in ("up_proj", "gate_proj", "down_proj"):
        projection = getattr(mlp, name).to_quantized(
            group_size=16 if mode == "nvfp4" else 32,
            bits=4 if mode == "nvfp4" else 8,
            mode=mode,
        )
        if mode == "nvfp4" and name == "up_proj":
            projection = ScaledQuantizedSwitchLinear(
                projection, mx.arange(1, 9) / 9, "weight_global_scale"
            )
        if mode == "nvfp4" and name == "down_proj":
            projection = ScaledQuantizedSwitchLinear(
                projection, mx.array(1.375), "weight_scale_2"
            )
        setattr(mlp, name, projection)
    mlp.eval()
    return mlp


@pytest.mark.parametrize("mode", ["nvfp4", "mxfp8"])
def test_grouped_experts_preserve_route_and_dot_order(mode):
    if not mx.metal.is_available() or mx.default_device() != mx.gpu:
        pytest.skip("Requires Metal")
    mlp = _experts(mode)
    # Expert 5 spans two tiles; the other groups include lengths 1, 2 and 3.
    indices = mx.array(
        [[[5, 1], [0, 5], [5, 3], [1, 5], [5, 2], [3, 5], [1, 5], [5, 6]]],
        mx.uint32,
    )
    x = mx.random.normal((1, 8, 1024), key=mx.random.key(21)).astype(mx.bfloat16)[
        ..., ::2
    ]
    actual = grouped_experts.grouped_expert_projections(mlp, x, indices)
    assert actual is not None
    expected = mx.concatenate(
        [mlp(x[:, i : i + 1], indices[:, i : i + 1]) for i in range(8)], axis=1
    )
    assert mx.array_equal(actual.view(mx.uint16), expected.view(mx.uint16)).item()


@pytest.mark.parametrize(
    "device,metal,batch,length,dtype",
    [
        (mx.cpu, True, 1, 8, mx.bfloat16),
        (mx.gpu, False, 1, 8, mx.bfloat16),
        (mx.gpu, True, 2, 8, mx.bfloat16),
        (mx.gpu, True, 1, 1, mx.bfloat16),
        (mx.gpu, True, 1, 9, mx.bfloat16),
        (mx.gpu, True, 1, 8, mx.float16),
    ],
)
def test_grouped_expert_device_and_shape_fallback(
    monkeypatch, device, metal, batch, length, dtype
):
    forbidden = _forbid("expert projection")

    monkeypatch.setattr(mx.metal, "is_available", lambda: metal)
    monkeypatch.setattr(mx, "default_device", lambda: device)
    monkeypatch.setattr(grouped_experts, "_project", forbidden)
    monkeypatch.setattr(grouped_experts, "_supports_projection", forbidden)
    mlp = SwitchGLU(128, 64, 8)
    mlp.eval()
    x = mx.zeros((batch, length, 128), dtype)
    indices = mx.zeros((batch, length, 2), mx.uint32)
    assert grouped_experts.grouped_expert_projections(mlp, x, indices) is None


def test_grouped_experts_preserve_unsupported_projections():
    if not mx.metal.is_available() or mx.default_device() != mx.gpu:
        pytest.skip("Requires Metal")
    mlp = _experts("nvfp4")
    mlp.gate_proj.bias = mx.ones((8, 96))
    x = mx.zeros((1, 2, 512), mx.bfloat16)
    indices = mx.zeros((1, 2, 2), mx.uint32)
    assert grouped_experts.grouped_expert_projections(mlp, x, indices) is None


@pytest.mark.parametrize(
    "verify,train,length", [(False, False, 4), (True, False, 1), (True, True, 4)]
)
def test_grouped_experts_only_run_for_multitoken_inference(
    monkeypatch, verify, train, length
):
    monkeypatch.setattr(
        grouped_experts,
        "grouped_expert_projections",
        _forbid("grouped expert projections"),
    )
    model = SparseMoE(
        Inkling2TextConfig(
            hidden_size=64,
            intermediate_size=32,
            n_routed_experts=4,
            num_experts_per_tok=2,
        )
    )
    model.train(train)
    x = mx.zeros((1, length, 64))
    mx.eval(model(x, speculative_verify=verify))
