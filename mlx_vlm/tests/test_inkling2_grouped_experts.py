import mlx.core as mx
import pytest

from mlx_vlm.models.inkling2 import grouped_experts
from mlx_vlm.models.inkling2.config import TextConfig
from mlx_vlm.models.inkling2.language import SparseMoE
from mlx_vlm.models.switch_layers import SwitchGLU
from mlx_vlm.quantization.nvfp4 import ScaledQuantizedSwitchLinear


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
    def forbidden(*args):
        pytest.fail("Unexpected projection dispatch")

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
    def forbidden(*args):
        pytest.fail("Unexpected grouped expert dispatch")

    monkeypatch.setattr(grouped_experts, "grouped_expert_projections", forbidden)
    model = SparseMoE(
        TextConfig(
            hidden_size=64,
            intermediate_size=32,
            n_routed_experts=4,
            num_experts_per_tok=2,
        )
    )
    model.train(train)
    x = mx.zeros((1, length, 64))
    mx.eval(model(x, speculative_verify=verify))
