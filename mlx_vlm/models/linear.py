# Copyright © 2026 Neywa Labs Inc.

"""Projection dispatch and dense kernels shared by model families."""

from functools import lru_cache

import mlx.core as mx
import mlx.nn as nn

from .quantized import _BLOCK_POSITION, _BLOCK_STORE, _nvfp4_kernel

DECODE_BLOCK_SIZE = 8


@lru_cache(maxsize=1)
def _dense_kernel():
    return mx.fast.metal_kernel(
        name="dense_verify_gemv",
        input_names=["x", "weight"],
        output_names=["out"],
        source=_BLOCK_POSITION + r"""
        const device vec<T, 4>* w = (const device vec<T, 4>*)weight + r * K / 4;
        const device vec<T, 4>* x4 = (const device vec<T, 4>*)x;
        constexpr int U = SINGLE ? 1 : 8;
        for (int base = 0; base < K / 4; base += L * U) {
            float4 decoded[U];
            #pragma unroll
            for (int i = 0; i < U; ++i)
                decoded[i] = float4(w[base + i * L + kl]);
            #pragma unroll
            for (int v = 0; v < V; ++v) {
                float acc = 0;
                #pragma unroll
                for (int i = 0; i < U; ++i) {
                    float4 a = float4(x4[min(first + v, M - 1) * (K / 4) +
                        base + i * L + kl]);
                    if constexpr (SINGLE) {
                        #pragma unroll
                        for (int j = 0; j < 4; ++j)
                            result[v] += decoded[i][j] * a[j];
                    } else {
                        acc += dot(decoded[i], a);
                    }
                }
                if constexpr (!SINGLE) result[v] += acc;
            }
        }
        """ + _BLOCK_STORE,
    )


def native_block_linear(layer, x):
    """Share dense/NVFP4 weights across positions without switching to GEMM."""
    from ..quantization.nvfp4 import ScaledQuantizedLinear

    if (
        type(layer) not in (nn.Linear, ScaledQuantizedLinear)
        or layer.training
        or x.ndim != 3
        or not 1 <= x.shape[0] <= 8
        or not 2 <= x.shape[1] <= 8
        or x.dtype not in (mx.bfloat16, mx.float16, mx.float32)
        or not mx.metal.is_available()
        or mx.default_device() != mx.gpu
    ):
        return None
    b, t, k = x.shape
    n = layer.weight.shape[0]
    quantized = isinstance(layer, ScaledQuantizedLinear)
    if quantized:
        if (
            layer.mode != "nvfp4"
            or layer.bits != 4
            or layer.group_size != 16
            or layer._quantize_activations
            or k % 512
            or k != layer.scales.shape[-1] * 16
        ):
            return None
        projected = x.astype(mx.float32)
        if "pre_quant_scale" in layer:
            projected *= layer.pre_quant_scale.astype(mx.float32)
        inputs = [projected, layer.weight, layer.scales]
        lanes = 32 if b == 1 else 16
        kernel = _nvfp4_kernel()
        dtype = mx.float32
    else:
        # Match the native GEMV/GEMV-wide dispatch, including its eight-step
        # reduction for multiple batch rows.
        if (
            "bias" in layer
            or x.dtype not in (mx.bfloat16, mx.float16)
            or layer.weight.dtype != x.dtype
            or k != layer.weight.shape[-1]
            or k % (128 if b == 1 else 1024)
        ):
            return None
        if k >= 16 * n:
            return None
        architecture = mx.device_info().get("architecture", "").split("_g")[-1][:2]
        if b > 1 and (not architecture.isdigit() or int(architecture) < 15):
            return None
        inputs = [x, layer.weight]
        lanes = 32 if b <= 5 or n <= 64 else 16
        kernel = _dense_kernel()
        dtype = x.dtype
    vectors = min(b * t, 5)
    groups = 4
    row_tile = 2 if quantized and (b == 1 or k > n) else 1
    rows = groups * (32 // lanes) * row_tile
    output = kernel(
        inputs=inputs,
        template=[
            ("T", x.dtype),
            ("O", dtype),
            ("K", k),
            ("N", n),
            ("M", b * t),
            ("V", vectors),
            ("SG", groups),
            ("L", lanes),
            ("R", row_tile),
            ("SINGLE", b == 1),
            ("QUANTIZED", quantized),
        ],
        grid=(
            32 * ((b * t + vectors - 1) // vectors),
            groups * ((n + rows - 1) // rows),
            1,
        ),
        threadgroup=(32, groups, 1),
        output_shapes=[(b, t, n)],
        output_dtypes=[dtype],
    )[0]
    if quantized:
        output = layer.apply_scale(output, layer.global_scale).astype(x.dtype)
    return output + layer.bias if "bias" in layer else output


def tokenwise(operation, x, *args, axis=1):
    """Apply an operation with the same reduction geometry at each position."""

    def at(value, index):
        slices = [slice(None)] * value.ndim
        slices[axis] = slice(index, index + 1)
        return mx.contiguous(value[tuple(slices)])

    outputs = [
        operation(at(x, i), *(at(arg, i) for arg in args)) for i in range(x.shape[axis])
    ]
    if isinstance(outputs[0], (tuple, list)):
        return tuple(mx.concatenate(parts, axis=axis) for parts in zip(*outputs))
    return mx.concatenate(outputs, axis=axis)


def native_batch_linear(module, x):
    """Match independent B×1 calls, including dense and mixed-dtype fallbacks."""
    if x.ndim != 3 or x.shape[1] <= 1:
        return module(x)
    from .quantized_verifier import exact_quantized_linear, singleton_quantized_linear

    output = native_block_linear(module, x)
    if output is not None:
        return output

    # Narrow and FP32 projections can use a different native QMV reduction.
    if (
        x.shape[0] == 1
        and x.shape[-1] % 512 == 0
        and x.dtype in (mx.bfloat16, mx.float16)
    ):
        output = singleton_quantized_linear(module, x)
        if output is not None:
            return output
    output = exact_quantized_linear(module, x)
    return tokenwise(module, x) if output is None else output


def linear(module, x):
    """Use decode-equivalent projections for short blocks, GEMM for prefill."""
    if (
        x.ndim == 3
        and 1 < x.shape[1] <= DECODE_BLOCK_SIZE
        and not getattr(module, "training", False)
    ):
        return native_batch_linear(module, x)
    return module(x)


def tiled_linear(module, x):
    """Give large prefill projections a fixed reduction geometry.

    Explicit calls prevent matmul from folding the tile axis into its row
    dimension. The final tile is padded so it uses the same reduction too.
    Small calls retain the model's native geometry, including draft replay.
    """
    tile_size = 256
    if x.ndim != 3 or x.shape[1] <= tile_size:
        return linear(module, x)
    length = x.shape[1]
    padding = (-length) % tile_size
    if padding:
        x = mx.pad(x, [(0, 0), (0, padding), (0, 0)])
    return mx.concatenate(
        [
            module(mx.contiguous(x[:, start : start + tile_size]))
            for start in range(0, length, tile_size)
        ],
        axis=1,
    )[:, :length]
