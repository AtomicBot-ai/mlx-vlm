# Copyright © 2026 Neywa Labs Inc.

"""Affine and NVFP4 projection kernels with native decoding arithmetic."""

from functools import lru_cache

import mlx.core as mx
import mlx.nn as nn

_BLOCK_POSITION = r"""
const int lane = thread_index_in_simdgroup;
const int kl = lane % L;
const int row = (threadgroup_position_in_grid.y * SG * (32 / L) +
    simdgroup_index_in_threadgroup * (32 / L) + lane / L) * R;
const int r = min(row, N - 1);
const int first = threadgroup_position_in_grid.x * V;
float result[R * V] = {0};
"""

_BLOCK_STORE = r"""
for (int rv = 0; rv < R * V; ++rv) {
    const int rr = rv / V;
    const int v = rv % V;
    if constexpr (SINGLE && QUANTIZED) {
        result[rv] = simd_sum(result[rv]);
    } else {
        #pragma unroll
        for (int step = L / 2; step >= 1; step /= 2)
            result[rv] += simd_shuffle_down(result[rv], step);
    }
    if (kl == 0 && row + rr < N && first + v < M)
        out[(first + v) * N + row + rr] = O(result[rv]);
}
"""


@lru_cache(maxsize=1)
def _nvfp4_kernel():
    return mx.fast.metal_kernel(
        name="nvfp4_verify_qmv",
        input_names=["x", "weight", "scales"],
        output_names=["out"],
        source=_BLOCK_POSITION + r"""
        for (int g = kl; g < K / 16; g += L) {
            float scale[R];
            float4 decoded[R][4];
            #pragma unroll
            for (int rr = 0; rr < R; ++rr) {
                int wr = min(row + rr, N - 1);
                const device ushort* w = (const device ushort*)weight + wr * K / 4;
                uchar bits = scales[wr * (K / 16) + g];
                int exponent = (bits >> 3) & 15;
                float s = exponent == 0 ? float(bits & 7) / 512.0f :
                    ldexp(float(8 + (bits & 7)), exponent - 10);
                if (bits & 128) s = -s;
                if ((bits & 127) == 127) s = NAN;
                scale[rr] = s;
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    ushort packed = w[g * 4 + j];
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        int q = (packed >> (i * 4)) & 15;
                        // E2M1 becomes an exact half after an exponent shift.
                        ushort h = ((q & 8) << 12) | ((q & 7) << 9);
                        decoded[rr][j][i] = float(as_type<half>(h) * half(16384));
                    }
                }
            }
            #pragma unroll
            for (int v = 0; v < V; ++v) {
                const device float4* xv = (const device float4*)x +
                    min(first + v, M - 1) * (K / 4) + g * 4;
                float acc[R] = {0};
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    float4 a = xv[j];
                    #pragma unroll
                    for (int rr = 0; rr < R; ++rr) {
                        float4 d = decoded[rr][j];
                        if constexpr (SINGLE)
                            acc[rr] += a[0] * d[0] + a[1] * d[1] +
                                a[2] * d[2] + a[3] * d[3];
                        else
                            acc[rr] += dot(d, a);
                    }
                }
                #pragma unroll
                for (int rr = 0; rr < R; ++rr) {
                    if constexpr (SINGLE) {
                        // QMV rounds the scaled group dot before adding it;
                        // a single expression would contract this into FMA.
                        float scaled = scale[rr] * acc[rr];
                        result[rr * V + v] += scaled;
                    } else {
                        result[rr * V + v] += scale[rr] * acc[rr];
                    }
                }
            }
        }
        """ + _BLOCK_STORE,
    )


@lru_cache(maxsize=1)
def _affine_kernel():
    # The reduction follows MLX's qmv_wide_impl. As in
    # Mistral Large 4's verifier, vector tiles revisit the same output rows.
    return mx.fast.metal_kernel(
        name="affine_batch_qmv",
        input_names=["x", "weight", "scales", "biases"],
        output_names=["out"],
        source=r"""
        const int lane = thread_index_in_simdgroup;
        const int kl = lane % 8;
        const int row = threadgroup_position_in_grid.y * SG * 4 +
            simdgroup_index_in_threadgroup * 4 + lane / 8;
        const int r = min(row, N - 1);
        const device uchar* w = (const device uchar*)weight + r * K * BITS / 8;
        for (int chunk = threadgroup_position_in_grid.x;
             chunk < (M + V - 1) / V; chunk += threadgroups_per_grid.x) {
            int first = chunk * V;
            float result[V] = {0};
            for (int g = kl; g < K / GS; g += 8) {
                float scale = scales[r * (K / GS) + g];
                float bias = biases[r * (K / GS) + g];
                #pragma unroll
                for (int sc = 0; sc < GS / 8; ++sc) {
                    int k = g * GS + sc * 8;
                    float decoded[8];
                    #pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        int shift = (i * BITS) % 8;
                        int mask = ((1 << BITS) - 1) << shift;
                        decoded[i] = (scale / float(1 << shift)) *
                            (w[(k + i) * BITS / 8] & mask) + bias;
                    }
                    #pragma unroll
                    for (int v = 0; v < V; ++v) {
                        const device T* xv = x + min(first + v, M - 1) * K + k;
                        float acc = 0;
                        #pragma unroll
                        for (int i = 0; i < 8; ++i) acc += float(xv[i]) * decoded[i];
                        result[v] += acc;
                    }
                }
            }
            for (int v = 0; v < V; ++v) {
                result[v] += simd_shuffle_down(result[v], 4);
                result[v] += simd_shuffle_down(result[v], 2);
                result[v] += simd_shuffle_down(result[v], 1);
                if (kl == 0 && row < N && first + v < M)
                    out[(first + v) * N + row] = T(result[v]);
            }
        }
        """,
    )


def native_affine_batch_linear(layer, x):
    """Return a native small-batch QMV block, or None for other dispatches."""
    if (
        type(layer) is not nn.QuantizedLinear
        or layer.training
        or not mx.metal.is_available()
        or mx.default_device() != mx.gpu
        or x.ndim != 3
        or not 2 <= x.shape[0] <= 8
        or not 2 <= x.shape[1] <= 8
        or x.dtype not in (mx.bfloat16, mx.float16)
        or layer.mode != "affine"
        or layer.bits not in (2, 4, 8)
        or layer.biases is None
        or layer.scales.dtype != x.dtype
        or layer.biases.dtype != x.dtype
        or x.shape[-1] <= 128
    ):
        return None
    architecture = mx.device_info().get("architecture", "").split("_g")[-1][:2]
    if not architecture.isdigit() or int(architecture) < 15:
        return None
    b, t, k = x.shape
    n = layer.weight.shape[0]
    vectors = 6 if b * t <= 12 else 4
    groups = 2 if b * t <= 6 or n < 10240 else 8
    columns = min(4, (b * t + vectors - 1) // vectors)
    result = _affine_kernel()(
        inputs=[x, layer.weight, layer.scales, layer.biases],
        template=[
            ("T", x.dtype),
            ("K", k),
            ("N", n),
            ("M", b * t),
            ("BITS", layer.bits),
            ("GS", layer.group_size),
            ("V", vectors),
            ("SG", groups),
        ],
        grid=(32 * columns, groups * ((n + 4 * groups - 1) // (4 * groups)), 1),
        threadgroup=(32, groups, 1),
        output_shapes=[(b, t, n)],
        output_dtypes=[x.dtype],
    )[0]
    return result + layer.bias if "bias" in layer else result
