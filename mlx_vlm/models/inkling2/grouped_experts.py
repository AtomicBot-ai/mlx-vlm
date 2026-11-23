# Copyright © 2026 Neywa Labs Inc, Abdur Rahim

"""Grouped native-quantized expert projections for short Inkling 2 Mini verifies."""

from functools import lru_cache

import mlx.core as mx

from ...quantization.nvfp4 import ScaledQuantizedSwitchLinear
from ..masked_quantized import (
    _HEADER,
    _global_scale,
    supports_masked_qmv,
    supports_paired_mxfp8_qmv,
)
from ..switch_layers import QuantizedSwitchLinear, SwitchGLU

# Each SIMD group computes four output channels. Routes through the same expert
# reuse its decoded weights, with MLX's lane assignment and FP32 sum order.
_TILE = r"""
template <int V, int bits, int group_size, int P>
inline void expert_tile(
    const device bfloat16_t* x, const device uint* weights, const device uchar* scales,
    device bfloat16_t* out, const thread int* src, const thread int* dst,
    int expert, int row, uint lane, int K, int N, float factor) {
    const size_t channel = size_t(expert) * N + row;
    const int weight_stride = K * bits / 8, scale_stride = K / group_size;
    const device uchar* packed = reinterpret_cast<const device uchar*>(weights)
        + channel * weight_stride;
    const device uchar* scale = scales + channel * scale_stride;
    float sum[V][4] = {};
    for (int block = 0; block < K; block += 32 * P) {
        int col = block + int(lane) * P;
        if (col >= K) break;
        float values[V][P];
        #pragma unroll
        for (int v = 0; v < V; ++v) {
            load_vector<bfloat16_t, float, P>(x + src[v] * K + col, values[v]);
        }
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            const device uchar* w = packed + r * weight_stride + col * bits / 8;
            float s = dequantize_scale<float, group_size>(scale[r * scale_stride + col / group_size]);
            float decoded[P];
            #pragma unroll
            for (int i = 0; i < P; ++i) {
                if constexpr (bits == 4) decoded[i] = Dequantize<4>{}(w[i / 2] >> (4 * (i % 2)));
                else decoded[i] = Dequantize<8>{}(w[i]);
            }
            #pragma unroll
            for (int v = 0; v < V; ++v) {
                float dot = 0;
                if constexpr (bits == 4) {
                    #pragma unroll
                    for (int i = 0; i < P; i += 4) {
                        dot += (values[v][i] * decoded[i] + values[v][i+1] * decoded[i+1]
                              + values[v][i+2] * decoded[i+2] + values[v][i+3] * decoded[i+3]);
                    }
                } else {
                    #pragma unroll
                    for (int i = 0; i < P; ++i) dot += values[v][i] * decoded[i];
                }
                const float product = dot * s;
                sum[v][r] += product;
            }
        }
    }
    #pragma unroll
    for (int v = 0; v < V; ++v) {
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            float result = simd_sum(sum[v][r]);
            if (lane == 0) out[dst[v] * N + row + r] = bfloat16_t(result * factor);
        }
    }
}
"""


@lru_cache(maxsize=1)
def _grouped_qmv_kernel():
    # Specialize each run length so a singleton only loads one activation.
    tile = "expert_tile<{v}, BITS, GS, P>(x, w, s, out, src, dst, expert, row, lane, K, N, factor);"
    return mx.fast.metal_kernel(
        name="inkling2_grouped_expert_qmv",
        input_names=["x", "w", "s", "g", "experts", "source", "destination"],
        output_names=["out"],
        header=_HEADER + _TILE,
        compile_options={"math_mode": "safe"},
        source=r"""
        int first = threadgroup_position_in_grid.z;
        int expert = experts[first];
        if (first > 0 && experts[first - 1] == expert) return;
        int end = first + 1;
        while (end < experts_shape[0] && experts[end] == expert) ++end;
        int K = x_shape[1], N = w_shape[1];
        int row = threadgroup_position_in_grid.y * 8 + simdgroup_index_in_threadgroup * 4;
        uint lane = thread_index_in_simdgroup;
        float factor = 1.0f;
        if constexpr (HAS_SCALE) factor = g[expert] / (F8E4M3_MAX * F4E2M1_MAX);
        for (int begin = first; begin < end; begin += 4) {
            int count = min(4, end - begin), src[4], dst[4];
            for (int i = 0; i < count; ++i) {
                src[i] = source[begin + i];
                dst[i] = destination[begin + i];
            }
            switch (count) {
        """
        + "\n".join(f"case {v}: {tile.format(v=v)} break;" for v in range(1, 5))
        + "}}",
    )


def _supports_projection(linear, experts, inputs, outputs):
    if type(linear) not in (QuantizedSwitchLinear, ScaledQuantizedSwitchLinear):
        return False
    if (
        inputs % 32
        or linear.weight.shape != (experts, outputs, inputs * linear.bits // 32)
        or linear.scales.shape != (experts, outputs, inputs // linear.group_size)
    ):
        return False
    return supports_masked_qmv(linear, input_alignment=32) or supports_paired_mxfp8_qmv(
        linear, input_alignment=32
    )


def _project(x, linear, experts, source, destination):
    bits = linear.bits
    # MLX doubles the values per lane when two packs tile the input exactly.
    pack = 32 // bits
    values_per_lane = 2 * pack if x.shape[1] % (64 * pack) == 0 else pack
    scaled = isinstance(linear, ScaledQuantizedSwitchLinear)
    factor = _global_scale(linear) if scaled else linear.scales
    return _grouped_qmv_kernel()(
        inputs=[x, linear.weight, linear.scales, factor, experts, source, destination],
        template=[
            ("BITS", bits),
            ("GS", linear.group_size),
            ("P", values_per_lane),
            ("HAS_SCALE", scaled),
        ],
        grid=(32, linear.output_dims // 8 * 2, experts.size),
        threadgroup=(32, 2, 1),
        output_shapes=[(experts.size, linear.output_dims)],
        output_dtypes=[mx.bfloat16],
    )[0]


def grouped_expert_projections(mlp, x, indices):
    """Project sorted expert routes, then write down-projection rows in router order."""
    if (
        not mx.metal.is_available()
        or mx.default_device() != mx.gpu
        or x.dtype != mx.bfloat16
        or x.shape[0] != 1
        or not 2 <= x.shape[1] <= 8
        or indices.shape[-1] >= 64
        or type(mlp) is not SwitchGLU
        or mlp.training
    ):
        return None
    hidden = x.shape[-1]
    intermediate = getattr(mlp.up_proj, "output_dims", None)
    if intermediate is None or not (
        _supports_projection(mlp.up_proj, mlp.num_experts, hidden, intermediate)
        and _supports_projection(mlp.gate_proj, mlp.num_experts, hidden, intermediate)
        and _supports_projection(mlp.down_proj, mlp.num_experts, intermediate, hidden)
    ):
        return None
    routes = indices.reshape(-1)
    order = mx.argsort(routes)
    experts = routes[order]
    source = order // indices.shape[-1]
    identity = mx.arange(routes.size, dtype=order.dtype)
    x = x.reshape(x.shape[1:])
    up = _project(x, mlp.up_proj, experts, source, identity)
    gate = _project(x, mlp.gate_proj, experts, source, identity)
    down = _project(mlp.activation(up, gate), mlp.down_proj, experts, identity, order)
    return down.reshape(*indices.shape, x.shape[-1])
