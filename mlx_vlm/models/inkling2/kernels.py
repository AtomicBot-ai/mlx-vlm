"""Inference kernels that retain Inkling 2's FP32 intermediate arithmetic."""

import math
from functools import lru_cache

import mlx.core as mx


@lru_cache(maxsize=None)
def _short_convolution(write_window, add_residual):
    source = r"""
        uint i = thread_position_in_grid.x;
        int B = x_shape[0], L = x_shape[1], C = x_shape[2];
        if (i >= B * L * C) return;
        int c = i % C, t = (i / C) % L, b = i / (C * L);
        float value = 0;
        for (int j = 0; j < W; ++j) {
            int p = t + j - W + 1;
            float xv = p < 0
                ? float(state[b * state_strides[0] + (p + W - 1) * state_strides[1]
                              + c * state_strides[2]])
                : ((HAS_MASK && !mask[b * mask_strides[0] + p * mask_strides[1]])
                    ? 0.0f
                    : float(x[b * x_strides[0] + p * x_strides[1] + c * x_strides[2]]));
            float w = weight[c * weight_strides[0] + j * weight_strides[1]];
            value = value + xv * w;
        }
        float input = float(x[b * x_strides[0] + t * x_strides[1] + c * x_strides[2]]);
        // Match the BF16 convolution output before the decoder's FP32 residual add.
        float result = float(T((value + input) / divisor));
    """
    inputs = ["state", "weight", "x", "mask", "divisor"]
    outputs = ["out"]
    if add_residual:
        inputs.append("residual")
        source += r"""
        result += float(residual[b * residual_strides[0] + t * residual_strides[1]
                                 + c * residual_strides[2]]);
        """
    source += "out[i] = OT(result);"
    if write_window:
        outputs.append("window")
        source += r"""
        int base = b * (W - 1 + L) * C + c;
        window[base + (W - 1 + t) * C] = input;
        if (t == 0) {
            for (int j = 0; j < W - 1; ++j) {
                window[base + j * C] = state[b * state_strides[0] + j * state_strides[1]
                                              + c * state_strides[2]];
            }
        }
        """
    return mx.fast.metal_kernel(
        name=f"inkling2_short_convolution_{int(write_window)}_{int(add_residual)}",
        input_names=inputs,
        output_names=outputs,
        ensure_row_contiguous=False,
        compile_options={"math_mode": "safe"},
        source=source,
    )


@lru_cache(maxsize=None)
def _divisor(width):
    return mx.array(math.sqrt(width + 1), mx.float32)


def short_convolution(
    state, weight, x, mask, width, residual=None, output_dtype=None, write_window=False
):
    """Optionally emit the unmasked FP32 history used for decode and rollback."""
    inputs = [state, weight, x, x if mask is None else mask, _divisor(width)]
    output_dtype = x.dtype if output_dtype is None else output_dtype
    shapes, dtypes = [x.shape], [output_dtype]
    if residual is not None:
        inputs.append(residual)
    if write_window:
        shapes.append((x.shape[0], width - 1 + x.shape[1], x.shape[2]))
        dtypes.append(mx.float32)
    outputs = _short_convolution(write_window, residual is not None)(
        inputs=inputs,
        template=[
            ("T", x.dtype),
            ("OT", output_dtype),
            ("W", width),
            ("HAS_MASK", mask is not None),
        ],
        grid=(x.size, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=shapes,
        output_dtypes=dtypes,
    )
    return outputs if write_window else outputs[0]
