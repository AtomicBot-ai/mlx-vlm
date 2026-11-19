# Copyright © 2026 Neywa Labs Inc.

"""Auxiliary ECHO parameters retained by ordinary loading and conversion."""

import mlx.core as mx
import mlx.nn as nn

from ..deepseek_v4.hyper_connection import HyperHead

ECHO_WEIGHT_NAMES = {
    "enorm.weight": "enorm.weight",
    "hnorm.weight": "hnorm.weight",
    "e_proj.weight": "e_proj.weight",
    "h_proj.weight": "h_proj.weight",
    "hc_head_fn": "hc_head.fn",
    "hc_head_base": "hc_head.base",
    "hc_head_scale": "hc_head.scale",
    "shared_head.norm.weight": "norm.weight",
    "block_sparse_moe.e_score_correction_bias": "e_score_correction_bias",
}


def extract_echo_weights(weights, config):
    prefix = f"model.layers.{config.num_hidden_layers}."
    out = {}
    for key, value in weights.items():
        key = key.removeprefix("language_model.")
        if key.startswith("echo."):
            out[key.removeprefix("echo.")] = value
        elif key.startswith(prefix):
            name = key.removeprefix(prefix)
            if name in (
                "e_proj.scales",
                "e_proj.biases",
                "h_proj.scales",
                "h_proj.biases",
            ):
                out[name] = value
            elif name in ECHO_WEIGHT_NAMES:
                out[ECHO_WEIGHT_NAMES[name]] = (
                    value.T if name == "hc_head_fn" else value
                )
    return out


class EchoWeights(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.enorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.e_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.h_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.hc_head = HyperHead(config)
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # Decoder weights are shared; the ECHO router's selection bias is not.
        self.e_score_correction_bias = mx.zeros((config.n_routed_experts,), mx.float32)
