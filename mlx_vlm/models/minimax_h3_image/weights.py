from __future__ import annotations

import json
from pathlib import Path

from mlx import nn

from ..minimax_h3.conditioner import MiniMaxH3Conditioner
from ..minimax_h3.weights import (
    _apply_exact_weights,
    _load_tokenizer,
    _prepare_conditioner_weights,
    _trim_conditioner_config,
    load_safetensors_strict,
)
from ..qwen3_vl.config import ModelConfig as QwenConfig
from ..qwen3_vl.qwen3_vl import Model as QwenModel
from .config import MiniMaxH3ImageTransformerConfig, MiniMaxH3ImageVAEConfig
from .download import text_encoder_path
from .transformer import MiniMaxH3ImageTransformer
from .vae import MiniMaxH3ImageVAE

FORMAT = "minimax_h3_image"


def read_config(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def component_path(root, name):
    root = Path(root)
    if name == "text_encoder":
        return text_encoder_path(root)
    if (
        name in ("transformer", "vae")
        and (root / f"{name}_diffusers" / "config.json").is_file()
    ):
        return root / f"{name}_diffusers"
    return root / name


def _load_weights(model, weights, config, component):
    quantization = config.get("quantization")
    if quantization:
        nn.quantize(
            model,
            **quantization,
            class_predicate=lambda path, module: path + ".scales" in weights,
        )
    _apply_exact_weights(component, model, weights)
    model.eval()
    return model


def load_transformer(root):
    path = component_path(root, "transformer")
    config = read_config(path / "config.json")
    if config.get("_class_name") not in (
        "MiniMaxH3ImageTransformer2DModel",
        "MiniMaxH3ImageTransformer",
    ):
        raise ValueError(
            "H3-Image requires transformer_diffusers/ from the official checkpoint or converted MLX weights"
        )
    model = MiniMaxH3ImageTransformer(MiniMaxH3ImageTransformerConfig.from_dict(config))
    return _load_weights(
        model, load_safetensors_strict(path), config, "H3 image transformer"
    )


def load_vae(root):
    path = component_path(root, "vae")
    config = read_config(path / "config.json")
    if config.get("_class_name") != "AutoencoderKLMiniMaxH3Image":
        raise ValueError(
            "H3-Image requires vae_diffusers/ from the official checkpoint or converted MLX weights"
        )
    model = MiniMaxH3ImageVAE(MiniMaxH3ImageVAEConfig.from_dict(config))
    weights = load_safetensors_strict(path)
    if config.get("mlx_vlm_format") != FORMAT:
        weights = model.sanitize(weights)
    return _load_weights(model, weights, config, "H3 image VAE")


def load_conditioner(root):
    root = Path(root)
    path = component_path(root, "text_encoder")
    config = read_config(path / "config.json")
    converted = config.get("mlx_vlm_format") == FORMAT
    if not converted:
        config = _trim_conditioner_config(config, text_only=False)
    model = QwenModel(QwenConfig.from_dict(config))
    weights = load_safetensors_strict(path)
    if not converted:
        weights = _prepare_conditioner_weights(weights, text_only=False)
    model = _load_weights(model, weights, config, "H3 image conditioner")
    return MiniMaxH3Conditioner(model, _load_tokenizer(root), has_vision=True)
