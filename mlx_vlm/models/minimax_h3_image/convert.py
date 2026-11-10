"""Convert H3-Image components into a standalone, optionally quantized MLX repo."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import mlx.core as mx
from mlx import nn
from mlx.utils import tree_flatten

from ..minimax_h3.weights import _save_safetensors, _trim_conditioner_config
from .download import SOURCE_REPO_ID, download_model
from .weights import (
    FORMAT,
    component_path,
    load_conditioner,
    load_transformer,
    load_vae,
    read_config,
)


def model_card(source_id=SOURCE_REPO_ID):
    return f"""---
library_name: mlx-vlm
pipeline_tag: text-to-image
license: other
license_name: minimax-h3-image-community-license-agreement
license_link: https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/blob/main/LICENSE
base_model: {source_id}
tags:
- mlx
- minimax-h3-image
- image-editing
---

# MiniMax-H3-Image for MLX

Converted from [{source_id}](https://huggingface.co/{source_id}).
Supports text-to-image generation and native editing with up to sixteen reference
images. Runtime inference uses MLX, without PyTorch or Diffusers.

```sh
mlx_vlm.generate --output-modality image --model /path/to/this/checkpoint \\
  --prompt "A red ceramic teapot on a wooden table, morning window light" \\
  --size 4096x4096 --seed 42 --output teapot.png
```

For editing, add `--task edit --image reference.png` and refer to the images in
order as `<Picture 1>`, `<Picture 2>`, and so on. Short prompts are automatically
expanded with the separately downloaded MiniMax-H3-Image-PE checkpoint. Pass
`--gen-kwargs '{{"upsample_prompt":false}}'` to use a finished caption directly.

Sampling is fixed at six steps with guidance 1. Small requests are generated on
at least a 2048x2048-equivalent canvas before resizing. The text encoder is
released before diffusion, and fixed-schedule AdaLN projections are cached.

Weights use the [MiniMax H3 Image Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/blob/main/LICENSE).
The reference Diffusers implementation is [Apache-2.0](https://github.com/huggingface/diffusers/blob/main/LICENSE).
"""


def convert_minimax_h3_image(
    source,
    destination,
    *,
    source_id=SOURCE_REPO_ID,
    dtype=None,
    quantize=False,
    q_bits=4,
    q_group_size=64,
):
    source, destination = Path(source).expanduser(), Path(destination).expanduser()
    if source.resolve() == destination.resolve():
        raise ValueError("Conversion output must differ from the source")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Conversion output is not empty: {destination}")
    if dtype not in (None, "bfloat16", "float16", "float32"):
        raise ValueError(f"Unsupported dtype: {dtype}")
    if quantize and (
        q_bits not in (2, 3, 4, 6, 8) or q_group_size not in (32, 64, 128)
    ):
        raise ValueError("Invalid affine quantization bits or group size")
    index = read_config(source / "model_index.json")
    if index.get("_class_name") != "MiniMaxH3ImagePipeline":
        raise ValueError("Not a MiniMax-H3-Image checkpoint")
    destination.mkdir(parents=True, exist_ok=True)
    for name, loader in (
        ("text_encoder", load_conditioner),
        ("transformer", load_transformer),
        ("vae", load_vae),
    ):
        print(f"[INFO] Converting H3-Image {name}", flush=True)
        config = read_config(component_path(source, name) / "config.json")
        if quantize and config.get("quantization"):
            raise ValueError(f"{name} is already quantized")
        loaded = loader(source)
        model = loaded.model if name == "text_encoder" else loaded
        if name == "text_encoder":
            config = _trim_conditioner_config(config, text_only=False)
            # H3 consumes raw prompts, without Qwen's chat formatting.
            loaded.tokenizer.chat_template = None
            loaded.tokenizer.save_pretrained(destination / "tokenizer")
        if dtype is not None and name != "vae":
            model.load_weights(
                [
                    (
                        key,
                        (
                            value
                            if not mx.issubdtype(value.dtype, mx.floating)
                            or (
                                name == "transformer"
                                and key.startswith(
                                    ("proj_in.", "time_embedder.", "pdd_proj_out.")
                                )
                            )
                            else value.astype(getattr(mx, dtype))
                        ),
                    )
                    for key, value in tree_flatten(model.parameters())
                ]
            )
        if quantize and name != "vae":

            def predicate(path, module, component=name):
                if (
                    not isinstance(module, (nn.Linear, nn.Embedding))
                    or module.weight.shape[-1] % q_group_size
                ):
                    return False
                # Keep the modulation and boundary projections at their released precision.
                return component != "transformer" or (
                    not path.startswith(
                        ("proj_in", "time_embedder", "pdd_proj_out", "norm_out")
                    )
                    and ".adaln_proj." not in path
                )

            quantization = {
                "bits": q_bits,
                "group_size": q_group_size,
                "mode": "affine",
            }
            nn.quantize(model, **quantization, class_predicate=predicate)
            config["quantization"] = quantization
        config["mlx_vlm_format"] = FORMAT
        _save_safetensors(destination / name, dict(tree_flatten(model.parameters())))
        (destination / name / "config.json").write_text(
            json.dumps(config, indent=2) + "\n"
        )
        del model, loaded
        gc.collect()
        mx.clear_cache()
    # The converted package is self-contained; remove the upstream external text encoder pointer.
    index = {
        "_class_name": "MiniMaxH3ImagePipeline",
        "model_type": FORMAT,
        "mlx_vlm_format": FORMAT,
        "source_model": source_id,
    }
    (destination / "model_index.json").write_text(json.dumps(index, indent=2) + "\n")
    (destination / "README.md").write_text(model_card(source_id))
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-path", default=SOURCE_REPO_ID)
    parser.add_argument("--mlx-path", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--source-id", default=SOURCE_REPO_ID)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"))
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--q-bits", type=int, default=4)
    parser.add_argument("--q-group-size", type=int, default=64)
    args = parser.parse_args()
    source = download_model(args.hf_path, revision=args.revision)
    convert_minimax_h3_image(
        source,
        args.mlx_path,
        source_id=args.source_id,
        dtype=args.dtype,
        quantize=args.quantize,
        q_bits=args.q_bits,
        q_group_size=args.q_group_size,
    )


if __name__ == "__main__":
    main()
