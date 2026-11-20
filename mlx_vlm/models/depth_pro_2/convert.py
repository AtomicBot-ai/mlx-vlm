"""Convert an Apple Depth Pro 2 checkpoint and its focal estimator to MLX."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import mlx.core as mx

from ...utils import get_model_path, save_config, save_weights
from .config import ModelConfig
from .depth_pro_2 import Model
from .focal import FOCAL_CHECKPOINT


def model_card(repo_id: str, base_model: str) -> str:
    return f"""---
license: other
license_name: apple-amlr
license_link: LICENSE
base_model:
  - {base_model}
  - {FOCAL_CHECKPOINT}
library_name: mlx-vlm
pipeline_tag: depth-estimation
tags:
  - mlx
  - metric-depth
  - monocular-depth
  - video-depth
---

# {repo_id.split("/")[-1]}

This MLX derivative of Apple's Depth Pro 2 estimates metric depth from photos,
video, and live RGB frames. It includes the shared focal-length estimator.
The conversion repacks the convolution kernels for MLX and preserves Apple's
float32 weights. It adds an MLX model configuration and bundles the focal
estimator with the depth model.

## Usage

```sh
pip install mlx-vlm
```

```python
from mlx_vlm import load
from mlx_vlm.models.depth_pro_2.generate import DepthPro2Predictor, read_image

model, processor = load("{repo_id}")
predictor = DepthPro2Predictor(model, processor)
result = predictor.infer(read_image("photo.jpg"))
depth_m = result["depth"]
focal_px = result["focallength_px"]
```

See the [model documentation](https://github.com/Neywa-Labs/mlx-vlm/blob/main/mlx_vlm/models/depth_pro_2/README.md)
for video and live streams.

## License

Apple Machine Learning Research Model is licensed under the Apple Machine
Learning Research Model License Agreement.

The weights and this MLX derivative are for non-commercial scientific research
and academic development. The full agreement is included in [LICENSE](LICENSE).
See the [original model family](https://huggingface.co/apple/DepthPro2) for
architecture, evaluation, and citation information.
"""


def convert(variant: str, output: str | Path, repo_id: str) -> None:
    """Write a complete checkpoint folder for one of Apple's released variants."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")

    base_model = f"apple/DepthPro2-{variant}"
    source = get_model_path(
        base_model, allow_patterns=["config.json", "model.safetensors", "LICENSE"]
    )
    focal = get_model_path(
        FOCAL_CHECKPOINT, allow_patterns=["model.safetensors", "LICENSE"]
    )
    agreement = (source / "LICENSE").read_bytes()
    if agreement != (focal / "LICENSE").read_bytes():
        raise ValueError("The depth and focal checkpoints have different licenses")
    config = ModelConfig.from_dict(
        {**json.loads((source / "config.json").read_text()), "fov": True}
    )

    weights = mx.load(str(source / "model.safetensors"))
    weights.update(
        (f"fov.{name}", value)
        for name, value in mx.load(str(focal / "model.safetensors")).items()
    )
    model = Model(config)
    model.load_weights(list(model.sanitize(weights).items()))
    del weights

    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix=f".{output.name}-") as tmp:
        bundle = Path(tmp) / output.name
        save_weights(bundle, model, donate_weights=True)
        save_config(config.to_dict(), bundle / "config.json")
        (bundle / "LICENSE").write_bytes(agreement)
        (bundle / "README.md").write_text(
            model_card(repo_id, base_model), encoding="utf-8"
        )
        bundle.rename(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, choices=("S", "B", "L", "H"))
    parser.add_argument("--mlx-path", required=True, type=Path)
    parser.add_argument(
        "--repo-id", help="Repository ID for the example in the model card"
    )
    args = parser.parse_args()
    convert(args.variant, args.mlx_path, args.repo_id or args.mlx_path.name)
    print(f"Saved {args.mlx_path}")


if __name__ == "__main__":
    main()
