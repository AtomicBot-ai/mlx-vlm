# MiniMax-H3-Image

Native MLX text-to-image generation and image editing for
[`MiniMaxAI/MiniMax-H3-Image`](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image).
The model uses a Qwen3-VL conditioner, a 33B diffusion transformer with six
step-specific output heads, and a tiled image VAE.

## Generate

```sh
mlx_vlm.generate --output-modality image \
  --model MiniMaxAI/MiniMax-H3-Image \
  --prompt "A red ceramic teapot on a wooden table beside a sunlit window, soft morning light, detailed natural textures" \
  --size 4096x4096 --seed 42 --output teapot.png
```

```python
from mlx_vlm.generate import generate_image, load_image_generation_model

model = load_image_generation_model("MiniMaxAI/MiniMax-H3-Image")
result = generate_image(
    model,
    "A red ceramic teapot on a wooden table beside a sunlit window",
    width=4096, height=4096, seed=42,
    output_path="teapot.png",
)
```

## Edit

```sh
mlx_vlm.generate --output-modality image --task edit \
  --model MiniMaxAI/MiniMax-H3-Image \
  --image teapot.png \
  --prompt "Make the teapot in <Picture 1> blue, preserving its shape, the room, and the lighting" \
  --size 4096x4096 --seed 42 --output blue-teapot.png
```

The model accepts one to sixteen ordered reference images. Refer to them as
`<Picture 1>`, `<Picture 2>`, and so on. Each reference keeps its aspect ratio
and is encoded by both Qwen3-VL and the VAE; it is not resized to the target
canvas. Editing uses the model's native reference conditioning.

## Prompt expansion

Short prompts are expanded automatically using
[`MiniMaxAI/MiniMax-H3-Image-PE`](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image-PE).
Its released NVFP4 weights load directly through the existing Qwen3.5 MLX
implementation, adding approximately 18 GB to the first download. The expander
uses MiniMax's released task prompts, thinking mode, sampling parameters, and
constrained JSON output. It is released before loading the image components.

The caption sent to the image model is available in
`result.metadata["revised_prompt"]`; its recommended ratio is in
`result.metadata["suggested_aspect_ratio"]`. Generation keeps the width and
height you request. For editing, the expander sees the ordered references,
their original aspect ratios, and the requested target ratio. Upstream provides
the editing task prompt for use with a VLM; a separately trained editing
expander is not part of the release.

Use an already expanded caption without another expansion:

```sh
mlx_vlm.generate --output-modality image \
  --model MiniMaxAI/MiniMax-H3-Image \
  --prompt "A detailed caption that is ready for the image model" \
  --gen-kwargs '{"upsample_prompt":false}' \
  --size 4096x4096 --output image.png
```

In Python, pass `upsample_prompt=False` when loading or generating. Set
`prompt_expansion_model` to a compatible local model or Hub ID to override the
default; the CLI exposes the same choice as `--prompt-expansion-model`, and
the image API accepts both `upsample_prompt` and `prompt_expansion_model`.

Use the expander independently to inspect its caption and choose a canvas:

```python
from mlx_vlm.models.minimax_h3_image import MiniMaxH3PromptUpsampler

expander = MiniMaxH3PromptUpsampler()
expanded = expander("A red ceramic teapot beside a sunny window", seed=42)
print(expanded.expanded_caption, expanded.aspect_ratio)

edited = expander(
    "Make the teapot blue",
    images=["teapot.png"],
    aspect_ratio="1:1",  # Or "adaptive" for a recommendation.
    seed=42,
)
print(edited.expanded_caption, edited.aspect_ratio)
```

The standalone class also accepts `max_tokens` (4096 by default, including
thinking) and `reasoning_effort` (`xhigh`, `medium`, or `low`). Truncated or
invalid answers raise an error before image generation starts.

## Settings

- Sampling is fixed at **6 steps, guidance 1**. Other values and negative
  prompts are rejected.
- With prompt expansion disabled, use detailed English descriptions.
- Prompts are truncated to 4096 tokens, without a chat template.
- The model works best near 16 megapixels. Small requests are generated on
  a canvas of at least 4 megapixels and then resized to the requested size.
  `--gen-kwargs '{"min_generation_area":1024}'` disables that minimum.
  Internal canvas dimensions round to multiples of 32.

Weights use the [MiniMax H3 Image Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/blob/main/LICENSE).
The bundled prompt-expansion templates come from
[MiniMax-H3-Image-PE](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image-PE/tree/main/prompt)
under the same MiniMax H3 Image Community License as the model weights.
