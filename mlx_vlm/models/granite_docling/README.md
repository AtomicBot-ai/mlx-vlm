# Granite for Docling

Support for the `granite_docling` architecture
(`GraniteDoclingForConditionalGeneration`). The model converts a page image to
DocLang markup: text, layout boxes, tables, and pictures.

The model has a SigLIP-style encoder for 512x512 tiles, a pixel-shuffle
connector with fixed 2D positions, and a Granite text decoder. The connector
also makes DeepStack features, which are added to the image tokens after the
first decoder layers.

Supported checkpoints:

- [`ibm-granite/granite-for-docling-2b`](https://huggingface.co/ibm-granite/granite-for-docling-2b)
- `ibm-granite/granite-for-docling-500m` (fine connector path and density router)

The processor is `GraniteDoclingProcessor` from Transformers. It needs
`transformers>=5.23.0`.

## Usage

```sh
python -m mlx_vlm.convert \
  --hf-path ibm-granite/granite-for-docling-2b \
  --mlx-path ./granite-for-docling-2b-8bit \
  -q --q-bits 8

python -m mlx_vlm.generate \
  --model ./granite-for-docling-2b-8bit \
  --image page.png --prompt "<doclang>" \
  --max-tokens 4096 --temperature 0
```

Use `<doclang>` to convert the full page. To select tasks, give them in
brackets, for example `<doclang> [<table>, <layout>]`.

## Fine connector path

The 500M checkpoint has a fine connector path, which gives four times as many
image tokens per tile, for dense pages. To use it, set the processor default
before you generate:

```python
from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template

model, processor = load("./granite-for-docling-500m-mlx")
processor.image_processor.fine_route = True
prompt = apply_chat_template(processor, model.config, "<doclang>", num_images=1)
result = generate(model, processor, prompt, image=["page.png"], max_tokens=4096)
```

`Model.predict_fine_route(pixel_values, pixel_attention_mask)` uses the density
router to tell which pages need the fine path. The 2B checkpoint has no fine
path.

## Limits

- No tool calling: the chat template has no tool format.
- The `mtp.*` heads in the checkpoint are not loaded.
