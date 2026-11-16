# MiniCPM-OCR Jev

Jev is a 2B Qwen3.5 vision-language classifier that scores whether candidate OCR
text and its rendered image match the original document crop.

Each score requires the original crop, the candidate text, and an image rendered
from that text. Render the text separately before scoring; a renderer is not
included.

## Convert and score

```bash
mlx_vlm.convert --hf-path /path/to/Jev-checkpoint \
  --mlx-path /path/to/MiniCPM-OCR-Jev-bf16 --dtype bfloat16

python -m mlx_vlm.models.minicpm_ocr_jev \
  --model /path/to/MiniCPM-OCR-Jev-bf16 \
  --orig-image original-crop.png --render-image rendered-ocr.png \
  --ocr-text-file candidate.txt
```

Use `--ocr-text "Candidate OCR text"` to pass text directly instead of reading a
file. The CLI prints the result as JSON.

## Python API

```python
from mlx_vlm import load
from mlx_vlm.models.minicpm_ocr_jev import score

model, processor = load("/path/to/MiniCPM-OCR-Jev-bf16")
result = score(model, processor, "original-crop.png", "rendered-ocr.png",
               "Candidate OCR text")
print(result["prob_1"])
```

`score` accepts image paths or PIL images and preserves the candidate text as
provided. It builds the scoring prompt automatically.

## Results

Both the CLI and Python API return:

- `prob_1`: P(consistent). Higher values indicate a closer match.
- `prob_0`: P(inconsistent).
- `logit_0` and `logit_1`: the raw classification scores.
- `n_tokens`: the number of input tokens, including image tokens.

No decision threshold is applied automatically. Choose a threshold using
representative data and the precision and input pipeline used in deployment.
