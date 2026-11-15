# MiniCPM-OCR Jev

Jev is the supplied 2B Qwen3.5 vision-language **sequence classifier**. It scores
whether an OCR text and its rendered image match the original document crop.
The port uses its two-output `score` head, with no language-generation head.

The source checkpoint incorrectly declares `Qwen3_5ForConditionalGeneration`.
Loading detects `problem_type="single_label_classification"` and selects the
classifier. Explicit `Qwen3_5ForSequenceClassification` architecture metadata and
`model_type="minicpm_ocr_jev"` are also supported.

## Convert and score

```bash
mlx_vlm.convert --hf-path /path/to/Jev-checkpoint \
  --mlx-path /path/to/MiniCPM-OCR-Jev-bf16 --dtype bfloat16

python -m mlx_vlm.models.minicpm_ocr_jev \
  --model /path/to/MiniCPM-OCR-Jev-bf16 \
  --orig-image original-crop.png --render-image rendered-ocr.png \
  --ocr-text-file candidate.txt
```

```python
from mlx_vlm import load
from mlx_vlm.models.minicpm_ocr_jev import score

model, processor = load("/path/to/MiniCPM-OCR-Jev-bf16")
result = score(model, processor, "original-crop.png", "rendered-ocr.png",
               "Candidate OCR text")
print(result["prob_1"])
```

`prob_1` is **P(consistent)**; `prob_0` is P(inconsistent). A value near zero means
confidently inconsistent. This is not a symmetric confidence score. No decision
threshold is applied automatically.

The value-head prompt matches the [pinned OpenBMB reference](https://github.com/OpenBMB/MiniCPM-OCR/blob/74c8876056c536ee55b204cc842112c515764512/scripts/minicpm-ocr-jev/infer_vllm.py#L50-L115):
original image first, rendered image second, a newline after each image
placeholder, and the suffix
`<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`.
The value head pools the last token of this empty assistant/think turn.
Candidate text is passed unchanged.
The processor preserves the checkpoint's 65,536–16,777,216 pixel budget per image.
Images with aspect ratios above 199:1 are padded on a white canvas as in the
reference. Rendering OCR text into an image is an upstream step; a renderer is
not included in the supplied package or this port.

The low-level model returns `.logits` of shape `[batch, 2]`, pooled at the last
non-padding token. Standard left/right padding is supported; padded rows are
processed individually after image encoding to preserve recurrent state. The
score head is excluded from quantization. Shifted language normalization scales
remain FP32, with activations cast back to the input dtype.

To evaluate at FP32 precision, convert to FP32 and disable reduced-precision
matrix operations when launching:

```bash
mlx_vlm.convert --hf-path /path/to/Jev-checkpoint \
  --mlx-path /path/to/MiniCPM-OCR-Jev-fp32 --dtype float32

MLX_ENABLE_TF32=0 python -m mlx_vlm.models.minicpm_ocr_jev \
  --model /path/to/MiniCPM-OCR-Jev-fp32 \
  --orig-image original-crop.png --render-image rendered-ocr.png \
  --ocr-text-file candidate.txt
```

MLX can otherwise use reduced-precision matrix operations on supported hardware. See
[MLX numerical precision](https://ml-explore.github.io/mlx/build/html/usage/precision.html).
Earlier numerical comparisons used a prompt that omitted these image newlines
and the assistant/think suffix; they do not validate the corrected prompt.
Scores change with this correction. Revalidate numerical agreement and any
routing threshold with the corrected prompt, deployed precision, and input
pipeline.
