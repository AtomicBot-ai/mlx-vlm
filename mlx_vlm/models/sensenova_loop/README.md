# SenseNova-Loop

Native MLX text and image inference for
[`sensenova/SenseNova-Loop-9B`](https://huggingface.co/sensenova/SenseNova-Loop-9B).
Download access requires an authorized Hugging Face account. The model and
processor run locally without executing the checkpoint's Python model code.

```python
from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template

model, processor = load("sensenova/SenseNova-Loop-9B")
prompt = apply_chat_template(
    processor, model.config, "Describe this image.",
    num_images=1, enable_thinking=False,
)
result = generate(
    model, processor, prompt, image=["image.jpg"],
    max_tokens=128, temperature=0,
)
print(result.text)
```

For text-only generation, set `num_images=0` and omit `image`. Pass
`enable_thinking=True` to use the checkpoint's thinking template. Multiple
images use `num_images=len(images)` and the corresponding image list.

## Conversion

```sh
python -m mlx_vlm convert \
  --hf-path sensenova/SenseNova-Loop-9B \
  --mlx-path ./SenseNova-Loop-9B-4bit \
  -q --q-bits 4 --q-group-size 64

python -m mlx_vlm generate \
  --model ./SenseNova-Loop-9B-4bit \
  --image image.jpg --prompt "Describe this image." --max-tokens 128
```

Conversion quantizes the language projections and embeddings while retaining
the vision tower in its original precision. The mHC router weights and biases
remain FP32. The training-only `loop_gate.logits` tensor is discarded; inference
uses the fixed `loop_settings.eval_count` recorded in the checkpoint.

## Architecture

The release has 32 physical Qwen3.5 decoder layers. Layers 4 through 27 execute
twice, yielding 56 execution visits. Weights are stored once per physical layer;
each visit has its own KV or recurrent cache. The checkpoint's explicit
`layer_types` determine which attention implementation each layer uses.

Within the mHC range, attention and feed-forward branches read from and write
to two residual streams. FP32 routers compute sigmoid read/write gates and a
mixing matrix normalized by 20 iterations of log-space Sinkhorn. Streams are
expanded and averaged at the mHC boundaries on each pass. Configuration
validation also supports four streams and fixed identity mixing, as in the
reference implementation.

The port reuses Qwen3.5's vision tower, YaRN M-RoPE, processing, and padded-batch
handling. Chunked prefill and autoregressive decoding retain independent state
for repeated visits. Speculative decoding through the Qwen3.5 fused verifier
is explicitly rejected because it bypasses mHC routing. Long-context behavior
at the checkpoint's full 512K limit has not been validated.

The routing and execution-plan implementation is adapted from the
[Apache-2.0 reference code](https://huggingface.co/sensenova/SenseNova-Loop-9B/blob/329a65f701ba8c48b73c30d0cbaa6f0449733673/modeling_sensenova_loop.py).
