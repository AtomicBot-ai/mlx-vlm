# Inkling 2 Mini and Small

Text and image inference for
[`thinkingmachines/Inkling-2-Mini`](https://huggingface.co/thinkingmachines/Inkling-2-Mini)
and
[`thinkingmachines/Inkling-2-Small`](https://huggingface.co/thinkingmachines/Inkling-2-Small).
Load the native checkpoint directly; no conversion is required. The loader
preserves mixed BF16/FP32 parameters and packed NVFP4/MXFP8 experts. Keep
`hf_quant_config.json` with the checkpoint.

## Generate with Mini

Mini's checkpoint is about 20 GB including its DSpark drafter. The decoder
visits 24 distinct layers, running the middle 12 three times for a total of
48 layer applications per token. Each visit has its own attention and
convolution cache.

```sh
mlx_vlm.generate --model thinkingmachines/Inkling-2-Mini \
  --image /path/to/image.jpg --prompt "What animal is in this image?" \
  --reasoning-effort none --temperature 0 --max-tokens 256
```

For text-only generation, omit `--image`. To use Mini's bundled drafter,
add `--draft-model thinkingmachines/Inkling-2-Mini`.

Tool calling works through `mlx_vlm.server`: send `tools` with a request and
the model's `invoke_tool_json` messages come back as structured tool calls.

## Generate with Small

Small's download is approximately 394 GB, including the bundled draft. Target weights
require about 390 GB of memory. On a 512 GB Mac, disable the allocation cache and
load layers sequentially to bound temporary memory:

```python
import mlx.core as mx
from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template

cache_limit = mx.set_cache_limit(0)
model, processor = load("thinkingmachines/Inkling-2-Small", lazy=True)
mx.eval(model.vision_tower.parameters())
for layer in model.layers:
    mx.eval(layer.parameters())
mx.eval(model.parameters())
mx.set_cache_limit(cache_limit)

prompt = apply_chat_template(
    processor, model.config, "Explain why the sky is blue.",
    reasoning_effort="none",
)
result = generate(
    model, processor, prompt, max_tokens=128, temperature=0,
    prefill_step_size=128, skip_special_tokens=True,
)
print(result.text)
```

For image input, add `num_images=1` to `apply_chat_template` and
`image=["/path/to/image.png"]` to `generate`. Multiple images, streaming,
batching, and prefix caching are supported.

## Distributed inference

Use the same checkout and checkpoint on each host:

```sh
mlx.launch --backend jaccl --hostfile /path/to/hosts.json \
  examples/sharded_generate.py \
  --backend jaccl --model thinkingmachines/Inkling-2-Small \
  --cache-limit 0 --max-tokens 128 --prompt "Explain why the sky is blue."
```

Use `ring` instead of `jaccl` on both flags for TCP. Add
`--image /path/to/image.png` for an image available at that path on every host.

Experts are partitioned across ranks; compatible two-rank Metal models also
partition attention heads and cache state. Two 512 GB Macs can share one target
model. Each host still needs the full checkpoint on disk. The standard
`sharded_load` and [cluster APIs](../../../docs/cluster-api.md) are supported.

## Bundled DSpark

Load the bundled `mtp.safetensors` through the standard drafter API; no extraction
or conversion is needed:

```python
from mlx_vlm.speculative.drafters import load_drafter

draft, kind = load_drafter("thinkingmachines/Inkling-2-Small")
result = generate(
    model, processor, prompt, max_tokens=128, temperature=0,
    draft_model=draft, draft_kind=kind, prefill_step_size=128,
    skip_special_tokens=True,
)
```

For CLI generation or the sharded example, add
`--draft-model thinkingmachines/Inkling-2-Small`. The default draft block contains
one anchor and seven proposals. Distributed DSpark supports the sharded Python/example
path and the [cluster API's draft option](../../../docs/cluster-api.md#speculative-decoding).

## Notes

- Inference uses floating-point activations, not the reference runtime's activation
  quantization. Strict numerical parity with the reference is not established.
- Prefill chunk size and speculative verification shape can change greedy outputs.
  DSpark speedups depend on the workload.
- Audio input and pipeline parallelism are not supported.
- Mini's weights and the reference loop implementation are licensed under
  [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0).
  [Transformers implementation](https://github.com/huggingface/transformers/pull/51050)
