# Ling 3.1 Flash

Text generation support for the `bailing_hybrid` architecture released in
[`inclusionAI/Ling-3.1-flash`](https://huggingface.co/inclusionAI/Ling-3.1-flash).
The decoder combines Kimi Delta Attention with gated multi-head latent
attention, shared and routed experts, and layer-specific SwiGLU clipping.

Convert the official block-FP8 checkpoint with the standard converter:

```sh
python -m mlx_vlm.convert \
  --hf-path inclusionAI/Ling-3.1-flash \
  --mlx-path ./Ling-3.1-flash-4bit \
  -q --q-bits 4 --q-group-size 64

python -m mlx_vlm.generate \
  --model ./Ling-3.1-flash-4bit \
  --trust-remote-code \
  --prompt "Explain why the sky is blue." --max-tokens 128
```

Add `--mtp` during conversion to extract the bundled NEXTN drafter into
`./Ling-3.1-flash-4bit-mtp`. Enable it during generation with
`--draft-model ./Ling-3.1-flash-4bit-mtp --draft-kind mtp --draft-block-size 2`.

Tensor parallelism uses the standard `examples/sharded_generate.py` entrypoint.
The number of ranks must divide the attention-head count and MLP widths.
