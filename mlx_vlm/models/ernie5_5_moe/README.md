# ERNIE 5.5 Flash Base

MLX support for [baidu/ERNIE-5.5-Flash-Base](https://huggingface.co/baidu/ERNIE-5.5-Flash-Base). This is a text completion model; use plain prompts rather than chat messages.

## Convert

```sh
python -m mlx_vlm.convert \
  --hf-path baidu/ERNIE-5.5-Flash-Base \
  --mlx-path ./ERNIE-5.5-Flash-Base-MXFP8
```

This converts the released FP8 projections to native MLX MXFP8. The source download is about 288 GB; allow additional space for the output.

Add `-q --q-bits 4` to convert routed experts to affine 4-bit, keeping other eligible projections in MXFP8. Without `-q`, the released FP8 projections stay MXFP8.

For routed experts, `-q --q-bits 8` uses MXFP8; `-q --q-mode nvfp4` or `mxfp4` uses the requested 4-bit format. Other eligible projections stay MXFP8.

## Generate

```sh
python -m mlx_vlm.generate \
  --model ./ERNIE-5.5-Flash-Base-MXFP8 \
  --prompt "Large language models are" \
  --max-tokens 128 --temperature 0
```

## ECHO (experimental)

ECHO uses the draft weights bundled in the same checkpoint:

```sh
python -m mlx_vlm.generate \
  --model ./ERNIE-5.5-Flash-Base-MXFP8 \
  --draft-model ./ERNIE-5.5-Flash-Base-MXFP8 \
  --draft-kind mtp --draft-block-size 3 \
  --prompt "Large language models are" \
  --max-tokens 128 --temperature 0
```

Block size `3` proposes two draft tokens. ECHO requires full-prompt prefill and a fresh cache. Outputs can differ from ordinary decoding, and speedups depend on the workload.

## Tensor parallelism

Use the same checkpoint and checkout on each host:

```sh
mlx.launch --hostfile hosts.json --backend jaccl \
  python examples/sharded_generate.py \
  --model ./ERNIE-5.5-Flash-Base-MXFP8 \
  --prompt "Large language models are" --max-tokens 128
```

This example runs ordinary decoding. Distributed ECHO is available through the Python API.

Attention heads and FFN intermediate dimensions are partitioned; ECHO reuses the sharded final decoder layer. Its auxiliary weights, latent caches, routers, embeddings and vocabulary head are replicated. Rank counts must divide the attention output groups and preserve whole quantization groups (2, 4 or 8 for the released checkpoint). Pipeline parallelism is not supported.

[Reference implementation](https://github.com/baidu/vLLM-ERNIE/tree/main/src/vllm_ernie/models/ernie5_5) · [Model license](https://huggingface.co/baidu/ERNIE-5.5-Flash-Base/blob/main/LICENSE)
