"""
Run with:

```
mlx.launch \
    --backend jaccl \
    --env MLX_METAL_FAST_SYNCH=1 \
    --hostfile /path/to/hosts.json \
    /path/to/sharded_generate.py \
    --prompt 'Describe this image.' --image /path/to/image.jpg
```

For more information on running distributed programs with MLX see the documentation:

https://ml-explore.github.io/mlx/build/html/usage/distributed.html .
"""

import argparse

import mlx.core as mx

from mlx_vlm.generate import stream_generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.speculative.drafters import load_drafter
from mlx_vlm.utils import sharded_load

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM distributed inference example")
    parser.add_argument(
        "--model",
        default="mlx-community/Qwen3-VL-30B-A3B-Instruct-bf16",
        help="HF repo or path to local model.",
    )
    parser.add_argument(
        "--prompt",
        "-p",
        default="Write a quicksort in C++.",
        help="Message to be processed by the model ('-' reads from stdin)",
    )
    parser.add_argument(
        "--image",
        "-i",
        default=None,
        help="Path to image to be processed by the model",
    )
    parser.add_argument(
        "--max-tokens",
        "-m",
        type=int,
        default=256,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--backend",
        default="any",
        choices=("any", "jaccl", "ring", "mpi", "nccl"),
        help="Communication backend; use jaccl for Thunderbolt RDMA.",
    )
    parser.add_argument(
        "--pipeline",
        action="store_true",
        help="Use pipelining instead of tensor parallelism",
    )
    parser.add_argument(
        "--draft-model",
        help="Drafter checkpoint, or the target path for a bundled MTP/DSpark head.",
    )
    parser.add_argument("--draft-block-size", type=int, default=None)
    parser.add_argument(
        "--seed", type=int, default=0, help="Shared sampling seed for all ranks."
    )
    parser.add_argument(
        "--cache-limit",
        type=int,
        default=None,
        help="MLX allocation cache limit in bytes; use 0 to release loading buffers.",
    )
    args = parser.parse_args()

    if args.cache_limit is not None:
        mx.set_cache_limit(args.cache_limit)
    group = mx.distributed.init(strict=args.backend != "any", backend=args.backend)
    rank = group.rank()
    pipeline_group = group if args.pipeline else None
    tensor_group = group if not args.pipeline else None

    def rprint(*args, **kwargs):
        if rank == 0:
            print(*args, **kwargs)

    model, processor = sharded_load(args.model, tensor_group, pipeline_group)
    draft_options = {}
    if args.draft_model:
        draft, kind = load_drafter(args.draft_model)
        draft_options = dict(
            draft_model=draft, draft_kind=kind, draft_block_size=args.draft_block_size
        )

    prompt = apply_chat_template(
        processor,
        model.config,
        args.prompt,
        num_images=1 if args.image is not None else 0,
        num_audios=0,
    )

    for response in stream_generate(
        model,
        processor,
        prompt=prompt,
        image=args.image,
        max_tokens=args.max_tokens,
        seed=args.seed,
        **draft_options,
    ):
        rprint(response.text, end="", flush=True)

    rprint()
    rprint("=" * 10)
    rprint(
        f"Prompt: {response.prompt_tokens} tokens, "
        f"{response.prompt_tps:.3f} tokens-per-sec"
    )
    rprint(
        f"Generation: {response.generation_tokens} tokens, "
        f"{response.generation_tps:.3f} tokens-per-sec"
    )
    rprint(f"Peak memory: {response.peak_memory:.3f} GB")
