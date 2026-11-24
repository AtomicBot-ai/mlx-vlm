# Memory and speed

On Apple silicon, memory is shared between the system and the GPU. `limits` reports `system.memory_gb`; not all of it is free for training.

## What uses memory

1. **The model's weights.** `validate-model` reports `weights_gb` when the model is on disk. A 4-bit model is about a quarter the size of its full-precision version.
2. **Activations and gradients.** These grow with `max_seq_length` and `batch_size`, and with the number of images.
3. **Optimizer state.** Small for `lora` and `dora`. For `full` it is about twice the size of the trained weights with `adam` or `adamw`.
4. **Extra models.** `dpo` loads the model twice. `grpo` keeps generation caches for `group_size` answers.
5. **The merged model**, when `export_qat_path` is set: one more copy at the end of the run.

## A first estimate

There is no formula that holds across models. Use these to decide whether a run is plausible, and say it is an estimate:

- `lora` or `dora` on a quantized model at short lengths needs roughly two to three times `weights_gb`. One measurement: a model with 0.63 GB of 4-bit weights peaked at 1.2 GB with text rows at `max_seq_length: 128`, and at 2.0 GB with small images at the default length.
- Doubling `max_seq_length` or `batch_size` roughly doubles the activation share.
- `full` training needs several times `weights_gb`. Expect it to fit only when the weights are a small fraction of `system.memory_gb`.
- If the estimate is above about 70% of `system.memory_gb`, plan a lighter setup before starting.

A 20-iteration pilot gives the real figure as `peak_memory`.

## What to lower first

In this order, stopping when it fits:

1. `max_seq_length`, down to the dataset's p95 length.
2. `batch_size: 1`, keeping the effective batch with `gradient_accumulation_steps`.
3. `grad_checkpoint: true`: less memory, slower steps.
4. A quantized base: a 4-bit model, or `quantization_bits: 4` for `lora` and `dora`.
5. For `grpo`: a smaller `group_size` and `max_completion_length`.
6. `orpo` instead of `dpo`, to avoid the second model.
7. A smaller model.

Leave `train_vision` off unless the images are unlike what the model knows; training the image encoder adds memory.

## Memory cap

When `limits` shows `max_memory_gb`, a run that passes it saves its weights and ends as `capped`. Continuing it unchanged will stop again; lighten the setup first.

## Speed

- `status` reports `iterations_per_second`. Total minutes ≈ iterations left / `iterations_per_second` / 60.
- The first iterations are slower while data is prepared; read the rate after a few reports.
- `grad_checkpoint` and longer rows slow each step. `grpo` is the slowest because it generates text every step.
- Compare the estimate with `max_minutes` before starting, and tell the user if the run will be cut off.
