# Deriving the best settings

"Best" means a sound first run for this model, this data, and this machine, with every choice explained. These are starting points drawn from what the commands report, not guarantees; a pilot run is what confirms them. Work through the steps in order. Each uses numbers you already have from `limits`, `validate-model`, and `inspect-dataset`.

## 1. Inputs

| Symbol | Where it comes from |
|---|---|
| `rows` | `inspect-dataset` → `splits.train.rows` |
| `chars_p95`, `chars_max` | `inspect-dataset` → `splits.train.text_chars` |
| `has_valid` | a `valid` or `validation` split exists |
| `images` | `rows_with_images` is above zero |
| `weights_gb`, `quantization_bits` | `validate-model` |
| `memory_gb` | `limits` → `system.memory_gb` |
| caps | `limits` → `limits` |

## 2. Algorithm and training type

- `algorithm`: the first entry of the dataset's `algorithms`, unless the user named one. See [algorithms](algorithms.md).
- `train_type`: `lora`. Choose `dora` only when the user asks for it. Choose `full` only when the user asks for it and the model is small enough; see [memory](memory.md).
- `train_on_completions`: `true` for `sft` on chat or question/answer rows, so only the assistant replies are learned. Leave it off for `uft`.

## 3. Sequence length

Rows are cut at `max_seq_length` tokens. Estimate tokens from characters: about one token per 3.5 characters for English and most European text, about one per 1.5 for Chinese, Japanese, or Korean. Each image adds its own tokens, often a few hundred; only a pilot shows how many.

- `max_seq_length` = the p95 token estimate, rounded up to the next multiple of 128, plus room for images.
- Do not go above what memory allows; see [memory](memory.md). If you must cut below p95, tell the user what share of rows is shortened.
- If p95 is well under 2048, lowering `max_seq_length` saves memory and time at no cost.
- A row that would lose its image tokens is rejected, so with images keep the length above the longest prompt.

## 4. Run length

One epoch is `rows // batch_size` iterations (complete batches only).

| Training rows | Passes |
|---|---|
| under 1,000 | `epochs: 3` |
| 1,000 to 10,000 | `epochs: 2` |
| over 10,000 | `epochs: 1`, or an `iters` budget the user agrees to |

- Use `epochs` or `iters`, never both. Use `iters` when the user gives a number of steps or a time budget.
- Check the result against `max_iters`. If it is over, propose the largest run that fits and say it is shorter than a full pass.
- `grpo` generates text every step and is much slower. Start it with a few hundred iterations at most.

## 5. Optimizer

| Setting | `lora` / `dora` | `full` |
|---|---|---|
| `learning_rate` | `1e-4` | `1e-5` |
| `optimizer` | `adamw` (default) | `adamw` (default) |
| `weight_decay` | default | `0.01` |

- `lr_schedule: "cosine_decay"` with `warmup_steps` of about 5% of the optimizer updates. Updates are `iters / gradient_accumulation_steps`.
- `grad_clip: 1.0`.
- `lora_rank: 8` with `lora_alpha: 16`. Use rank 16 and alpha 32 when the task is broad or the dataset is over about 5,000 rows.
- Read each option's `requires` in `describe`: an option only applies when the option it names holds one of the listed values.

## 6. Batch

- `batch_size: 1` and `gradient_accumulation_steps: 4`. This behaves like a batch of 4 without its memory.
- Raise `batch_size` only after a pilot shows memory to spare. Rows are batched with others of similar length.

## 7. Reporting, validation, and saving

Let `total` be the run's iterations.

- `steps_per_report`: about `total / 50`, at least 1 and at most 10.
- With a validation split: `steps_per_eval` about `total / 10`, and `val_batches` no more than the validation rows.
- `steps_per_save`: about `total / 5`, so a crash loses little.
- Without a validation split, say so: overfitting cannot be seen, and a held-out split is worth making.

## 8. Fit and time

Estimate memory with [memory](memory.md). Then present the settings and offer a pilot.

## The pilot

A pilot is the same options with `iters: 20` and `steps_per_report: 1`. It counts as a run against the daily cap, so ask first.

Read from its `status`:

- `peak_memory` in GB: the real memory need at these settings.
- `iterations_per_second`: total minutes ≈ `total / iterations_per_second / 60`. Compare with `max_minutes`.
- `train_loss` should fall across the 20 steps. If it does not move, raise `learning_rate`; if it jumps or turns into `nan`, lower it.

Then adjust and present the final settings. Delete the pilot only if the user asks.

## What to tell the user

A table like this, with only the settings that differ from their defaults:

| Setting | Value | Why |
|---|---|---|
| `algorithm` | `sft` | the rows are chat messages |
| `train_on_completions` | `true` | learn the replies, not the prompts |
| `learning_rate` | `1e-4` | usual for LoRA; the default suits full training |
| `max_seq_length` | `1024` | 95% of rows fit; the longest are cut |
| `epochs` | `2` | about 9,000 rows |

Follow it with the run length, the memory estimate, and anything you could not determine.
