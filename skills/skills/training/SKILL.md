---
name: training
description: Use this skill when the user wants to train or fine-tune a model with MLX-VLM, or to check, start, monitor, stop, or configure a training run through mlx_vlm.train_runs, including choosing an algorithm and options, dataset row formats, LoRA/DoRA/full training, usage caps, and runs that were refused, capped, stopped, or failed.
---

# Training

Use this workflow to turn a training request into a validated background run with `mlx_vlm.train_runs`. Every command prints one JSON document; read it instead of guessing.

## Rules

1. Never assume which algorithms or options exist. Ask the backend with `describe`; new trainers appear there on their own.
2. Always run `check` before `start`. Start only when `check` reports `"valid": true`.
3. After `start`, follow the run with `status`. Do not claim a run finished until its `status` says so.
4. Report only what the commands return. Never invent losses, iteration counts, or paths.
5. Respect the usage caps. Never raise a cap, change `--runs-dir` or the `MLX_VLM_TRAIN_*` variables, or split a run into pieces to get around one.
6. Stop a run only when the user asks.

## Commands

Use `uv run mlx_vlm.train_runs ...` in a checkout, or `python -m mlx_vlm train_runs ...`.

| Command | Purpose |
|---|---|
| `describe` | Tasks, algorithms, and every option with type, default, choices, and help. |
| `limits` | The usage caps, active runs, and runs started in the last 24 hours. |
| `validate-model <model>` | Check the architecture is supported for training. Reads `config.json` only. |
| `inspect-dataset --options '<json>'` | Check the dataset loads; reports splits, row counts, and columns. |
| `check --options '<json>'` | Validate options with the real training parser. Starts nothing. |
| `start --options '<json>' [--name <label>]` | Start a run in the background; returns its `id`. |
| `status [run]` | Status, iteration, and latest metrics. No argument means the newest run. |
| `log [run] --lines N` | The end of the run's log. |
| `list` | Recent runs, newest first. |
| `stop [run]` | Stop a run. Saved checkpoints are kept. |

`run` is a run id, an unambiguous id prefix, or the `--name` label. Pass long option sets with `--options-file <path>` (`-` reads stdin).

## Workflow

1. **Learn the surface.** Run `describe`. Pick the task and algorithm that match the request, and take option names from that algorithm's `options` list. Options in the `managed` group are set by the run manager; do not pass them.
2. **Read the caps.** Run `limits` so the plan fits before you build it.
3. **Preflight.** Run `validate-model` on the model and `inspect-dataset` on the data. Fix problems here, before any run exists.
4. **Build the options.** A JSON object keyed by option name. Set `model`, the data source, the `algorithm`, and only the options the user asked for or that the request clearly needs. Leave the rest at their defaults.
5. **Check.** Run `check`. Show the user `errors` and `notes`. Fix and repeat until valid.
6. **Start.** Run `start` with the same options. Tell the user the run `id` and where the weights will be saved (`adapter_path`).
7. **Follow.** Run `status` to report `iteration` of `iters` and the latest metrics. Use `log` when the status looks wrong.

```bash
uv run mlx_vlm.train_runs describe
uv run mlx_vlm.train_runs limits
uv run mlx_vlm.train_runs validate-model mlx-community/Qwen3.5-0.8B-4bit
uv run mlx_vlm.train_runs inspect-dataset --options '{"data": "data"}'
uv run mlx_vlm.train_runs check --options '{"model": "mlx-community/Qwen3.5-0.8B-4bit", "algorithm": "sft", "data": "data", "epochs": 2}'
uv run mlx_vlm.train_runs start --options '{"model": "mlx-community/Qwen3.5-0.8B-4bit", "algorithm": "sft", "data": "data", "epochs": 2}' --name colors
uv run mlx_vlm.train_runs status colors
```

## Choosing the Algorithm

Pick from the data the user has. `describe` is the truth; if it lists something this section does not, read its description and options there.

| Algorithm | Data | What it does | Cost |
|---|---|---|---|
| `sft` | Chat or question/answer rows | Teaches the model to produce the answers. Set `train_on_completions` so only assistant replies are learned. | Baseline |
| `uft` | Plain text | Learns the whole sequence; adapts a model to a domain or style. Not for `train_on_completions`. | Baseline |
| `orpo` | `prompt`, `chosen`, `rejected` | Learns the chosen answer while pushing away the rejected one. No second model. | About baseline |
| `dpo` | `prompt`, `chosen`, `rejected` | Compares against a frozen copy of the model, which keeps it close to its original behaviour. | Loads the model twice |
| `grpo` | `prompt`, `answer`, a reward function | The model writes `group_size` answers per step and is rewarded for better ones. `max_completion_length` must be below `max_seq_length`. | Slowest: generates every step |

If two fit, ask the user.

## Choosing Settings

Set only what the user asked for or the task needs, leave the rest at defaults, and tell the user which settings you chose and why.

Quality:

- **`train_type`**: `lora` trains small adapters and is the default; `dora` is a little heavier; `full` updates every weight and only fits small models.
- **`lora_rank`**: adapter capacity. 8 is a good start, 16-32 for harder tasks; keep `lora_alpha` near twice the rank.
- **`learning_rate`**: about `1e-4` for `lora`/`dora`, `1e-5` or lower for `full`. Too high makes the loss jump or turn into `nan`; too low and it barely moves.
- **Length**: `epochs` when the user speaks in passes over the data, otherwise `iters`; never both. 1-3 epochs suits small datasets; more tends to memorise.
- **Schedule**: `lr_schedule: "cosine_decay"` with `warmup_steps` of 5-10% of the run gives a smoother finish. `grad_clip: 1.0` guards against spikes.
- **Validation**: a `valid.jsonl` split enables `val_loss` every `steps_per_eval`; use it to see overfitting.

Memory:

- **`max_seq_length`** is the main lever: set it just above the longest row. Longer rows are cut, and a row that would lose its image tokens is rejected.
- **`batch_size: 1`** with `gradient_accumulation_steps` of 4-8 behaves like a larger batch without the memory.
- **`grad_checkpoint: true`** trades speed for memory. `quantization_bits: 4` shrinks the base model for `lora`/`dora`.
- **`train_vision`** also trains the image encoder; use it only when the images are unlike what the model knows.

Other:

- **Model and data**: a Hugging Face id or a local path. `data` is a folder with `train.jsonl` (and optionally `valid.jsonl`), a `.jsonl` file, or a Hub dataset; use `dataset` for one loaded with Hugging Face `datasets`. Local paths may be relative; the manager makes them absolute.
- **Other columns**: pass custom column names through `dataset_config`, as described in the option's help.

For an untested model and dataset, offer a short pilot of 20-50 iterations first to confirm the loss falls and memory fits. It counts as a run against the daily cap.

## Reading a Run

- **Healthy**: `train_loss` falls and token accuracy rises.
- **Overfitting**: `val_loss` rises while `train_loss` keeps falling. Stop earlier, or lower `learning_rate`.
- **Flat loss**: raise `learning_rate`, or add capacity with a higher `lora_rank`.
- **`nan` or spikes**: lower `learning_rate` and set `grad_clip`.
- **Out of memory**: lower `max_seq_length`, turn on `grad_checkpoint`, or quantize the base.

Loss numbers are training diagnostics. Whether the model is actually better at the task still needs a look at its outputs.

## Dataset Rows

Supervised fine-tuning rows hold chat messages and optional images:

```json
{"images": ["images/red.png"], "messages": [{"role": "user", "content": "Which color?"}, {"role": "assistant", "content": "Red."}]}
```

`conversations` with `from`/`value` and plain `question`/`answer` rows are also accepted. Image paths are relative to the dataset folder.

Preference rows share media and a prompt:

```json
{"images": ["images/red.png"], "prompt": "Which color?", "chosen": "Red.", "rejected": "Blue."}
```

Reward-based rows hold a prompt and the answer the reward checks:

```json
{"prompt": "What color is the object?", "images": ["images/red.png"], "answer": "red"}
```

Check `inspect-dataset` output against the format the chosen algorithm needs before starting.

## Usage Caps

`limits` reports five caps; `null` means unlimited.

| Cap | What happens at the cap |
|---|---|
| `max_concurrent_runs` | `start` is refused while that many runs are active. |
| `max_runs_per_day` | `start` is refused once that many runs began in the last 24 hours. |
| `max_iters` | `check` and `start` refuse a longer run, including one whose `epochs` work out longer. |
| `max_minutes` | The run saves its weights and ends as `capped`. |
| `max_memory_gb` | The run saves its weights and ends as `capped`. |

The caps belong to whoever set up the trainer. Work inside them.

### When a run is refused

`check` or `start` returns an error naming the cap. Tell the user which cap and its value, then offer something that fits:

- **Concurrent runs:** report the active run from `list` and its progress. Offer to wait for it, or to stop it if the user wants the new run more. Do not stop it on your own.
- **Runs per day:** say the limit is reached and when the oldest run of the last 24 hours ages out (its `created_at` plus one day).
- **Iterations:** offer a run at or under `max_iters`, for example fewer `epochs`. Say plainly that it is shorter than asked.
- **Epoch-derived length:** an `epochs` run can pass `check` and still be refused once the dataset is loaded. It then ends `failed` with an "over the cap" error in `status`; treat it as a refusal and offer fewer epochs or an explicit `iters`.

### When a run is cut off

A run that ends `capped` is not a failure. Its `status` has a `capped` reason and the weights up to that point are saved in `adapter_path`.

1. Tell the user which cap ended the run and the `iteration` reached out of `iters`.
2. Report the last metrics so they can judge whether the result is already useful.
3. Offer to continue in a new run: the same options plus `resume_adapter_file` set to the saved `adapters.safetensors`, with `iters` set to what remains. The optimizer starts fresh. This counts as a new run against the daily cap.
4. For a memory cap, continuing unchanged will stop again. Offer a lighter setup first: smaller `batch_size` or `max_seq_length`, `grad_checkpoint`, or a quantized base.

Do not restart a capped run automatically. Ask first.

## Run States

| Status | Meaning | What to do |
|---|---|---|
| `running` | Training is in progress. | Report progress from `status`. |
| `completed` | Finished all iterations. | Report final metrics and `adapter_path`. |
| `capped` | Ended at a time or memory cap; weights saved. | See "When a run is cut off". |
| `stopped` | Ended by `stop`. | Checkpoints saved so far are in `adapter_path`. |
| `failed` | The process exited with an error. | Read `error` in `status` and `log --lines 80`; report the real error. |

For a failed run, fix the cause before trying again; do not retry unchanged. Common causes are a dataset row in the wrong format, a model that does not fit in memory, and a missing optional dependency (`pip install "mlx-vlm[train]"`).

## Using the Result

Weights and numbered checkpoints are in the run's `adapter_path` beside `adapter_config.json`; keep them together. With `lora` or `dora` this is an adapter that only works on top of its base model: load it with `--adapter-path <adapter_path>` on `mlx_vlm.generate`; see `Skill("mlx-vlm-skills:cli-inference")`.

For a standalone model, set `export_qat_path` to a new folder in the options. When the run ends the trained weights are fused into a full model there, loadable like any other model. It works for `lora`, `dora`, and `full`, the folder must not exist yet, and `status` reports it as `export_path`.

`history [run]` returns every metric as `[iteration, value]` points, for plotting or for judging the shape of the loss.

## Pitfalls

- Starting without `check`, or after `check` reported errors.
- Hard-coding algorithm or option names instead of reading `describe`.
- Passing `adapter_file`, `output_path`, `progress_file`, or any `max_*` option; the manager owns them and ignores yours.
- Polling `status` in a tight loop. Check at a pace that matches the run's length.
- Calling a `capped` run a failure, or continuing it without asking.
