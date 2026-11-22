# Runs

Each run is a background process with its own folder. Runs keep going when the caller exits, and their state is read from disk by every command.

## States

| Status | Meaning | What to do |
|---|---|---|
| `running` | Training is in progress. | Report progress from `status`. |
| `completed` | Finished all iterations. | Report final metrics and `adapter_path`. |
| `capped` | Ended at a time or memory cap; weights and trainer state saved. | Report where it got to and offer `continue`. |
| `stopped` | Ended by `stop`; weights and trainer state saved. | Offer `continue` when the user wants more. |
| `failed` | The process exited with an error. | Read `error` and `log`; see [troubleshooting](troubleshooting.md). |

`status` also returns `can_continue`, `continued_from`, `notes`, `iteration`, `iters`, and `metrics`.

## Reading the metrics

`status` gives the latest values; `history` gives every metric over time. Judge the shape, not one number.

- **Healthy**: `train_loss` falls and `train_token_accuracy` rises, quickly at first and then more slowly.
- **Overfitting**: `val_loss` rises while `train_loss` keeps falling. The best weights were around the lowest `val_loss`. Stop earlier, lower `learning_rate`, or train fewer epochs.
- **Flat loss**: `train_loss` barely moves. Raise `learning_rate`, or add capacity with a higher `lora_rank`.
- **Spikes or `nan`**: lower `learning_rate` and set `grad_clip: 1.0`.
- **Loss near zero very early**: the dataset is tiny or repeats itself; the model is memorising.
- **`learning_rate`** in `history` shows the schedule actually used. Check it when a schedule was set.

For `grpo`, read the reward metrics it reports and `sample` for the latest output. Rising reward with sensible outputs is the goal; rising reward with odd outputs means the reward is being gamed.

Loss numbers are diagnostics. Whether the model is better at the task still needs a look at its outputs.

## Pace

Check `status` at a pace that fits the run: every few seconds for a pilot, every few minutes for a long run. Report when something changes, not every poll.

## Usage caps

`limits` reports five caps; `null` means unlimited. They belong to whoever set up the trainer. Work inside them.

| Cap | What happens at the cap |
|---|---|
| `max_concurrent_runs` | `start` is refused while that many runs are active. |
| `max_runs_per_day` | `start` is refused once that many runs began in the last 24 hours. |
| `max_iters` | `check` and `start` refuse a longer run, including one whose `epochs` work out longer. |
| `max_minutes` | The run saves its weights and ends as `capped`. |
| `max_memory_gb` | The run saves its weights and ends as `capped`. |

When `check` or `start` is refused, name the cap and its value and offer something that fits:

- **Concurrent runs**: report the active run from `list` and its progress. Offer to wait, or to stop it if the user wants the new run more. Do not stop it on your own.
- **Runs per day**: say the limit is reached and when the oldest run of the last 24 hours ages out.
- **Iterations**: offer a run at or under `max_iters`, and say it is shorter than asked.
- **Epoch-derived length**: a run set in `epochs` can pass `check` and still end `failed` with an "over the cap" error once the dataset is loaded. Treat it as a refusal and offer fewer epochs or an explicit `iters`.

## Stop and continue

- `stop` asks the run to end at its next report. It saves its weights and its trainer state first, which can take up to half a minute.
- `continue <run>` starts a new run that carries on a `stopped` or `capped` run exactly: same optimizer state, learning-rate schedule, random state, and place in the data. It trains only the iterations that were left, and its metrics continue the iteration count.
- A run that `failed` or was killed saved no state. Continuing it starts from its last saved weights with a fresh optimizer, and `notes` says so. Tell the user.
- A continuation is a new run: it has its own `id`, records the earlier one as `continued_from`, and counts against the daily cap.
- Validation and checkpoint intervals count from where the new run starts.
- Do not continue a run automatically. Ask first. For a run that ended at a memory cap, lighten the setup first.

## Using the result

- Weights and numbered checkpoints are in the run's `adapter_path`, beside `adapter_config.json`. Keep them together.
- With `lora` or `dora` this is an adapter that only works on top of its base model. Load it with `--adapter-path <adapter_path>` on `mlx_vlm.generate`; see `Skill("mlx-vlm-skills:cli-inference")`.
- For a standalone model, set `export_qat_path` to a new folder in the options. When training completes the trained weights are fused into a full model there, and `status` reports the destination as `export_path`. Stopped and capped runs defer export until completion.

## Housekeeping

- `list` shows recent runs. `config <run>` returns a run's options for starting another like it.
- `delete <run>` removes a run's folder: its log, metrics, and saved weights. A merged model saved elsewhere is kept. Only delete when the user asks, and never a running run.
