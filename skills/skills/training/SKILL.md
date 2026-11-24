---
name: training
description: Use this skill when the user wants to train or fine-tune a model with MLX-VLM, asks for the best training settings for a model and dataset, asks whether a model or dataset can be trained or will fit in memory, or wants to check, start, monitor, stop, continue, compare, or debug training runs through mlx_vlm.train_runs, including algorithm choice, dataset row formats, LoRA/DoRA/full training, usage caps, and runs that were refused, capped, stopped, or failed.
---

# Training

Turn a training request into a validated background run with `mlx_vlm.train_runs`, and answer training questions from what its commands report. Every command prints one JSON document; read it instead of guessing.

Work autonomously. Anything a command can tell you, find out yourself instead of asking the user. Ask only for what no command can answer: which model, which data, and what the model should learn. Make the remaining decisions, say what you chose and why, and ask for one confirmation before the step that costs time: `start`.

## Rules

1. Never assume which algorithms or options exist. `describe` is the truth; if it differs from this skill, follow `describe`.
2. Run `check` before every `start`. Start only when `check` reports `"valid": true`.
3. Do not `start` a run the user has not asked for or agreed to. A request for settings, advice, or a check is not a request to train.
4. After `start`, follow the run with `status`. Do not say a run finished until `status` says so.
5. Report only what the commands return. Never invent losses, iteration counts, memory figures, or paths. Mark every estimate as an estimate.
6. Respect the usage caps. Never raise a cap, change `--runs-dir` or the `MLX_VLM_TRAIN_*` variables, or split a run into pieces to get around one.
7. `stop`, `continue`, and `delete` only when the user asks or agrees.

## Commands

Use `uv run mlx_vlm.train_runs ...` in a checkout, or `python -m mlx_vlm train_runs ...`.

| Command | Purpose |
|---|---|
| `describe` | Tasks, algorithms, and every option with type, default, choices, help, and `requires`. |
| `limits` | The usage caps, active runs, runs in the last 24 hours, and the machine's memory. |
| `validate-model <model>` | Whether the architecture can be trained, its quantization, and its weight size if it is on disk. |
| `inspect-dataset --options '<json>'` | Splits with row counts, columns, the row format, matching algorithms, and text lengths. |
| `check --options '<json>'` | Validate options with the real training parser. Starts nothing. |
| `start --options '<json>' [--name <label>]` | Start a run in the background; returns its `id`. |
| `status [run]` | Status, iteration, and latest metrics. No argument means the newest run. |
| `log [run] --lines N` | The end of the run's log. |
| `history [run]` | Every metric as `[iteration, value]` points. |
| `list` | Recent runs, newest first. |
| `config [run]` | The options a run was started with. |
| `sample [run]` | The latest generated output of a run that generates (`grpo`). |
| `stop [run]` | Stop a run. It saves its weights and trainer state first, so it can be continued. |
| `continue [run]` | Carry on an ended run in a new run. |
| `delete <run>` | Remove an ended run and its files. |

`run` is a run id, an unambiguous id prefix, or the `--name` label. Pass long option sets with `--options-file <path>` (`-` reads stdin). Options are a JSON object keyed by the option names from `describe`. Options in the `managed` group are set by the run manager; do not pass them.

## Playbooks

Pick the playbook that matches the request and work through its steps in order. Each step names the command that answers it.

### "Train this model on this data"

1. `describe`: confirm the task and note the algorithms.
2. `limits`: read the caps and `system.memory_gb`.
3. `validate-model <model>`: stop here if it is not supported.
4. `inspect-dataset`: read `format`, `algorithms`, `rows`, and `text_chars`.
5. Choose the algorithm from the dataset's `algorithms`; see [algorithms](references/algorithms.md).
6. Derive the settings with the recipe in [settings](references/settings.md).
7. `check`: fix every error and repeat until valid.
8. Show the user the settings table and the estimates, then ask to start.
9. `start`, then report the run `id` and `adapter_path`.
10. Follow with `status`; judge the run with [runs](references/runs.md).
11. When it ends, report the final metrics, where the weights are, and how to use them.

### "Give me the best settings for this model and dataset"

Steps 1 to 7 above, then stop before `start`. Reply with:

- a table of every setting you would change from its default, its value, and one reason each;
- the estimated run length in iterations and, if a pilot ran, in minutes;
- the estimated memory fit, and what you would lower first if it does not fit;
- what you could not determine, and the pilot that would settle it.

Offer the pilot from [settings](references/settings.md) when the model and dataset have not been trained together before. Do not start it without a yes.

### "Will this fit?" / "How long will it take?"

1. `limits` for `system.memory_gb`, `validate-model` for `weights_gb` and `quantization_bits`, `inspect-dataset` for lengths.
2. Estimate with [memory](references/memory.md). Say it is an estimate.
3. For a real number, offer a 20-iteration pilot: its `status` reports `peak_memory` and `iterations_per_second`.

### "Is my dataset OK?"

1. `inspect-dataset`. Report splits, rows, `format`, and `text_chars`.
2. Compare the columns with what the algorithm needs; see [datasets](references/datasets.md).
3. Report what is missing or risky: no validation split, very long rows, a `format` of `unknown`, too few rows.

### "How is my run doing?" / "Is it any good?"

1. `status` for the state, iteration, and latest metrics.
2. `history` for the shape of the loss, not just the last value.
3. For `grpo`, `sample` for the latest output.
4. Judge it with the signs in [runs](references/runs.md) and say what you would change.

### "It failed" / "It stopped"

1. `status` for `status`, `error`, `capped`, and `notes`.
2. `log --lines 80` for the real error.
3. Match the cause in [troubleshooting](references/troubleshooting.md) and fix it before any retry.
4. For a stopped or capped run, offer `continue`; see [runs](references/runs.md).

### "Do it again with a change" / "Compare these runs"

1. `config <run>` returns the earlier run's options. Change only what was asked, then `check` and `start`.
2. To compare, read `status` and `history` of each run and compare the same metric at the same iteration. A comparison is fair only when one setting changed.

## Reporting

- Lead with the outcome: started, finished, refused, or failed, and the run `id`.
- Give numbers with their source: "`train_loss` 0.42 at iteration 300 of 500".
- List the settings you chose and the reason for each. Keep defaults out of the list.
- Loss is a training diagnostic. Whether the model is better at the task needs a look at its outputs; say so when the user asks if it is good.

## References

- [Deriving the best settings](references/settings.md): the recipe, formulas, and the pilot.
- [Algorithms](references/algorithms.md): which one fits which data, and their own options.
- [Datasets](references/datasets.md): row formats, splits, and custom columns.
- [Memory and speed](references/memory.md): estimating fit, and what to lower first.
- [Runs](references/runs.md): states, caps, stop and continue, reading metrics, using the result.
- [Troubleshooting](references/troubleshooting.md): errors and their fixes.
- [Worked examples](references/examples.md): complete option sets for common requests.

## Pitfalls

- Asking the user for facts `inspect-dataset`, `validate-model`, or `limits` would have returned.
- Starting without `check`, or after `check` reported errors.
- Leaving `learning_rate` at its default for `lora`; the default suits `full` training.
- Hard-coding algorithm or option names instead of reading `describe`.
- Passing `adapter_file`, `output_path`, `progress_file`, `resume_state`, or any `max_*` option; the manager owns them.
- Setting both `epochs` and `iters`.
- Polling `status` in a tight loop. Check at a pace that matches the run's length.
- Calling a `capped` or `stopped` run a failure, or continuing it without asking.
