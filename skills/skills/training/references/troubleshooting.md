# Troubleshooting

Start with `status` for `error` and `notes`, then `log --lines 80` for the full message. Report the real error. Fix the cause before trying again; do not retry unchanged.

| What you see | Likely cause | Fix |
|---|---|---|
| `check` reports an unknown option | An option the chosen algorithm does not have, or a typo | Take names from `describe` for that algorithm. |
| `check` says options "need adam or adamw" | An Adam-only option set with another optimizer | Remove it, or read `requires` in `describe`. |
| "Usage cap" on `start` | A concurrent or daily cap | See caps in [runs](runs.md). |
| "over the cap" in a failed run | `epochs` worked out longer than `max_iters` | Fewer epochs or an explicit `iters`. |
| Model type "is not supported for training" | The architecture has no training support | Choose another model. |
| Dataset "has no non-empty split" | Wrong split name or an empty file | Set `train_split`, or fix the data. |
| "Feature type ... not found" loading a dataset | The `datasets` library is older than the dataset's format | Try the same id as `data`, or upgrade `datasets`. |
| A missing column or key error at the start | Rows do not match the algorithm's format | See [datasets](datasets.md); map columns with `dataset_config`. |
| A row "would lose its image tokens" | `max_seq_length` is shorter than a prompt with images | Raise `max_seq_length`. |
| "No complete batches fit the dataset" | `batch_size` is larger than the rows available | Lower `batch_size`. |
| Out of memory, or the process is killed with no error | The run needs more memory than is free | Follow the order in [memory](memory.md). |
| Loss is `nan` or jumps | `learning_rate` is too high | Lower it and set `grad_clip: 1.0`. |
| Loss does not move | `learning_rate` is too low, or nothing is being learned | Raise it; check `train_on_completions` and the row format. |
| `grpo` reward stays at zero | Outputs never match the answer format | Read `sample`; adjust the prompt, the answers, or the reward. |
| A module is missing | An optional dependency is not installed | `pip install "mlx-vlm[train]"`. |
| `continue` says the run "saved no weights" | The run ended before its first save | Start again; use a smaller `steps_per_save` next time. |
| `continue` reports a state mismatch | The options changed since the run was stopped | Continue with the original options. |
| `stopped` but nothing to continue | The run finished its iterations before the stop reached it | It is complete; nothing is left to train. |

When the cause is not in this table, quote the last error line from the log and say what you would check next. Do not guess a cause and present it as fact.
