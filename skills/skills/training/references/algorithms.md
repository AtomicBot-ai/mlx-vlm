# Algorithms

`describe` lists the algorithms that exist and each one's options; follow it over this page. `inspect-dataset` names the algorithms a dataset's rows suit.

| Algorithm | Rows | What it does | Cost |
|---|---|---|---|
| `sft` | chat messages, or question and answer | Teaches the model to produce the answers. | Baseline |
| `uft` | plain text | Learns the whole sequence; adapts a model to a domain or style. | Baseline |
| `orpo` | `prompt`, `chosen`, `rejected` | Learns the chosen answer while pushing away the rejected one. No second model. | About baseline |
| `dpo` | `prompt`, `chosen`, `rejected` | Compares against a frozen copy of the model, which keeps it close to its original behaviour. | Loads the model twice |
| `grpo` | `prompt`, `answer`, and a reward function | The model writes several answers per step and is rewarded for better ones. | Slowest: generates every step |

## Choosing

- Chat or question/answer rows: `sft` with `train_on_completions: true`.
- Raw text with no question and answer structure: `uft`. Do not set `train_on_completions`.
- Pairs of a better and a worse answer: `orpo` first, because it needs no second model. Use `dpo` when staying close to the original model matters and memory allows two copies.
- A task with a checkable answer and no example outputs to imitate: `grpo`.
- If two fit and the request does not decide it, pick the cheaper one and say which other one would also work.

## Options of their own

Read the defaults and help in `describe` for the chosen algorithm.

- `dpo`: `beta` (how strongly preferences are enforced), `loss_type`, `delta`, and `reference_model` (defaults to the model itself).
- `orpo`: `beta` and `eps`.
- `grpo`: `group_size` (answers written per prompt), `max_completion_length` (must be below `max_seq_length`), `temperature`, `top_p`, `top_k`, `reward_functions`, `reward_weights`, and `reward_functions_file` for custom rewards. `describe` lists the built-in reward functions as `reward_functions`.

## Notes on `grpo`

- Each step generates `group_size` answers of up to `max_completion_length` tokens. Halving either roughly halves the step time.
- Keep `max_completion_length` just above the longest expected answer.
- `sample` returns the latest prompt, raw output, expected answer, and reward. Use it to see whether the reward is measuring what was meant.
- A reward that stays at zero means the outputs never match. Check the answer format in `sample` before changing any setting.
