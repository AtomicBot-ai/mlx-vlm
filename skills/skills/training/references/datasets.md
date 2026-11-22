# Datasets

## Sources

- `dataset`: a Hugging Face dataset id such as `user/name`, loaded with the `datasets` library. Use `hf_dataset_config` for a named configuration. A private dataset needs the machine to be logged in to Hugging Face.
- `data`: a local folder with `train.jsonl` and optionally `valid.jsonl`, or a single `.jsonl` file, or a Hub repository made of such files. Local paths may be relative; the manager makes them absolute.
- When both are set, `dataset` wins. If `dataset` fails to load a Hub repository made of JSONL files, try the same id as `data`.
- `train_split` and `validation_split` name the splits when they are not `train` and `validation` (or `valid`).

## What `inspect-dataset` reports

For each split: `rows`, `columns`, `format`, `algorithms`, `sampled_rows`, `text_chars` (`median`, `p95`, `max`), and `rows_with_images`. Lengths are counted in characters over an even sample of at most 2,000 rows, so no model is loaded. Columns whose names start with `_` are treated as bookkeeping and left out of the lengths.

| `format` | Columns | Algorithms |
|---|---|---|
| `messages` | `messages` | `sft` |
| `conversations` | `conversations` | `sft` |
| `question_answer` | `question`, `answer` | `sft` |
| `text` | `text` | `uft` |
| `preference` | `chosen`, `rejected` | `orpo`, `dpo` |
| `reward` | `prompt`, `answer` | `grpo` |
| `unknown` | anything else | map the columns with `dataset_config` |

## Row examples

Chat rows hold messages and optional images. Image paths are relative to the dataset folder.

```json
{"images": ["images/red.png"], "messages": [{"role": "user", "content": "Which color?"}, {"role": "assistant", "content": "Red."}]}
```

Preference rows share media and a prompt:

```json
{"images": ["images/red.png"], "prompt": "Which color?", "chosen": "Red.", "rejected": "Blue."}
```

Reward rows hold a prompt and the answer the reward checks:

```json
{"prompt": "What color is the object?", "images": ["images/red.png"], "answer": "red"}
```

Text-only rows are fine: `images` is optional in every format.

## Custom column names

Pass them through `dataset_config`, for example `{"prompt_feature": "question", "chosen_feature": "good", "rejected_feature": "bad"}`. Read the option's help in `describe` for the keys.

## Checks worth reporting

- No validation split: `val_loss` cannot be reported, so overfitting stays invisible. Suggest holding out 5 to 10% of the rows.
- Fewer than about 100 rows: enough to shift style, rarely enough to teach a task. Say so.
- `text_chars.max` far above `p95`: a few very long rows. They are cut at `max_seq_length`; raising the length for them costs memory on every row of their batch.
- `format` of `unknown`: the trainer will not find its columns. Fix the rows or map them before starting.
- Extra fields inside messages, such as `reasoning_content`, are kept for the model's chat template.
