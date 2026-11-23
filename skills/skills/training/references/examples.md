# Worked examples

Complete option sets for common requests. Model and dataset ids are placeholders. Pass each object to `check` first and to `start` only when it is valid and the user agreed.

## Chat fine-tune on a few thousand rows

Request: "Fine-tune this model on my chat dataset."

`inspect-dataset` reported the `messages` format, 4,000 training rows, a validation split, and a p95 length of about 2,800 characters, so roughly 800 tokens.

```json
{
  "model": "org/model-4bit",
  "algorithm": "sft",
  "dataset": "user/chat-dataset",
  "train_on_completions": true,
  "epochs": 2,
  "learning_rate": 1e-4,
  "lr_schedule": "cosine_decay",
  "warmup_steps": 100,
  "grad_clip": 1.0,
  "batch_size": 1,
  "gradient_accumulation_steps": 4,
  "max_seq_length": 896,
  "steps_per_report": 10,
  "steps_per_eval": 800,
  "steps_per_save": 1600
}
```

Two epochs of 4,000 rows at batch size 1 are 8,000 iterations and 2,000 optimizer updates; the warmup is 5% of those.

## The pilot for the same run

Twenty iterations to read memory and speed before committing.

```json
{
  "model": "org/model-4bit",
  "algorithm": "sft",
  "dataset": "user/chat-dataset",
  "train_on_completions": true,
  "iters": 20,
  "learning_rate": 1e-4,
  "batch_size": 1,
  "max_seq_length": 896,
  "steps_per_report": 1
}
```

## Image question answering from a local folder

Request: "Train on the images in this folder."

```json
{
  "model": "org/vision-model-4bit",
  "algorithm": "sft",
  "data": "datasets/colors",
  "train_on_completions": true,
  "epochs": 3,
  "learning_rate": 1e-4,
  "grad_clip": 1.0,
  "max_seq_length": 512,
  "steps_per_report": 5
}
```

Leave `train_vision` off unless the images are unlike what the model has seen.

## Domain adaptation on plain text

Request: "Make it write like these documents."

```json
{
  "model": "org/model-4bit",
  "algorithm": "uft",
  "data": "datasets/manuals",
  "epochs": 1,
  "learning_rate": 1e-4,
  "lr_schedule": "linear_schedule",
  "min_learning_rate": 1e-6,
  "max_seq_length": 1024
}
```

## Preference pairs

Request: "I have good and bad answers for each prompt."

```json
{
  "model": "org/model-4bit",
  "algorithm": "orpo",
  "data": "datasets/preferences",
  "epochs": 2,
  "learning_rate": 5e-5,
  "grad_clip": 1.0,
  "max_seq_length": 768
}
```

Use `dpo` with the same options when staying close to the original model matters and memory allows two copies of it.

## Reward-based training

Request: "Reward it for getting the answer right."

```json
{
  "model": "org/model-4bit",
  "algorithm": "grpo",
  "data": "datasets/arithmetic",
  "iters": 200,
  "group_size": 4,
  "max_completion_length": 64,
  "max_seq_length": 512,
  "learning_rate": 1e-5,
  "steps_per_report": 5
}
```

Follow it with `sample` to read the outputs the reward is scoring.

## A standalone model at the end

Add `export_qat_path` to any of the above to fuse the trained weights into a full model:

```json
{
  "model": "org/model-4bit",
  "algorithm": "sft",
  "data": "datasets/colors",
  "train_on_completions": true,
  "epochs": 3,
  "learning_rate": 1e-4,
  "export_qat_path": "models/colors-merged"
}
```

## Asked only for settings

Request: "Give me the best settings for this model and dataset."

Run `limits`, `validate-model`, `inspect-dataset`, and `check`, and reply with the settings table from [settings](settings.md), the run length, the memory estimate, and the pilot offer. Start nothing.
