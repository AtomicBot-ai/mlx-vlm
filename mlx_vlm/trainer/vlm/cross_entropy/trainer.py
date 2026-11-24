# Copyright © 2026 MLX-VLM

import logging
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ...common import engine
from ...common.config import CoreTrainingArgs
from ...common.task import TrainingTask
from ...common.utils import batch_shuffle_rng, grad_checkpoint
from ...core import TrainingArgs, save_adapter
from ...losses.cross_entropy import cross_entropy
from ..config import SFTTrainConfig
from .runtime import (
    _image_token_id,
    _squeeze_batch,
    model_uses_padding_mask,
    vlm_batch_metrics,
    vlm_vision_mask,
)


def _collate_arrays(values):
    """Stack same-shaped arrays, or concatenate variable-length feature rows."""
    try:
        return mx.stack(values)
    except ValueError:
        first = values[0]
        if (
            isinstance(first, mx.array)
            and first.ndim > 1
            and all(
                isinstance(v, mx.array)
                and v.ndim == first.ndim
                and v.shape[1:] == first.shape[1:]
                for v in values
            )
        ):
            return mx.concatenate(values, axis=0)
        raise


def _collate_grid_thw(values):
    """Pack per-example image/video grids as (total_media, 3)."""
    rows = []
    for value in values:
        value = value if isinstance(value, mx.array) else mx.array(value)
        if value.ndim == 1:
            value = value[None, :]
        if value.ndim != 2 or value.shape[1] != 3:
            raise ValueError(
                f"Expected grid_thw values with shape (num_media, 3), got {value.shape}"
            )
        rows.append(value)
    return mx.concatenate(rows, axis=0)


def _resolve_adapter_file(args: TrainingArgs) -> Path:
    adapter_file = getattr(args, "adapter_file", None)
    if adapter_file:
        return Path(adapter_file)

    adapter_path = getattr(args, "adapter_path", None)
    if adapter_path:
        return Path(adapter_path) / "adapters.safetensors"

    return Path(TrainingArgs.__dataclass_fields__["adapter_file"].default)


def vision_language_loss_fn(
    model, batch, train_on_completions=False, assistant_id=77091, mtp_loss_weight=0.0
):
    """Keep historical logit alignment and completion masking for prepared batches."""
    if mtp_loss_weight:
        from .runtime import vlm_sft_loss

        return vlm_sft_loss(
            model,
            batch,
            train_on_completions=train_on_completions,
            assistant_id=assistant_id,
            mtp_loss_weight=mtp_loss_weight,
        )
    pixel_values = batch["pixel_values"]
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]

    batch_size, seq_length = input_ids.shape

    input_ids = input_ids[:, :-1]
    attention_mask = attention_mask[:, :-1]

    lengths = mx.sum(attention_mask, axis=1)

    labels = batch["input_ids"][:, 1:]

    kwargs = {
        k: v
        for k, v in batch.items()
        if k not in ["input_ids", "pixel_values", "attention_mask", "completion_mask"]
    }

    model_attention_mask = attention_mask if model_uses_padding_mask(model) else None
    outputs = model(input_ids, pixel_values, model_attention_mask, **kwargs)
    logits = outputs.logits.astype(mx.float32)

    def align_logits_with_labels(logits, labels):
        if logits.shape[1] < labels.shape[1]:
            pad_length = labels.shape[1] - logits.shape[1]
            pad_width = ((0, 0), (0, pad_length), (0, 0))
            return mx.pad(logits, pad_width, mode="constant", constant_values=-100)
        elif logits.shape[1] > labels.shape[1]:
            return logits[:, -labels.shape[1] :, :]
        return logits

    logits = align_logits_with_labels(logits, labels)

    seq_len = input_ids.shape[1]
    lengths = mx.minimum(lengths, seq_len)
    length_mask = mx.arange(seq_len)[None, :] < lengths[:, None]
    loss_mask = length_mask

    if train_on_completions:
        if "completion_mask" in batch:
            loss_mask = loss_mask * batch["completion_mask"][:, 1:]
        else:
            completion_mask = mx.ones_like(batch["attention_mask"])

            assistant_response_index = np.full((batch_size,), -1, dtype=np.int32)
            input_ids_np = np.array(batch["input_ids"])
            for row_idx, row in enumerate(input_ids_np):
                positions = np.where(row == assistant_id)[0]
                if positions.size > 0:
                    assistant_response_index[row_idx] = positions[0]

            range_matrix = mx.repeat(
                mx.expand_dims(mx.arange(seq_length), 0), batch_size, axis=0
            )
            assistant_mask = range_matrix <= mx.array(assistant_response_index).reshape(
                -1, 1
            )
            completion_mask = mx.where(
                assistant_mask,
                mx.zeros_like(completion_mask),
                completion_mask,
            )
            loss_mask = loss_mask * completion_mask[:, 1:]

    loss, metrics = cross_entropy(
        logits, labels, loss_mask, vision_mask=vlm_vision_mask(model, batch)
    )
    return loss, {**metrics, **vlm_batch_metrics(batch)}


def _truncates_image_tokens(item, image_token_id, max_seq_length):
    ids = np.asarray(item["input_ids"]).reshape(-1)
    return np.any(ids[max_seq_length:] == image_token_id)


def iterate_batches(dataset, batch_size, max_seq_length, train=False, seed=None):
    indices = range(len(dataset))
    if len(dataset) < batch_size:
        raise ValueError(f"Dataset must have at least {batch_size} examples")

    world = mx.distributed.init()
    rank, world_size = world.rank(), world.size()
    if batch_size % world_size != 0:
        raise ValueError("Batch size must be divisible by number of workers")

    rng = batch_shuffle_rng(seed, world_size)
    image_token_id = _image_token_id(dataset)
    warned_indices = set()
    if world_size > 1 and image_token_id is not None:
        # Filter before sharding. Skipping an empty local batch independently
        # would desynchronize both the optimizer updates and the collectives.
        dropped = np.zeros(len(dataset), dtype=np.int32)
        for idx in range(rank, len(dataset), world_size):
            if _truncates_image_tokens(dataset[idx], image_token_id, max_seq_length):
                dropped[idx] = 1
        dropped = mx.distributed.all_sum(mx.array(dropped), stream=mx.cpu)
        indices = np.flatnonzero(np.asarray(dropped) == 0).tolist()
        if len(indices) < batch_size:
            raise ValueError(
                "No complete global batch remains after filtering truncated image placeholders."
            )
        if rank == 0 and len(indices) < len(dataset):
            logging.warning(
                "Skipping %d examples with truncated image placeholders",
                len(dataset) - len(indices),
            )

    batch_indices = [
        indices[i + rank : i + batch_size : world_size]
        for i in range(0, len(indices) - batch_size + 1, batch_size)
    ]
    while True:
        order = (
            rng.permutation(len(batch_indices)) if train else range(len(batch_indices))
        )
        yielded_any = False
        for b in order:
            items = [dataset[idx] for idx in batch_indices[b]]

            # Drop examples whose image placeholders would not survive truncation:
            # pixel_values keeps every sub-image, so training on them would fail the
            # model's feature/token alignment check.
            if image_token_id is not None and world_size == 1:
                kept = []
                for idx, item in zip(batch_indices[b], items):
                    if _truncates_image_tokens(item, image_token_id, max_seq_length):
                        if idx not in warned_indices:
                            warned_indices.add(idx)
                            logging.warning(
                                "Skipping example %d: its image tokens do not fit in "
                                "max_seq_length=%d. Raise max_seq_length or lower the image resolution.",
                                idx,
                                max_seq_length,
                            )
                        continue
                    kept.append(item)
                if not kept:
                    # Every example here was skipped; move on rather than failing
                    # the run, since other batches may still be trainable.
                    continue
                items = kept

            lengths = [
                min(np.asarray(x["input_ids"]).size, max_seq_length) for x in items
            ]

            max_len = min(max(lengths), max_seq_length)
            pad_to = 32
            padded_len = 1 + pad_to * ((max_len + pad_to - 1) // pad_to)
            padded_len = min(padded_len, max_seq_length)

            input_ids_batch = np.zeros((len(items), padded_len), dtype=np.int32)
            attention_mask_batch = np.zeros((len(items), padded_len), dtype=np.int32)
            has_completion_mask = any("completion_mask" in item for item in items)
            completion_mask_batch = np.zeros((len(items), padded_len), dtype=np.int32)

            for i, item in enumerate(items):
                arr = np.array(_squeeze_batch(item["input_ids"])).reshape(-1)
                L = min(len(arr), padded_len)
                input_ids_batch[i, :L] = arr[:L]

                if "attention_mask" in item:
                    mask = np.array(_squeeze_batch(item["attention_mask"])).reshape(-1)
                    attention_mask_batch[i, :L] = mask[:L]
                else:
                    attention_mask_batch[i, :L] = 1

                if "completion_mask" in item:
                    completion_mask = np.array(
                        _squeeze_batch(item["completion_mask"])
                    ).reshape(-1)
                    completion_mask_batch[i, :L] = completion_mask[:L]

            pixel_values_batch = None
            if "pixel_values" in items[0] and items[0]["pixel_values"] is not None:
                pixel_values_batch = _collate_arrays(
                    [_squeeze_batch(item["pixel_values"]) for item in items]
                )

            batch = {
                "input_ids": mx.array(input_ids_batch),
                "attention_mask": mx.array(attention_mask_batch),
                "pixel_values": pixel_values_batch,
            }
            if has_completion_mask:
                batch["completion_mask"] = mx.array(completion_mask_batch)

            extra_keys = [
                k
                for k in items[0]
                if k
                not in (
                    "input_ids",
                    "attention_mask",
                    "completion_mask",
                    "pixel_values",
                )
            ]
            for k in extra_keys:
                if k in ("image_grid_thw", "video_grid_thw"):
                    batch[k] = _collate_grid_thw([item[k] for item in items])
                    continue

                vals = [_squeeze_batch(item[k]) for item in items]
                if isinstance(vals[0], mx.array):
                    try:
                        batch[k] = _collate_arrays(vals)
                    except Exception:
                        batch[k] = vals[0]
                else:
                    batch[k] = vals[0]

            yielded_any = True
            yield batch

        if not yielded_any:
            raise ValueError(
                "No trainable examples: every example has more image tokens than "
                f"max_seq_length={max_seq_length} allows. Raise max_seq_length or "
                "lower the image resolution."
            )
        if not train:
            break


def _checkpoint_layers(model):
    for module in model.children().values():
        if hasattr(module, "layers"):
            grad_checkpoint(module.layers[0])


def _save_weights(model, path, iteration=None):
    save_adapter(model, path)
    if iteration is not None:
        save_adapter(
            model, Path(path).with_name(f"{iteration:07d}_adapters.safetensors")
        )


def _legacy_task(args, loss_fn, train_on_completions, assistant_id):
    if getattr(args, "mtp_loss_weight", 0.0):
        loss_fn = partial(loss_fn, mtp_loss_weight=args.mtp_loss_weight)
    loss = partial(
        loss_fn, train_on_completions=train_on_completions, assistant_id=assistant_id
    )

    return TrainingTask(
        loss=loss,
        batches=partial(
            iterate_batches,
            batch_size=args.batch_size,
            max_seq_length=args.max_seq_length,
        ),
        checkpoint=_checkpoint_layers,
        save_weights=_save_weights,
        unit="tokens",
    )


def evaluate(
    model,
    dataset,
    batch_size,
    num_batches,
    max_seq_length=2048,
    loss_fn=vision_language_loss_fn,
    train_on_completions=False,
    assistant_id=77091,
    return_metrics=False,
):
    """Evaluate prepared examples with the historical loss and batch format."""
    args = SFTTrainConfig(
        batch_size=batch_size, max_seq_length=max_seq_length, val_batches=num_batches
    )
    task = _legacy_task(args, loss_fn, train_on_completions, assistant_id)
    model.eval()
    result = engine.evaluate(model, dataset, task, args, return_metrics=return_metrics)
    mx.clear_cache()
    return result


def train(
    model,
    optimizer,
    train_dataset,
    val_dataset=None,
    args: TrainingArgs = TrainingArgs(),
    loss_fn=vision_language_loss_fn,
    train_on_completions=False,
    assistant_id=77091,
):
    """Train prepared examples on the shared engine with the legacy SFT API."""
    if mx.metal.is_available():
        limit = mx.device_info().get("max_recommended_working_set_size")
        if limit is not None:
            mx.set_wired_limit(limit)

    task = _legacy_task(args, loss_fn, train_on_completions, assistant_id)
    config = CoreTrainingArgs(
        batch_size=args.batch_size,
        iters=args.iters,
        val_batches=args.val_batches,
        steps_per_report=args.steps_per_report,
        steps_per_eval=args.steps_per_eval,
        steps_per_save=args.steps_per_save,
        adapter_file=str(_resolve_adapter_file(args)),
        grad_checkpoint=args.grad_checkpoint,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        grad_clip=args.grad_clip,
        clear_cache_threshold=1,
    )
    engine.train(
        model,
        optimizer,
        train_dataset,
        val_dataset,
        config,
        task,
        optimizer_update=optimizer.update,
        evaluate_before_step=True,
    )


@dataclass
class SFTTrainer:
    """Configure and run the functional supervised fine-tuning recipe."""

    model: nn.Module
    optimizer: Any
    train_dataset: Any
    val_dataset: Any = None
    args: TrainingArgs = field(default_factory=TrainingArgs)
    train_on_completions: bool = False
    assistant_id: int = 77091

    def fit(self):
        """Run training with the configured model, data, and options."""
        return train(
            model=self.model,
            optimizer=self.optimizer,
            train_dataset=self.train_dataset,
            val_dataset=self.val_dataset,
            args=self.args,
            train_on_completions=self.train_on_completions,
            assistant_id=self.assistant_id,
        )
