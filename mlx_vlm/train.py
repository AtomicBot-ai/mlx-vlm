"""Thin shared CLI over the notebook Trainer API (python -m mlx_vlm.train)."""

import argparse
import contextlib
import json
import math
import sys
import time
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, fields
from pathlib import Path
from typing import get_args

from .trainer.api import TASKS, Trainer, task_spec
from .trainer.common.schedule import (
    build_schedule,
    resolve_schedule_config,
    total_optimizer_updates,
)


def _read_config(path):
    if path is None:
        return {}
    if path == "-":
        values = json.load(sys.stdin)
    else:
        with Path(path).open(encoding="utf-8") as file:
            if Path(path).suffix == ".json":
                values = json.load(file)
            else:
                import yaml

                values = yaml.safe_load(file) or {}
    if not isinstance(values, dict):
        raise ValueError("Training config must contain a mapping.")
    return values


def _parse_schedule(value):
    if not isinstance(value, str):
        return value
    value = value.strip()
    return json.loads(value) if value.startswith("{") else value


# Schedules that work from their name alone, offered as choices by ``describe``.
NAMED_SCHEDULES = ("cosine_decay", "linear_schedule")


def _named_schedule(schedule, learning_rate, min_learning_rate, updates, warmup):
    """Give a bare ``linear_schedule`` its arguments; pass anything else on."""
    if schedule != "linear_schedule":
        return schedule
    return {
        "name": schedule,
        "arguments": [learning_rate, min_learning_rate, max(updates - warmup, 1)],
    }


def registered_tasks():
    """Map each task to its algorithms, straight from the trainer registry."""
    tasks = {}
    for task, algorithm in TASKS:
        tasks.setdefault(task, []).append(algorithm)
    return tasks


def build_parser(task="vlm", algorithm=None):
    """Generate training flags from the same dataclasses notebooks use."""
    algorithm, (_, config_type) = task_spec(task, algorithm)
    tasks = registered_tasks()
    parser = argparse.ArgumentParser(description=f"MLX {task}/{algorithm} training")
    parser.add_argument("--task", choices=tuple(tasks), default=task)
    parser.add_argument(
        "--algorithm",
        "--train-mode",
        choices=tuple(
            dict.fromkeys(name for names in tasks.values() for name in names)
        ),
        default=algorithm,
    )
    parser.add_argument(
        "--describe",
        action="store_true",
        help="Print every task, algorithm and option as JSON, then exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the options, print the result as JSON, then exit.",
    )
    parser.add_argument(
        "--progress-file",
        help="Append training events to this file as JSON lines.",
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        help="Usage cap: refuse runs longer than this many iterations.",
    )
    parser.add_argument(
        "--max-minutes",
        type=float,
        help="Usage cap: save and stop after this many minutes of wall time.",
    )
    parser.add_argument(
        "--max-memory-gb",
        type=float,
        help="Usage cap: save and stop when MLX memory use passes this size.",
    )
    parser.add_argument(
        "-c", "--config", help="JSON or YAML config file. Use '-' for JSON on stdin."
    )
    parser.add_argument("--model", "--model-path", help="Model path or Hub identifier.")
    parser.add_argument(
        "--mtp-model",
        help="Matching floating-point Qwen3.5 native checkpoint or split MTP head.",
    )
    parser.add_argument(
        "--data",
        default="data",
        help="Local folder of <split>.jsonl files or one .jsonl file (also a Hub "
        "repo of JSONL files). Used when --dataset is not set.",
    )
    parser.add_argument(
        "--dataset",
        help="Hugging Face dataset ID, such as user/name; overrides --data.",
    )
    parser.add_argument(
        "--hf-dataset-config", help="Optional Hugging Face dataset configuration name."
    )
    parser.add_argument("--train-split", "--split", default="train")
    parser.add_argument("--validation-split", default="validation")
    parser.add_argument(
        "--epochs", type=int, help="Complete batch passes; overrides default iters."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--optimizer", choices=("adam", "adamw", "sgd", "muon"), default="adamw"
    )
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--beta1", type=float, default=0.9, help="Adam/AdamW beta1.")
    parser.add_argument("--beta2", type=float, default=0.999, help="Adam/AdamW beta2.")
    parser.add_argument(
        "--optimizer-eps", type=float, default=1e-8, help="Adam/AdamW epsilon."
    )
    parser.add_argument(
        "--bias-correction",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Adam/AdamW bias correction.",
    )
    parser.add_argument(
        "--lr-schedule",
        type=_parse_schedule,
        help="cosine_decay or linear_schedule go from learning_rate to "
        "min_learning_rate over the run. A JSON/YAML object with name, "
        "arguments, warmup, warmup_init gives full control.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=0,
        help="Linear warmup optimizer updates (applies to --lr-schedule).",
    )
    parser.add_argument("--warmup-init", type=float, default=0.0)
    parser.add_argument(
        "--min-learning-rate",
        type=float,
        default=0.0,
        help="Final rate for a bare cosine_decay or linear_schedule.",
    )
    parser.add_argument(
        "--train-type", choices=("lora", "dora", "full"), default="lora"
    )
    parser.add_argument(
        "--full-finetune", dest="train_type", action="store_const", const="full"
    )
    parser.add_argument(
        "--train-vision",
        action="store_true",
        help="Also unfreeze the vision tower/projector.",
    )
    parser.add_argument(
        "--qat",
        action="store_true",
        help="Weight-only affine QAT for --train-type full (defaults to 4 bits).",
    )
    parser.add_argument(
        "--export-model-path",
        "--export-qat-path",
        dest="export_qat_path",
        help="Export a standalone inference model; detects QAT automatically.",
    )
    parser.add_argument(
        "--export-quantize",
        action="store_true",
        default=None,
        help="Also quantize compatible non-QAT layers during model export.",
    )
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument(
        "--target-modules",
        nargs="+",
        help="Linear suffixes or qualified module names; auto-selects language transformer when omitted.",
    )
    parser.add_argument(
        "--quantization-bits",
        type=int,
        choices=(4, 8),
        help="Optionally quantize the base before adapter training.",
    )
    parser.add_argument(
        "--quantization-group-size",
        type=int,
        choices=(32, 64, 128),
        default=None,
        help="Quantization group size (saved QAT setting on resume, otherwise 64).",
    )
    parser.add_argument(
        "--resume-adapter-file",
        "--adapter-path",
        help="Load weights; add --resume-state to restore the optimizer too.",
    )
    parser.add_argument(
        "--resume-state",
        help="trainer_state.safetensors written when a run was stopped. Restores "
        "the optimizer, random state and position, and trains the iterations "
        "left of the same --iters/--epochs.",
    )
    parser.add_argument(
        "--output-path", help="Output file or folder; alias for --adapter-file."
    )
    parser.add_argument(
        "--reference-model", help="Frozen DPO/GRPO reference; defaults to --model."
    )
    if algorithm == "grpo":
        parser.add_argument(
            "--reward-functions",
            default="accuracy_reward",
            help="Comma-separated batched reward callback names.",
        )
        parser.add_argument(
            "--reward-functions-file",
            help="Python file defining custom reward callbacks.",
        )
        parser.add_argument(
            "--reward-weights", type=json.loads, help="JSON array of reward weights."
        )
    parser.add_argument("--wandb", help="Optional W&B project.")
    parser.add_argument(
        "--dataset-config",
        type=json.loads,
        default={},
        help="Preprocessing/column options as a JSON object.",
    )
    # Value types of the generated flags, kept for --describe.
    parser.training_options = {}
    for item in fields(config_type):
        kwargs = {"default": item.default, "help": item.metadata.get("help")}
        parser.training_options[item.name] = getattr(item.type, "__name__", "str")
        if item.type is bool:
            kwargs["action"] = argparse.BooleanOptionalAction
        else:
            types = get_args(item.type)
            if type(None) in types:
                converter = next(t for t in types if t is not type(None))
                parser.training_options[item.name] = converter.__name__
                kwargs["type"] = lambda value, converter=converter: (
                    None if value.lower() == "none" else converter(value)
                )
            else:
                kwargs["type"] = item.type
        parser.add_argument("--" + item.name.replace("_", "-"), **kwargs)
    return parser


def parse_args(argv=None, *, task="vlm"):
    # Select the dataclass before full parsing, including task/algorithm in a config.
    probe = argparse.ArgumentParser(add_help=False)
    for name in ("task", "algorithm", "config"):
        probe.add_argument("--" + name)
    probe.add_argument("-c", dest="config")
    probe.add_argument("--train-mode", dest="algorithm")
    probe.add_argument("--iters", type=int)
    early, _ = probe.parse_known_args(argv)
    config = _read_config(early.config)
    task = early.task or config.get("task", task)
    algorithm = early.algorithm or config.get("algorithm")
    try:
        parser = build_parser(task, algorithm)
    except ValueError as error:
        probe.error(str(error))
    unknown = set(config) - {action.dest for action in parser._actions}
    if unknown:
        parser.error("Unknown config options: " + ", ".join(sorted(unknown)))
    parser.set_defaults(**config)
    args = parser.parse_args(argv)
    if args.export_quantize and not args.export_qat_path:
        parser.error("--export-quantize requires --export-model-path.")
    if not args.model:
        parser.error("--model is required.")
    if args.output_path:
        destination = Path(args.output_path)
        args.adapter_file = str(
            destination
            if destination.suffix == ".safetensors"
            else destination / "adapters.safetensors"
        )
    if not isinstance(args.dataset_config, dict):
        parser.error("--dataset-config must be a JSON object.")
    for name in ("max_iters", "max_minutes", "max_memory_gb"):
        value = getattr(args, name)
        if value is not None and not (math.isfinite(value) and value > 0):
            parser.error(f"{name} must be positive.")
    if args.reference_model and args.algorithm not in {"dpo", "grpo"}:
        parser.error("--reference-model is only used by DPO or GRPO.")
    if args.optimizer not in {"adam", "adamw", "sgd", "muon"}:
        parser.error("optimizer must be adam, adamw, sgd, or muon.")
    if (
        not math.isfinite(args.learning_rate)
        or args.learning_rate <= 0
        or not math.isfinite(args.weight_decay)
        or args.weight_decay < 0
    ):
        parser.error(
            "learning_rate must be positive and weight_decay nonnegative and finite."
        )
    if not (math.isfinite(args.optimizer_eps) and args.optimizer_eps > 0):
        parser.error("optimizer_eps must be positive and finite.")
    for name in ("beta1", "beta2"):
        if not 0 <= getattr(args, name) < 1:
            parser.error(f"{name} must be in [0, 1).")
    if (
        args.warmup_steps < 0
        or not math.isfinite(args.warmup_init)
        or args.warmup_init < 0
    ):
        parser.error("warmup_steps and warmup_init must be nonnegative.")
    if not math.isfinite(args.min_learning_rate) or args.min_learning_rate < 0:
        parser.error("min_learning_rate must be nonnegative and finite.")
    if args.optimizer not in {"adam", "adamw"} and (
        (args.beta1, args.beta2, args.optimizer_eps, args.bias_correction)
        != (0.9, 0.999, 1e-8, False)
    ):
        parser.error(
            "beta1, beta2, optimizer_eps and bias_correction need adam or adamw."
        )
    try:
        schedule = resolve_schedule_config(
            _named_schedule(
                args.lr_schedule,
                args.learning_rate,
                args.min_learning_rate,
                10**9,
                args.warmup_steps,
            ),
            learning_rate=args.learning_rate,
            total_updates=10**9,
            warmup_steps=args.warmup_steps,
            warmup_init=args.warmup_init,
            min_learning_rate=args.min_learning_rate,
        )
        if schedule is not None:
            build_schedule(schedule)
    except (ValueError, TypeError) as error:
        parser.error(str(error))
    _, (_, config_type) = task_spec(args.task, args.algorithm)
    settings = {item.name: getattr(args, item.name) for item in fields(config_type)}
    if args.epochs is not None:
        if args.epochs < 1:
            parser.error("epochs must be at least 1.")
        if early.iters is not None or config.get("iters") is not None:
            parser.error("Set either iters or epochs, not both.")
    try:
        config_type(**settings).validate()
    except (ValueError, TypeError) as error:
        parser.error(str(error))
    if args.epochs is None and args.max_iters and settings["iters"] > args.max_iters:
        parser.error(f"iters {settings['iters']} is over the cap of {args.max_iters}.")
    return args, config_type(**settings)


# Set by a run manager or the operator rather than chosen per run.
MANAGED_OPTIONS = {
    "progress_file",
    "max_iters",
    "max_minutes",
    "max_memory_gb",
    "resume_state",
}
# Options only some optimizers read. ``describe`` reports them as ``requires``
# so a host can hide an option the chosen optimizer ignores.
SCHEDULE_OPTIONS = {"min_learning_rate": NAMED_SCHEDULES}
OPTIMIZER_OPTIONS = {
    "beta1": ("adam", "adamw"),
    "beta2": ("adam", "adamw"),
    "optimizer_eps": ("adam", "adamw"),
    "bias_correction": ("adam", "adamw"),
    "weight_decay": ("adamw", "muon"),
}


def _option_type(action, parser):
    if action.dest in parser.training_options:
        return parser.training_options[action.dest]
    if action.nargs == 0:
        return "bool"
    if action.type in (int, float):
        return action.type.__name__
    if action.type is json.loads:
        return "json"
    return "list" if action.nargs == "+" else "str"


def _reward_functions(task, algorithm):
    """Names of built-in reward callbacks, when the algorithm ships any."""
    from importlib import import_module

    try:
        module = import_module(f"mlx_vlm.trainer.{task}.{algorithm}.rewards")
    except ImportError:
        return []
    return sorted(
        name
        for name, value in vars(module).items()
        if name.endswith("_reward") and callable(value)
    )


def describe():
    """Describe every registered task/algorithm and its options.

    Everything is read from the trainer registry and the config dataclasses,
    so a new algorithm or modality appears here without editing this file.
    """
    from .version import __version__

    internal = {"help", "describe", "dry_run", "task", "algorithm", "config"}
    tasks = {}
    for (task, algorithm), (_, config_type) in TASKS.items():
        parser = build_parser(task, algorithm)
        options = []
        for action in parser._actions:
            if action.dest in internal or action.const == "full":
                continue
            default = action.default
            options.append(
                {
                    "name": action.dest,
                    "flag": action.option_strings[0],
                    "type": _option_type(action, parser),
                    "default": default if default is not argparse.SUPPRESS else None,
                    "choices": (
                        list(NAMED_SCHEDULES)
                        if action.dest == "lr_schedule"
                        else list(action.choices) if action.choices else None
                    ),
                    "help": action.help,
                    "requires": (
                        {"optimizer": list(OPTIMIZER_OPTIONS[action.dest])}
                        if action.dest in OPTIMIZER_OPTIONS
                        else (
                            {"lr_schedule": list(SCHEDULE_OPTIONS[action.dest])}
                            if action.dest in SCHEDULE_OPTIONS
                            else None
                        )
                    ),
                    "group": (
                        "training"
                        if action.dest in parser.training_options
                        else "managed" if action.dest in MANAGED_OPTIONS else "general"
                    ),
                }
            )
        doc = (config_type.__doc__ or "").strip().split("\n")[0]
        entry = {
            # Dataclasses without a docstring get a generated signature; skip it.
            "description": "" if doc.startswith(config_type.__name__ + "(") else doc,
            "config": config_type.__name__,
            "options": options,
        }
        rewards = _reward_functions(task, algorithm)
        if rewards:
            entry["reward_functions"] = rewards
        tasks.setdefault(task, {"algorithms": {}})["algorithms"][algorithm] = entry
    for task, entry in tasks.items():
        try:
            # What the CLI trains when no algorithm is named.
            entry["default_algorithm"] = task_spec(task)[0]
        except ValueError:
            entry["default_algorithm"] = next(iter(entry["algorithms"]))
    return {"version": __version__, "tasks": tasks}


class UsageCapReached(Exception):
    """Raised from the progress callback to end training at a usage cap."""


class StopRequested(Exception):
    """Raised from the progress callback when the run was asked to stop."""


STOP_FILE = "stop"  # beside the progress file; its presence asks a run to stop
STATE_FILE = "trainer_state.safetensors"  # beside the saved weights
SAMPLE_FILE = "last_generation.json"  # beside the progress file
SAMPLE_CHARS = 4000


def _clip(value):
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text if len(text) <= SAMPLE_CHARS else text[:SAMPLE_CHARS] + " …"


def record_samples(reward, path):
    """Wrap a reward callback so the latest generation is kept in one file.

    Only the first completion of the latest group is written, replacing the
    previous one, so nothing accumulates in memory or on disk.
    """
    from functools import wraps

    path, state = Path(path), {"calls": 0}

    @wraps(reward)
    def wrapped(*args, **kwargs):
        scores = reward(*args, **kwargs)
        state["calls"] += 1
        with contextlib.suppress(Exception):  # never let this break training
            first = lambda key: (kwargs.get(key) or [None])[0]  # noqa: E731
            record = {
                "time": time.time(),
                "group": state["calls"],
                "prompt": _clip(first("prompts")),
                "completion": _clip(first("completions")),
                "answer": _clip(first("answer")),
                "reward": None if not scores or scores[0] is None else float(scores[0]),
                "reward_function": getattr(reward, "__name__", None),
            }
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(record))
            tmp.replace(path)
        return scores

    return wrapped


class ProgressCallback:
    """Write JSON-lines progress and enforce the wall-time and memory caps.

    Caps are checked at each training report, so a run ends within
    ``steps_per_report`` iterations of passing one.
    """

    train_report_on_all_ranks = True

    def __init__(self, path=None, wrapped=None, max_minutes=None, max_memory_gb=None):
        import mlx.core as mx

        self.world = mx.distributed.init()
        self.path = path if self.world.rank() == 0 else None
        self.wrapped = wrapped
        self.max_minutes, self.max_memory_gb = max_minutes, max_memory_gb
        self.started = time.monotonic()
        # Set when a run continues another: iterations and optimizer updates
        # already done, and the iterations of the whole run.
        self.done, self.updates_done, self.total = 0, 0, None
        self.last = {}  # the latest training report, as the trainer gave it
        if self.path:
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)

    def write(self, event, **info):
        if not self.path:
            return
        record = {"event": event, "time": time.time(), **info}
        try:
            line = json.dumps(record, default=float, allow_nan=False)
        except ValueError:
            # Divergent metrics (including scalar arrays) become JSON null.
            record = json.loads(
                json.dumps(record, default=float), parse_constant=lambda _: None
            )
            line = json.dumps(record, allow_nan=False)
        with open(self.path, "a", encoding="utf-8") as file:
            file.write(line + "\n")

    def whole_run(self, info):
        """Report positions within the whole run, not just this part of it."""
        if not self.done:
            return info
        info = dict(info)
        if "iteration" in info:
            info["iteration"] += self.done
            if self.total and "progress" in info:
                info["progress"] = info["iteration"] / self.total
        if "optimizer_step" in info:
            info["optimizer_step"] += self.updates_done
        return info

    def on_train_loss_report(self, train_info):
        import mlx.core as mx

        memory = mx.get_active_memory() / 1e9
        self.last = dict(train_info)
        if self.world.rank() == 0:
            self.write(
                "train", **self.whole_run(train_info), active_memory_gb=round(memory, 3)
            )
            if self.wrapped is not None:
                self.wrapped.on_train_loss_report(train_info)
        if self.total is not None and train_info["iteration"] + self.done == self.total:
            return  # Finish validation, the final checkpoint and any requested export.
        stopped = bool(self.path and (Path(self.path).parent / STOP_FILE).exists())
        minutes = (time.monotonic() - self.started) / 60
        time_cap = (
            self.max_minutes if self.max_minutes and minutes >= self.max_minutes else 0
        )
        memory_cap = (
            self.max_memory_gb
            if self.max_memory_gb and memory > self.max_memory_gb
            else 0
        )
        if self.world.size() > 1:
            # Every rank must leave at the same report, before the next
            # validation or gradient collective. A cap can trip on any rank.
            with mx.stream(mx.cpu):
                stopped, time_cap, memory_cap, memory = mx.distributed.all_max(
                    mx.array([float(stopped), time_cap, memory_cap, memory]),
                    group=self.world,
                ).tolist()
        if stopped:
            raise StopRequested("stop requested")
        if time_cap:
            raise UsageCapReached(f"max_minutes ({time_cap:g}) reached")
        if memory_cap:
            raise UsageCapReached(
                f"max_memory_gb ({memory_cap:g}) passed: {memory:.2f} GB in use"
            )

    def on_val_loss_report(self, val_info):
        self.write("validation", **self.whole_run(val_info))
        if self.wrapped is not None:
            self.wrapped.on_val_loss_report(val_info)


def save_trainer_state(trainer, iteration, optimizer_step):
    """Save what a later run needs to carry on exactly where this one ended."""
    import mlx.core as mx
    from mlx.utils import tree_flatten

    arrays = {"optimizer." + k: v for k, v in tree_flatten(trainer.optimizer.state)}
    arrays["random.key"] = mx.random.state[0]
    accumulation = trainer.accumulation
    metadata = {"iteration": str(iteration), "optimizer_step": str(optimizer_step)}
    if accumulation is not None:
        metadata["pending_microsteps"] = str(accumulation.microsteps)
        if accumulation.microsteps:
            metadata["gradient_accumulation_steps"] = str(
                trainer.args.gradient_accumulation_steps
            )
            arrays.update(
                ("gradient." + key, grad)
                for key, grad in tree_flatten(accumulation.gradients)
            )
    path = Path(trainer.args.adapter_file).parent / STATE_FILE
    mx.save_safetensors(
        str(path),
        arrays,
        metadata=metadata,
    )
    return path


def restore_trainer_state(trainer, path, progress=None):
    """Continue a stopped run: same optimizer, random state and data position.

    ``trainer.args.iters`` still counts the whole run here, so the schedule
    was built as before; only the iterations left are trained. Batches the
    earlier run already used are drawn and discarded to reach its position.
    """
    import mlx.core as mx
    from mlx.utils import tree_flatten, tree_map_with_path

    from .trainer.common.engine import GradientAccumulation

    arrays, meta = mx.load(str(path), return_metadata=True)
    done, updates = int(meta["iteration"]), int(meta["optimizer_step"])
    total = trainer.args.iters
    if done >= total:
        raise ValueError(f"The run already trained its {total} iterations.")
    pending = int(meta.get("pending_microsteps", 0))
    window = trainer.args.gradient_accumulation_steps
    gradients = {
        k.removeprefix("gradient."): v
        for k, v in arrays.items()
        if k.startswith("gradient.")
    }
    if not 0 <= pending < window or bool(pending) != bool(gradients):
        raise ValueError("Invalid pending gradient accumulation in trainer state.")
    trainer.accumulation = GradientAccumulation()
    if pending:
        if int(meta["gradient_accumulation_steps"]) != window:
            raise ValueError(
                "Pending gradients require the original gradient_accumulation_steps."
            )
        parameters = trainer.model.trainable_parameters()
        if {k: v.shape for k, v in tree_flatten(parameters)} != {
            k: v.shape for k, v in gradients.items()
        }:
            raise ValueError("Saved gradients do not match the trainable weights.")
        trainer.accumulation = GradientAccumulation(
            tree_map_with_path(lambda key, _: gradients[key], parameters), pending
        )
    random_key = arrays.get("random.key")
    saved = {k[10:]: v for k, v in arrays.items() if k[:10] == "optimizer."}
    optimizer, init = trainer.optimizer, trainer.optimizer.init

    def init_and_restore(parameters):
        init(parameters)
        # Write each saved value over the fresh one at the same place, so the
        # state keeps the shape the optimizer built for these weights.
        if set(saved) != {key for key, _ in tree_flatten(optimizer.state)}:
            raise ValueError(
                "The saved optimizer state does not match this run's optimizer "
                "or trainable weights."
            )
        for key, value in saved.items():
            *parents, leaf = key.split(".")
            node = optimizer.state
            for part in parents:
                node = node[int(part)] if isinstance(node, list) else node[part]
            if isinstance(node, list):
                node[int(leaf)] = value
            else:
                node[leaf] = value
        if random_key is not None:
            mx.random.state[0][:] = random_key

    optimizer.init = init_and_restore
    batches = trainer.task.batches

    def later_batches(*args, **kwargs):
        iterator = iter(batches(*args, **kwargs))
        if kwargs.get("train"):
            for _ in range(done):
                next(iterator)
            if random_key is not None:  # drawing batches must not move it
                mx.random.state[0][:] = random_key
        return iterator

    trainer.task.batches = later_batches
    trainer.args.iters = total - done
    if progress is not None:
        progress.done, progress.updates_done, progress.total = done, updates, total
    return done, total


def _load_model(task, path):
    """Keep upstream loaders here so future mlx-omni integration changes one place."""
    from mlx_vlm import load

    return load(path)


def _configure_model(model, args):
    from .trainer.common.model import prepare_model_for_training

    model = prepare_model_for_training(
        model,
        train_type=args.train_type,
        qat=getattr(args, "qat", False),
        lora_rank=getattr(args, "lora_rank", 8),
        lora_alpha=getattr(args, "lora_alpha", 16),
        lora_dropout=getattr(args, "lora_dropout", 0),
        target_modules=getattr(args, "target_modules", None),
        quantization_bits=getattr(args, "quantization_bits", None),
        quantization_group_size=getattr(args, "quantization_group_size", None),
        checkpoint_path=getattr(args, "resume_adapter_file", None),
    )
    if getattr(args, "train_vision", False):
        from .trainer.peft import unfreeze_modules

        unfreeze_modules(
            model,
            [
                "vision_model",
                "vision_tower",
                "mm_projector",
                "multi_modal_projector",
                "aligner",
                "connector",
                "vision_resampler",
            ],
        )
    return model


def _load_dataset(args):
    if args.dataset:
        from datasets import load_dataset

        return load_dataset(args.dataset, args.hf_dataset_config)
    from .trainer.datasets.loading import load

    return load(args.data)


def _bundles_mtp_head(checkpoint_path):
    """Return whether a checkpoint stores ``mtp_head.*`` beside the main weights."""
    if not checkpoint_path:
        return False
    import mlx.core as mx

    from .trainer.common.model import _read_checkpoint

    path, _ = _read_checkpoint(checkpoint_path, require_config=False)
    return any(key.startswith("mtp_head.") for key in mx.load(str(path)))


def build_optimizer(args, iters, accumulation_steps=1):
    """Create the optimizer; schedules count updates, not microsteps."""
    import mlx.optimizers as optim

    optimizer_type = {
        "adam": optim.Adam,
        "adamw": optim.AdamW,
        "sgd": optim.SGD,
        "muon": optim.Muon,
    }[args.optimizer]
    options = {}
    if args.optimizer in OPTIMIZER_OPTIONS["beta1"]:
        options.update(
            betas=[args.beta1, args.beta2],
            eps=args.optimizer_eps,
            bias_correction=args.bias_correction,
        )
    if args.optimizer in OPTIMIZER_OPTIONS["weight_decay"]:
        options["weight_decay"] = args.weight_decay
    learning_rate = args.learning_rate
    updates = total_optimizer_updates(iters, accumulation_steps)
    config = resolve_schedule_config(
        _named_schedule(
            args.lr_schedule,
            args.learning_rate,
            args.min_learning_rate,
            updates,
            args.warmup_steps,
        ),
        learning_rate=args.learning_rate,
        total_updates=updates,
        warmup_steps=args.warmup_steps,
        warmup_init=args.warmup_init,
        min_learning_rate=args.min_learning_rate,
    )
    if config is not None:
        learning_rate = build_schedule(config)
    return optimizer_type(learning_rate=learning_rate, **options)


def run(args, settings, training_callback=None):
    """Load resources and call the same Trainer used in notebooks."""
    import mlx.core as mx

    from mlx_vlm.trainer.common.callbacks import WandBCallback

    mx.random.seed(args.seed)
    primary = mx.distributed.init().rank() == 0
    max_iters = getattr(args, "max_iters", None)
    max_minutes = getattr(args, "max_minutes", None)
    max_memory_gb = getattr(args, "max_memory_gb", None)
    progress = None
    if getattr(args, "progress_file", None) or max_minutes or max_memory_gb:
        progress = training_callback = ProgressCallback(
            getattr(args, "progress_file", None),
            training_callback,
            max_minutes,
            max_memory_gb,
        )
    if max_memory_gb:
        mx.set_memory_limit(int(max_memory_gb * 1e9))
    model, processor = _load_model(args.task, args.model)
    reference = None
    if args.algorithm == "dpo" or (args.algorithm == "grpo" and settings.beta):
        reference, _ = _load_model(args.task, args.reference_model or args.model)
    raw = _load_dataset(args)
    if isinstance(raw, Mapping):
        train_rows = raw.get(args.train_split, [])
        val_rows = raw.get(args.validation_split)
        if val_rows is None and args.validation_split == "validation":
            val_rows = raw.get("valid")
    else:
        train_rows, val_rows = raw, None
    if len(train_rows) == 0:
        raise ValueError(f"Dataset has no non-empty {args.train_split!r} split.")
    head = None
    if settings.mtp_loss_weight:
        if not args.mtp_model:
            raise ValueError("--mtp-loss-weight requires --mtp-model.")
        from .speculative.drafters import load_drafter
        from .trainer.vlm.mtp import attach_mtp_head

        head, _ = load_drafter(args.mtp_model, kind="mtp")
        legacy = _bundles_mtp_head(args.resume_adapter_file)
        if args.train_type == "full" and legacy:
            # Legacy full checkpoints bundle the head and restore strictly.
            attach_mtp_head(model, head)
            head = None
    elif getattr(args, "mtp_model", None):
        raise ValueError("--mtp-model requires a positive --mtp-loss-weight.")
    model = _configure_model(model, args)
    if head is not None:
        attach_mtp_head(model, head, checkpoint_path=args.resume_adapter_file)
    reward_options = {}
    if args.algorithm == "grpo":
        from .trainer.vlm.grpo import rewards

        if args.reward_functions_file:
            import importlib.util

            spec = importlib.util.spec_from_file_location(
                "vlm_custom_rewards", args.reward_functions_file
            )
            rewards = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(rewards)
        reward_funcs = [
            getattr(rewards, name.strip()) for name in args.reward_functions.split(",")
        ]
        if primary and getattr(args, "progress_file", None):
            reward_funcs[0] = record_samples(
                reward_funcs[0], Path(args.progress_file).parent / SAMPLE_FILE
            )
        reward_options = {
            "reward_funcs": reward_funcs,
            "reward_weights": args.reward_weights,
        }
    optimizer = build_optimizer(
        args, settings.iters, settings.gradient_accumulation_steps
    )
    trainer = Trainer(
        model,
        settings,
        optimizer=optimizer,
        task=args.task,
        algorithm=args.algorithm,
        train_dataset=train_rows,
        eval_dataset=val_rows if val_rows is not None and len(val_rows) else None,
        processing_class=processor,
        reference_model=reference,
        dataset_config=args.dataset_config,
        seed=args.seed,
        training_callback=training_callback,
        **reward_options,
    )
    if args.epochs is not None:
        signature = getattr(trainer.train_dataset, "media_signature", None)
        if args.algorithm == "grpo":
            # GRPO scores rows independently and accepts mixed text/image batches.
            signature = None
        counts = Counter(
            signature(i) if signature else () for i in range(len(train_rows))
        )
        batches = sum(count // settings.batch_size for count in counts.values())
        if not batches:
            raise ValueError("No complete batches fit the dataset; reduce batch_size.")
        trainer.args.iters = args.epochs * batches
        # Rebuild so schedules see the final epoch-derived update count.
        trainer.optimizer = build_optimizer(
            args, trainer.args.iters, settings.gradient_accumulation_steps
        )
    if max_iters and trainer.args.iters > max_iters:
        raise ValueError(f"iters {trainer.args.iters} is over the cap of {max_iters}.")
    total_iters, resumed_at = trainer.args.iters, 0
    if progress is not None:
        from .trainer.common.engine import GradientAccumulation

        trainer.accumulation = GradientAccumulation()
    if getattr(args, "resume_state", None):
        if not args.resume_adapter_file:
            raise ValueError("--resume-state needs --resume-adapter-file.")
        resumed_at, _ = restore_trainer_state(trainer, args.resume_state, progress)
        print(f"Continuing at iteration {resumed_at + 1} of {total_iters}.")
    if progress is not None:
        progress.total = total_iters
        progress.write(
            "start",
            task=args.task,
            algorithm=args.algorithm,
            iters=total_iters,
            resumed_at=resumed_at,
            adapter_file=settings.adapter_file,
        )

    def interrupted(event, **info):
        """Keep the work so far, and the state to carry on from it."""
        metrics = progress.whole_run(progress.last)
        if trainer.accumulation.microsteps and progress.world.size() > 1:
            from .trainer.common.utils import average_worker_gradients

            # Each rank has a different partial sum. Restoring their mean on
            # every rank preserves its contribution at the next gradient reduction.
            trainer.accumulation.gradients = average_worker_gradients(
                trainer.accumulation.gradients, progress.world
            )
            mx.eval(trainer.accumulation.gradients)
        if not primary:
            return metrics
        trainer.save_adapter()
        if "iteration" in metrics:
            save_trainer_state(
                trainer, metrics["iteration"], metrics.get("optimizer_step", 0)
            )
        progress.write(event, **info, **metrics)
        return metrics

    callback = None
    try:
        if args.wandb and primary:
            callback = WandBCallback(
                args.wandb,
                str(Path(settings.adapter_file).parent),
                vars(args) | asdict(trainer.args),
                training_callback,
            )
            trainer.training_callback = callback
        try:
            metrics = trainer.train()
        except UsageCapReached as reached:
            # Keep the work done so far; the cap is not a failure.
            metrics = interrupted("capped", reason=str(reached))
            if primary:
                print(
                    f"Stopped at usage cap: {reached}. Weights saved.", file=sys.stderr
                )
        except StopRequested:
            metrics = interrupted("stopped")
            if primary:
                print("Stopped on request. Weights saved.", file=sys.stderr)
        else:
            if primary and getattr(args, "export_qat_path", None):
                from .export import export_model

                export_model(
                    model,
                    processor,
                    args.export_qat_path,
                    model_path=args.model,
                    quantize=getattr(args, "export_quantize", None),
                )
            if progress is not None:
                progress.write("end", **progress.whole_run(metrics))
        return metrics
    finally:
        if callback is not None:
            callback.finish()


def main(argv=None, *, task="vlm"):
    if "--describe" in (sys.argv[1:] if argv is None else argv):
        print(json.dumps(describe(), indent=2, default=str))
        return
    args, settings = parse_args(argv, task=task)
    if args.dry_run:
        report = {
            "valid": True,
            "task": args.task,
            "algorithm": args.algorithm,
            "iters": None if args.epochs is not None else settings.iters,
            "epochs": args.epochs,
        }
        print(json.dumps(report))
        return
    run(args, settings)


if __name__ == "__main__":
    main()
