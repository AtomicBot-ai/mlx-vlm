"""Run manager for the training backend (python -m mlx_vlm.train_runs).

A host application (for example a Nativ extension) drives training through
this command instead of importing the trainers. Every command prints one JSON
document, so no Python needs to be embedded in the host.

Each run is ``python -m mlx_vlm.train -c <run>/config.json`` started as a
detached background process. Everything about a run lives in its own folder
under the runs root (default ``~/.mlx-vlm/runs``)::

    <run_id>/config.json     training options, as passed to mlx_vlm.train
    <run_id>/run.json        run state (survives restarts of the host)
    <run_id>/train.log       stdout + stderr of the training process
    <run_id>/progress.jsonl  one JSON event per report
    <run_id>/adapters/       trained weights and checkpoints

An optional ``limits.json`` in the runs root holds usage caps that apply to
every client of that folder.

Nothing here knows which tasks, algorithms or options exist: ``describe`` and
option validation are delegated to ``mlx_vlm.train``, which reads the trainer
registry. A new algorithm or modality therefore needs no change in this file.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
import re
import secrets
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

RUNS_DIR_ENV = "MLX_VLM_RUNS_DIR"
PYTHON_ENV = "MLX_VLM_TRAIN_PYTHON"
LIMIT_ENV_PREFIX = "MLX_VLM_TRAIN_"
DEFAULT_RUNS_DIR = "~/.mlx-vlm/runs"
LIMITS_FILE = "limits.json"
ACTIVE = {"running"}
# Options the manager owns; values passed by a caller are ignored.
MANAGED = {
    "config",
    "adapter_file",
    "output_path",
    "progress_file",
    "max_iters",
    "max_minutes",
    "max_memory_gb",
    "describe",
    "dry_run",
}

_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_ERROR = re.compile(r"^(\w+\.)*\w*(Error|Exception)\b.*:|error: ")


@dataclass
class Limits:
    """Usage caps set by whoever runs the manager. ``None`` means unlimited."""

    max_concurrent_runs: int | None = 1
    max_iters: int | None = None
    max_minutes: float | None = None
    max_memory_gb: float | None = None
    max_runs_per_day: int | None = None

    @classmethod
    def from_env(cls, **overrides: Any) -> "Limits":
        """Read caps from ``MLX_VLM_TRAIN_<NAME>``; explicit values win."""
        return cls(**cls._explicit(overrides))

    @classmethod
    def _explicit(cls, overrides: dict[str, Any]) -> dict[str, Any]:
        values: dict[str, Any] = {}
        for item in fields(cls):
            raw = overrides.get(item.name)
            if raw is None:
                raw = os.environ.get(LIMIT_ENV_PREFIX + item.name.upper())
            if raw is None or raw == "":
                continue
            values[item.name] = cls._number(item, raw)
        return values

    @staticmethod
    def _number(item, raw: Any) -> int | float:
        number = float(raw)
        if number <= 0:
            raise ValueError(f"{item.name} must be positive.")
        return number if "int" not in str(item.type) else int(number)

    @classmethod
    def load(cls, runs_dir: Path, **overrides: Any) -> "Limits":
        """Combine ``<runs_dir>/limits.json`` with flags and environment.

        The file holds the caps of whoever owns the runs folder, for example
        a host application's settings. A flag or environment variable can
        tighten a cap from the file but never loosen it, so every client of
        the same runs folder works under the same budget.
        """
        values = cls._explicit(overrides)
        try:
            stored = json.loads((runs_dir / LIMITS_FILE).read_text())
        except OSError:
            return cls(**values)
        if not isinstance(stored, dict):
            raise ValueError(f"{LIMITS_FILE} must hold a JSON object.")
        for item in fields(cls):
            if item.name not in stored:
                continue
            cap = stored[item.name]
            if cap is None:  # explicitly unlimited in the file
                values.setdefault(item.name, None)
                continue
            cap = cls._number(item, cap)
            given = values.get(item.name)
            values[item.name] = cap if given is None else min(cap, given)
        return cls(**values)


def _now() -> datetime:
    return datetime.now().astimezone()


def _read_tail(path: Path, max_bytes: int = 256_000) -> str:
    try:
        with open(path, "rb") as file:
            file.seek(max(0, path.stat().st_size - max_bytes))
            return file.read().decode("utf-8", "replace")
    except OSError:
        return ""


def _pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def resolve_paths(options: dict[str, Any], base: Path) -> dict[str, Any]:
    """Make local paths absolute, because a run works in its own folder.

    An existing local path is made absolute whatever the option is called, so
    options added by future trainers are covered. Output options that do not
    exist yet are recognised by their ``_path``/``_file`` suffix.
    """
    resolved = dict(options)
    for key, value in options.items():
        if not isinstance(value, str) or not value.strip():
            continue
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = base / path
        if path.exists() or (
            key.endswith(("_path", "_file")) and value.startswith((".", "~", "/"))
        ):
            resolved[key] = str(path.resolve())
    return resolved


class RunManager:
    """Training runs on disk. Imports no MLX, so it starts fast."""

    def __init__(
        self,
        runs_dir: str | Path | None = None,
        limits: Limits | None = None,
        python: str | None = None,
    ):
        root = runs_dir or os.environ.get(RUNS_DIR_ENV) or DEFAULT_RUNS_DIR
        self.root = Path(root).expanduser().resolve()
        self.limits = limits or Limits.load(self.root)
        self.python = python or os.environ.get(PYTHON_ENV) or sys.executable
        # Runs started by this process, so finished children get reaped.
        self._procs: dict[str, subprocess.Popen] = {}

    # -- the training CLI ---------------------------------------------------
    def _train_command(self, *args: str) -> list[str]:
        return [self.python, "-m", "mlx_vlm.train", *args]

    def _env(self) -> dict[str, str]:
        # Runs change directory, so keep this checkout importable.
        package_root = str(Path(__file__).resolve().parent.parent)
        path = os.environ.get("PYTHONPATH")
        return {
            **os.environ,
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": package_root + (os.pathsep + path if path else ""),
        }

    def describe(self) -> dict[str, Any]:
        """Tasks, algorithms and options, as reported by the trainer itself."""
        done = subprocess.run(
            self._train_command("--describe"),
            capture_output=True,
            text=True,
            env=self._env(),
        )
        if done.returncode:
            raise RuntimeError(done.stderr.strip() or "mlx_vlm.train --describe failed")
        return json.loads(done.stdout)

    def _config(self, options: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
        if not isinstance(options, dict) or any(
            not isinstance(key, str) for key in options
        ):
            raise ValueError("Training options must be a JSON object.")
        notes = [
            f"Ignored '{key}'; the run manager sets it."
            for key in sorted(MANAGED & set(options))
        ]
        config = {k: v for k, v in options.items() if k not in MANAGED}
        config = resolve_paths(config, Path.cwd())
        for name in ("max_iters", "max_minutes", "max_memory_gb"):
            if getattr(self.limits, name) is not None:
                config[name] = getattr(self.limits, name)
        return config, notes

    def check(self, options: dict[str, Any]) -> dict[str, Any]:
        """Validate options with the real training parser; start nothing."""
        config, notes = self._config(options)
        done = subprocess.run(
            self._train_command("-c", "/dev/stdin", "--dry-run"),
            input=json.dumps(config),
            capture_output=True,
            text=True,
            env=self._env(),
        )
        errors = []
        resolved = {}
        if not done.returncode:
            # The parser fills in defaults such as the task and algorithm.
            with contextlib.suppress(ValueError):
                resolved = json.loads(done.stdout)
            for key in ("task", "algorithm"):
                if resolved.get(key):
                    config[key] = resolved[key]
        if done.returncode:
            lines = [line for line in done.stderr.strip().splitlines() if line.strip()]
            message = lines[-1] if lines else "Validation failed."
            errors.append(message.split("error: ", 1)[-1])
        return {"valid": not errors, "errors": errors, "notes": notes, "config": config}

    # -- state --------------------------------------------------------------
    def _save(self, run: dict[str, Any]) -> None:
        path = self.root / run["id"] / "run.json"
        # Several processes (a host app's polls, its tools) can settle the same
        # run at once, so each writes its own temporary file before the swap.
        tmp = path.with_name(f"run.{os.getpid()}.{secrets.token_hex(4)}.tmp")
        try:
            tmp.write_text(json.dumps(run, indent=2))
            tmp.replace(path)
        finally:
            tmp.unlink(missing_ok=True)

    def _load(self, run_id: str) -> dict[str, Any] | None:
        try:
            return json.loads((self.root / run_id / "run.json").read_text())
        except (OSError, ValueError):
            return None

    def _refresh(self, run: dict[str, Any]) -> dict[str, Any]:
        """Settle the status of a run whose process may have ended."""
        if run["status"] not in ACTIVE:
            return run
        proc = self._procs.get(run["id"])
        if proc:
            code = proc.poll()
            if code is None:
                return run
        else:
            # The wrapper writes this marker when training exits. Prefer it to
            # the saved PID, which the OS may have reused for another process.
            code = None
            with contextlib.suppress(OSError, ValueError):
                code = int((self.root / run["id"] / "exit_code").read_text().strip())
            if code is None and _pid_alive(run.get("pid")):
                return run
        self._procs.pop(run["id"], None)
        run["exit_code"] = code
        run["ended_at"] = run.get("ended_at") or _now().isoformat(timespec="seconds")
        if run.get("stop_requested"):
            run["status"] = "stopped"
        elif code == 0:
            capped = self.progress(run).get("capped")
            run["status"] = "capped" if capped else "completed"
        else:
            run["status"] = "failed"
        self._save(run)
        return run

    def runs(self) -> list[dict[str, Any]]:
        if not self.root.exists():
            return []
        found = [self._load(p.name) for p in self.root.iterdir() if p.is_dir()]
        runs = [self._refresh(r) for r in found if r]
        return sorted(runs, key=lambda r: r["created_at"], reverse=True)

    def get(self, run_id: str = "") -> dict[str, Any]:
        """Find a run by id, id prefix or name; empty means the newest run."""
        runs = self.runs()
        if not runs:
            raise LookupError("No training runs yet.")
        if run_id in ("", "latest"):
            return runs[0]
        for run in runs:
            if run_id in (run["id"], run.get("name")):
                return run
        matches = [r for r in runs if r["id"].startswith(run_id)]
        if len(matches) == 1:
            return matches[0]
        raise LookupError(f"No run '{run_id}'.")

    def usage(self) -> dict[str, Any]:
        """The caps and how much of each is in use right now."""
        runs = self.runs()
        day_ago = (_now() - timedelta(days=1)).isoformat(timespec="seconds")
        return {
            "limits": asdict(self.limits),
            "active_runs": sum(r["status"] in ACTIVE for r in runs),
            "runs_last_24h": sum(r["created_at"] >= day_ago for r in runs),
        }

    # -- actions ------------------------------------------------------------
    def start(self, options: dict[str, Any], name: str = "") -> dict[str, Any]:
        checked = self.check(options)
        if not checked["valid"]:
            raise ValueError(" ".join(checked["errors"]))
        self.root.mkdir(parents=True, exist_ok=True)
        # Several hosts may share one runs root. Hold one lock over the cap
        # check, the launch and the first state write.
        with open(self.root / ".start.lock", "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            return self._start(name, checked["config"], checked["notes"])

    def _start(
        self, name: str, config: dict[str, Any], notes: list[str]
    ) -> dict[str, Any]:
        usage = self.usage()
        cap = self.limits.max_concurrent_runs
        if cap is not None and usage["active_runs"] >= cap:
            raise RuntimeError(
                f"Usage cap: {usage['active_runs']} of {cap} concurrent runs in use. "
                "Stop a run first."
            )
        cap = self.limits.max_runs_per_day
        if cap is not None and usage["runs_last_24h"] >= cap:
            raise RuntimeError(
                f"Usage cap: {cap} runs were already started in the last 24 hours."
            )
        now = _now()
        algorithm = config.get("algorithm") or "default"
        run_id = f"{now:%Y%m%d-%H%M%S}-{algorithm}-{secrets.token_hex(2)}"
        run_dir = self.root / run_id
        run_dir.mkdir(parents=True)
        config["adapter_file"] = str(run_dir / "adapters" / "adapters.safetensors")
        config["progress_file"] = str(run_dir / "progress.jsonl")
        (run_dir / "adapters").mkdir()
        (run_dir / "config.json").write_text(json.dumps(config, indent=2))

        command = self._train_command("-c", str(run_dir / "config.json"))
        # sh records the exit code, so the outcome is known after a restart.
        wrapper = [
            "/bin/sh",
            "-c",
            '"$@"; rc=$?; echo $rc > exit_code; exit $rc',
            "sh",
            *command,
        ]
        with open(run_dir / "train.log", "ab") as log:
            proc = subprocess.Popen(
                wrapper,
                cwd=run_dir,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=self._env(),
                start_new_session=True,  # own process group, so stop kills it all
            )
        self._procs[run_id] = proc
        run = {
            "id": run_id,
            "name": name.strip() or None,
            "status": "running",
            "pid": proc.pid,
            "created_at": now.isoformat(timespec="seconds"),
            "ended_at": None,
            "exit_code": None,
            "stop_requested": False,
            "task": config.get("task"),
            "algorithm": config.get("algorithm"),
            "model": config.get("model"),
            "run_dir": str(run_dir),
            "adapter_path": str(run_dir / "adapters"),
            "log_path": str(run_dir / "train.log"),
            "progress_path": config["progress_file"],
            # A merged, standalone model, when the run asks for one.
            "export_path": config.get("export_qat_path"),
            "command": command,
            "limits": asdict(self.limits),
            "notes": notes,
        }
        self._save(run)
        return run

    def stop(self, run_id: str = "", wait_s: float = 5.0) -> dict[str, Any]:
        run = self.get(run_id)
        if run["status"] not in ACTIVE:
            return run
        run["stop_requested"] = True
        self._save(run)
        for sig in (signal.SIGTERM, signal.SIGKILL):
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(run["pid"], sig)
            deadline = time.monotonic() + wait_s
            while time.monotonic() < deadline:
                if self._refresh(run)["status"] not in ACTIVE:
                    return run
                time.sleep(0.2)
        return run

    def progress(self, run: dict[str, Any]) -> dict[str, Any]:
        """Latest metrics from the run's progress file."""
        info: dict[str, Any] = {}
        for line in _read_tail(Path(run["progress_path"])).splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue  # a partial first or last line
            kind = event.pop("event", None)
            event.pop("time", None)
            if kind == "start":
                info["iters"] = event.get("iters")
            elif kind in ("train", "validation"):
                info["iteration"] = event.get("iteration", info.get("iteration"))
                info.setdefault("metrics", {}).update(event)
            elif kind == "capped":
                info["capped"] = event.get("reason")
        return info

    def history(self, run: dict[str, Any], max_points: int = 400) -> dict[str, Any]:
        """Every numeric metric of a run as ``[iteration, value]`` points.

        Reads the whole progress file. Whatever the algorithm reported is
        returned under its own name, so charts follow the algorithm without
        this code knowing any metric. Long runs are thinned to ``max_points``
        per metric, always keeping the first and last point.
        """
        series: dict[str, list[list[float]]] = {}
        iters = None
        try:
            lines = Path(run["progress_path"]).read_text().splitlines()
        except OSError:
            lines = []
        for line in lines:
            try:
                event = json.loads(line)
            except ValueError:
                continue
            kind = event.get("event")
            if kind == "start":
                iters = event.get("iters")
            if kind not in ("train", "validation"):
                continue
            iteration = event.get("iteration")
            if not isinstance(iteration, (int, float)):
                continue
            for name, value in event.items():
                if name in ("event", "time", "iteration") or isinstance(value, bool):
                    continue
                if isinstance(value, (int, float)) and value == value:  # skips NaN
                    series.setdefault(name, []).append([iteration, value])
        max_points = max(2, max_points)
        for name, points in series.items():
            if len(points) > max_points:
                step = (len(points) - 1) / (max_points - 1)
                series[name] = [points[round(i * step)] for i in range(max_points)]
        return {
            "id": run["id"],
            "name": run.get("name"),
            "status": run["status"],
            "task": run.get("task"),
            "algorithm": run.get("algorithm"),
            "iters": iters,
            "series": series,
        }

    def log_tail(self, run: dict[str, Any], lines: int = 30) -> str:
        text = _ANSI.sub("", _read_tail(Path(run["log_path"]), 64_000))
        # tqdm redraws with \r; keep only the last redraw of each line.
        rows = [line.split("\r")[-1].rstrip() for line in text.split("\n")]
        rows = [row for row in rows if row]
        return "\n".join(rows[-max(1, min(lines, 500)) :])

    def summary(self, run: dict[str, Any]) -> dict[str, Any]:
        keep = (
            "id",
            "name",
            "status",
            "task",
            "algorithm",
            "model",
            "created_at",
            "ended_at",
            "exit_code",
            "adapter_path",
            "export_path",
            "log_path",
            "notes",
        )
        out = {key: run.get(key) for key in keep}
        out.update(self.progress(run))
        if run["status"] == "failed":
            errors = [
                row
                for row in self.log_tail(run, 200).splitlines()
                if _ERROR.search(row)
            ]
            out["error"] = errors[-1].strip() if errors else None
        return out


# -- preflight checks ---------------------------------------------------------
def validate_model(model: str) -> dict[str, Any]:
    """Check that a model's architecture is supported for training.

    Reads ``config.json`` only: no weights are downloaded and nothing is
    trained, so this says nothing about memory fit.
    """
    from mlx_vlm.trainer.core import not_supported_for_training
    from mlx_vlm.utils import get_model_and_args

    path = Path(model).expanduser()
    if path.is_dir():
        config_path = path / "config.json"
    else:
        from huggingface_hub import hf_hub_download

        config_path = Path(hf_hub_download(model, "config.json"))
    config = json.loads(config_path.read_text())
    _, model_type = get_model_and_args(config)
    if model_type in not_supported_for_training:
        raise ValueError(f"Model type '{model_type}' is not supported for training.")
    return {"valid": True, "model": model, "model_type": model_type}


def inspect_dataset(options: dict[str, Any]) -> dict[str, Any]:
    """Load a dataset the way training would and report its splits."""
    from argparse import Namespace
    from collections.abc import Mapping

    from mlx_vlm.train import _load_dataset

    args = Namespace(
        data=options.get("data", "data"),
        dataset=options.get("dataset"),
        hf_dataset_config=options.get("hf_dataset_config"),
    )
    with contextlib.redirect_stdout(sys.stderr):  # keep stdout pure JSON
        raw = _load_dataset(args)
    splits = raw if isinstance(raw, Mapping) else {"train": raw}
    train_split = options.get("train_split", "train")
    report = {}
    for name, rows in splits.items():
        first = rows[0] if len(rows) else {}
        report[name] = {
            "rows": len(rows),
            "columns": sorted(first) if isinstance(first, Mapping) else [],
        }
    if not report.get(train_split, {}).get("rows"):
        raise ValueError(f"Dataset has no non-empty '{train_split}' split.")
    return {"valid": True, "splits": report}


# -- command line -------------------------------------------------------------
def _options(args: argparse.Namespace) -> dict[str, Any]:
    if args.options_file:
        text = (
            sys.stdin.read()
            if args.options_file == "-"
            else Path(args.options_file).read_text()
        )
    else:
        text = args.options or "{}"
    options = json.loads(text)
    if not isinstance(options, dict):
        raise ValueError("Training options must be a JSON object.")
    return options


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Start, follow and stop mlx_vlm training runs. Prints JSON."
    )
    parser.add_argument(
        "--runs-dir",
        help=f"Where runs are stored (env {RUNS_DIR_ENV}, default {DEFAULT_RUNS_DIR}).",
    )
    for item in fields(Limits):
        parser.add_argument(
            "--" + item.name.replace("_", "-"),
            type=float,
            help=f"Usage cap (env {LIMIT_ENV_PREFIX}{item.name.upper()}).",
        )
    commands = parser.add_subparsers(dest="command", required=True)

    def add(name: str, help: str, *, options: bool = False, run: bool = False):
        command = commands.add_parser(name, help=help)
        if options:
            command.add_argument("--options", help="Training options as a JSON object.")
            command.add_argument(
                "--options-file", help="File holding the options; '-' reads stdin."
            )
        if run:
            command.add_argument(
                "run_id", nargs="?", default="", help="Run id or name; default latest."
            )
        return command

    add("describe", "List tasks, algorithms and their options.")
    add("limits", "Show the usage caps and current usage.")
    add("validate-model", "Check a model is supported.").add_argument("model")
    add("inspect-dataset", "Check a dataset loads.", options=True)
    add("check", "Validate training options without starting.", options=True)
    add("start", "Start a training run.", options=True).add_argument(
        "--name", default="", help="Optional label for the run."
    )
    add("status", "Show a run's progress.", run=True)
    add("log", "Show the end of a run's log.", run=True).add_argument(
        "--lines", type=int, default=30
    )
    add("history", "Show a run's metrics over time.", run=True).add_argument(
        "--max-points", type=int, default=400
    )
    add("list", "List recent runs.").add_argument("--limit", type=int, default=20)
    add("stop", "Stop a run; saved checkpoints are kept.", run=True)
    return parser


def run_command(args: argparse.Namespace) -> Any:
    manager = RunManager(args.runs_dir)
    flags = {item.name: getattr(args, item.name) for item in fields(Limits)}
    limits = manager.limits = Limits.load(manager.root, **flags)
    command = args.command
    if command == "describe":
        return manager.describe() | {"limits": asdict(limits)}
    if command == "limits":
        return manager.usage()
    if command == "validate-model":
        return validate_model(args.model)
    if command == "inspect-dataset":
        return inspect_dataset(_options(args))
    if command == "check":
        return manager.check(_options(args))
    if command == "start":
        return manager.summary(manager.start(_options(args), args.name))
    if command == "list":
        return {"runs": [manager.summary(r) for r in manager.runs()[: args.limit]]}
    if command == "stop":
        return manager.summary(manager.stop(args.run_id))
    run = manager.get(args.run_id)
    if command == "log":
        return {"id": run["id"], "log": manager.log_tail(run, args.lines)}
    if command == "history":
        return manager.history(run, args.max_points)
    return manager.summary(run)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        result = run_command(args)
    except Exception as error:  # every failure is reported as JSON
        print(json.dumps({"error": str(error), "type": type(error).__name__}))
        sys.exit(1)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
