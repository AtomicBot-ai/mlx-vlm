"""The self-describing training CLI and the run manager built on it."""

import io
import json
import os
import signal
import socket
import subprocess
import sys
import time
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from mlx_vlm import train as train_cli
from mlx_vlm.train import ProgressCallback, UsageCapReached, describe, parse_args
from mlx_vlm.train_runs import Limits, RunManager, inspect_dataset, main, resolve_paths
from mlx_vlm.trainer.api import TASKS
from mlx_vlm.trainer.vlm.config import VLMTrainConfig

FAKE_PYTHON = """#!/bin/sh
# Stands in for `python -m mlx_vlm.train`; the run folder is the working dir.
case "$*" in
  *--dry-run*) cat > /dev/null; echo '{"valid": true}'; exit 0 ;;
esac
echo '{"event": "start", "iters": 10}' >> progress.jsonl
echo '{"event": "train", "iteration": 5, "train_loss": 1.5}' >> progress.jsonl
case "$FAKE_TRAIN" in
  fail) echo "ValueError: boom"; exit 1 ;;
  hang) exec sleep 60 ;;
  ignore-term) trap '' TERM; touch ready; exec sleep 60 ;;
  wait)  # a trainer that saves and ends when asked to stop
    while [ ! -e stop ]; do sleep 0.05; done
    : > adapters/adapters.safetensors
    : > adapters/trainer_state.safetensors
    echo '{"event": "stopped", "iteration": 5}' >> progress.jsonl
    exit 0 ;;
  capped) echo '{"event": "capped", "reason": "max_minutes (1) reached"}' >> progress.jsonl ;;
esac
echo '{"event": "validation", "iteration": 10, "val_loss": 1.2}' >> progress.jsonl
"""


@pytest.fixture
def manager(tmp_path, monkeypatch):
    python = tmp_path / "fake-python"
    python.write_text(FAKE_PYTHON)
    python.chmod(0o755)
    monkeypatch.delenv("FAKE_TRAIN", raising=False)
    return RunManager(tmp_path / "runs", Limits(), python=str(python))


def wait_for_end(manager, run_id, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        run = manager.get(run_id)
        if run["status"] != "running":
            return run
        time.sleep(0.05)
    raise AssertionError("run did not finish")


# -- the training CLI describes itself ----------------------------------------
def test_describe_lists_every_registered_algorithm_and_its_options():
    report = describe()
    for task, algorithm in TASKS:
        entry = report["tasks"][task]["algorithms"][algorithm]
        names = {option["name"] for option in entry["options"]}
        assert {"model", "iters", "batch_size", "learning_rate"} <= names
    grpo = report["tasks"]["vlm"]["algorithms"]["grpo"]
    assert "accuracy_reward" in grpo["reward_functions"]
    options = {o["name"]: o for o in grpo["options"]}
    assert options["iters"]["type"] == "int" and options["iters"]["group"] == "training"
    assert options["grad_clip"]["type"] == "float"  # Optional[float] unwrapped
    assert options["train_type"]["choices"] == ["lora", "dora", "full"]
    assert options["max_minutes"]["group"] == "managed"
    json.dumps(report)  # must be plain JSON


def test_new_algorithm_and_modality_appear_without_cli_changes():
    @dataclass
    class EchoConfig(VLMTrainConfig):
        """Settings for an imaginary echo objective."""

        echo_strength: float = 0.5

    extra = {
        ("vlm", "echo"): ("vlm.echo.runtime", EchoConfig),
        ("asr", "ctc"): ("asr.ctc.runtime", EchoConfig),
    }
    with patch.dict(TASKS, extra):
        report = describe()
        echo = report["tasks"]["vlm"]["algorithms"]["echo"]
        assert echo["description"] == "Settings for an imaginary echo objective."
        assert "echo_strength" in {o["name"] for o in echo["options"]}
        assert "ctc" in report["tasks"]["asr"]["algorithms"]
        assert report["tasks"]["vlm"]["default_algorithm"] == "sft"
        assert report["tasks"]["asr"]["default_algorithm"] == "ctc"  # no sft there
        args, settings = parse_args(
            [
                "--task",
                "asr",
                "--algorithm",
                "ctc",
                "--model",
                "m",
                "--echo-strength",
                "2",
            ]
        )
        assert args.task == "asr" and settings.echo_strength == 2.0
    assert "asr" not in describe()["tasks"]


def test_dry_run_validates_without_loading_anything(capsys):
    with patch.object(train_cli, "run") as run:
        train_cli.main(["--model", "m", "--iters", "7", "--dry-run"])
        train_cli.main(["--model", "m", "--epochs", "2", "--dry-run"])
    run.assert_not_called()
    first, second = map(json.loads, capsys.readouterr().out.strip().splitlines())
    assert first == {
        "valid": True,
        "task": "vlm",
        "algorithm": "sft",
        "iters": 7,
        "epochs": None,
    }
    assert second["epochs"] == 2 and second["iters"] is None


@pytest.mark.parametrize(
    "flags",
    [
        ["--iters", "50", "--max-iters", "20"],
        ["--max-minutes", "0"],
        ["--max-memory-gb", "-1"],
    ],
)
def test_cap_flags_are_validated(flags):
    with pytest.raises(SystemExit):
        parse_args(["--model", "m", *flags])


def test_progress_callback_writes_events_and_enforces_caps(tmp_path):
    path = tmp_path / "nested" / "progress.jsonl"
    wrapped = MagicMock()
    callback = ProgressCallback(path, wrapped, max_minutes=60)
    callback.on_train_loss_report({"iteration": 10, "train_loss": 2.0})
    callback.on_val_loss_report({"iteration": 10, "val_loss": 1.9})
    events = [json.loads(line) for line in path.read_text().splitlines()]
    assert [e["event"] for e in events] == ["train", "validation"]
    assert events[0]["train_loss"] == 2.0 and "active_memory_gb" in events[0]
    wrapped.on_train_loss_report.assert_called_once()
    wrapped.on_val_loss_report.assert_called_once()

    callback.started -= 3600  # an hour has passed
    with pytest.raises(UsageCapReached, match="max_minutes"):
        callback.on_train_loss_report({"iteration": 20, "train_loss": 1.0})
    with patch("mlx.core.get_active_memory", return_value=3e9):
        with pytest.raises(UsageCapReached, match="max_memory_gb"):
            ProgressCallback(max_memory_gb=2).on_train_loss_report({"iteration": 1})


def test_run_saves_weights_when_a_cap_stops_training(tmp_path):
    progress = tmp_path / "progress.jsonl"
    args, settings = parse_args(
        [
            "--model",
            "m",
            "--progress-file",
            str(progress),
            "--max-minutes",
            "1",
            "--max-iters",
            "500",
        ]
    )
    import mlx.core as mx

    trainer = MagicMock()
    trainer.args.iters = settings.iters
    trainer.args.adapter_file = str(tmp_path / "adapters.safetensors")
    trainer.optimizer.state = {"step": mx.array(10), "weight": {"m": mx.zeros(2)}}
    trainer.log_history = [{"iteration": 10, "optimizer_step": 10, "train_loss": 2.0}]

    def build_trainer(*_, training_callback=None, **__):
        def train():
            training_callback.started -= 3600
            training_callback.on_train_loss_report(trainer.log_history[0])

        trainer.train.side_effect = train
        return trainer

    with (
        patch("mlx_vlm.train._load_model", return_value=(MagicMock(), MagicMock())),
        patch("mlx_vlm.train._load_dataset", return_value={"train": [{}]}),
        patch("mlx_vlm.train._configure_model", side_effect=lambda model, _: model),
        patch("mlx_vlm.train.Trainer", side_effect=build_trainer),
        patch("mlx.core.set_memory_limit"),
    ):
        metrics = train_cli.run(args, settings)
    trainer.save_adapter.assert_called_once()
    assert metrics["iteration"] == 10
    # The state to carry on from is saved beside the weights.
    state, meta = mx.load(
        str(tmp_path / "trainer_state.safetensors"), return_metadata=True
    )
    assert meta == {
        "iteration": "10",
        "optimizer_step": "10",
        "pending_microsteps": "0",
    }
    assert {"optimizer.step", "optimizer.weight.m", "random.key"} <= set(state)
    events = [json.loads(line)["event"] for line in progress.read_text().splitlines()]
    assert events == ["start", "train", "capped"]


def test_run_refuses_epoch_derived_iters_over_the_cap():
    args, settings = parse_args(["--model", "m", "--epochs", "3", "--max-iters", "5"])
    trainer = MagicMock()
    trainer.train_dataset = SimpleNamespace(media_signature=None)
    with (
        patch("mlx_vlm.train._load_model", return_value=(MagicMock(), MagicMock())),
        patch("mlx_vlm.train._load_dataset", return_value={"train": [{}, {}]}),
        patch("mlx_vlm.train._configure_model", side_effect=lambda model, _: model),
        patch("mlx_vlm.train.Trainer", return_value=trainer),
        pytest.raises(ValueError, match="over the cap"),
    ):
        train_cli.run(args, settings)
    trainer.train.assert_not_called()


def _distributed_stop_worker(root):
    import mlx.core as mx
    import numpy as np

    from mlx_vlm.tests.test_training_cli import TinyVLM

    mx.set_default_device(mx.cpu)
    rank = mx.distributed.init(strict=True, backend="ring").rank()
    for case in ("stop", "memory", "time"):
        directory = Path(root) / case / str(rank)
        args, settings = parse_args(
            [
                "--model",
                "tiny",
                "--iters",
                "3",
                "--batch-size",
                "2",
                "--steps-per-report",
                "1",
                "--steps-per-eval",
                "1",
                "--max-memory-gb",
                "2",
                "--max-minutes",
                "0.000001" if case == "time" and rank == 1 else "60",
                "--target-modules",
                "projection",
                "--progress-file",
                str(directory / "progress.jsonl"),
                "--adapter-file",
                str(directory / "adapters.safetensors"),
            ]
        )
        # Only rank zero sees the stop file. Only rank one exceeds the cap.
        memory = 3_000_000_000 if case == "memory" and rank == 1 else 0
        rows = [{"text": "first"}, {"text": "second"}]

        def prepare(**kwargs):
            return {"input_ids": np.array([[1, 2, 3, 4]])}

        with (
            patch.object(
                train_cli,
                "_load_model",
                return_value=(TinyVLM(), SimpleNamespace(pad_token_id=0)),
            ),
            patch.object(
                train_cli, "_load_dataset", return_value={"train": rows, "valid": rows}
            ),
            patch(
                "mlx_vlm.trainer.vlm.cross_entropy.dataset._load_vlm_helpers",
                return_value=({}, None, prepare, None),
            ),
            patch.object(mx, "get_active_memory", return_value=memory),
        ):
            metrics = train_cli.run(args, settings)
        assert metrics["iteration"] == 1
    _check_accumulation_resume(Path(root) / "distributed-resume")


def _check_accumulation_resume(root, compile_steps=False):
    import mlx.core as mx
    import numpy as np

    from mlx_vlm.tests.test_training_cli import TinyVLM

    world = mx.distributed.init()
    rank = world.rank()
    rows = [{"text": str(i)} for i in range(4)]

    def prepare(*, prompts, **kwargs):
        # Distinct microbatches and worker gradients expose dropped sums.
        token = 1 + int(prompts[0]) + 4 * rank
        return {"input_ids": np.array([[token, 2, 3, 4]])}

    for name, previous, stop in (
        ("baseline", None, False),
        ("first", None, True),
        ("second", "first", True),
        ("resumed", "second", False),
    ):
        directory = root / name
        flags = [
            "--model",
            "tiny",
            "--train-type",
            "full",
            "--batch-size",
            str(world.size()),
            "--iters",
            "5",
            "--gradient-accumulation-steps",
            "3",
            "--steps-per-report",
            "1",
            "--optimizer",
            "sgd",
            "--learning-rate",
            "0.03",
            "--progress-file",
            str(directory / "progress.jsonl"),
            "--adapter-file",
            str(directory / "adapters.safetensors"),
        ]
        if compile_steps:
            flags.append("--compile")
        if previous:
            flags += [
                "--resume-adapter-file",
                str(root / previous / "adapters.safetensors"),
                "--resume-state",
                str(root / previous / "trainer_state.safetensors"),
            ]
        args, settings = parse_args(flags)
        if stop and rank == 0:
            directory.mkdir(parents=True)
            (directory / "stop").touch()
        with (
            patch.object(
                train_cli,
                "_load_model",
                side_effect=lambda *_: (TinyVLM(), SimpleNamespace(pad_token_id=0)),
            ),
            patch.object(train_cli, "_load_dataset", return_value={"train": rows}),
            patch(
                "mlx_vlm.trainer.vlm.cross_entropy.dataset._load_vlm_helpers",
                return_value=({}, None, prepare, None),
            ),
        ):
            train_cli.run(args, settings)
        if world.size() > 1:
            mx.eval(mx.distributed.all_sum(mx.array(1), stream=mx.cpu))
        if stop:
            arrays, meta = mx.load(
                str(directory / "trainer_state.safetensors"), return_metadata=True
            )
            assert meta["pending_microsteps"] == ("1" if name == "first" else "2")
            assert "gradient.projection.weight" in arrays

    for name in ("baseline", "resumed"):
        end = json.loads((root / name / "progress.jsonl").read_text().splitlines()[-1])
        assert (end["event"], end["iteration"], end["optimizer_step"]) == ("end", 5, 2)
    baseline, resumed = [
        mx.load(str(root / name / "adapters.safetensors"))
        for name in ("baseline", "resumed")
    ]
    assert resumed.keys() == baseline.keys()
    for key in baseline:
        if world.size() == 1:
            assert mx.array_equal(baseline[key], resumed[key]).item(), key
        else:
            assert mx.allclose(
                baseline[key], resumed[key], atol=1e-7, rtol=1e-6
            ).item(), key


@pytest.mark.parametrize("compile_steps", [False, True])
def test_resume_preserves_pending_gradients_and_final_partial_update(
    tmp_path, compile_steps
):
    _check_accumulation_resume(tmp_path, compile_steps)


def test_legacy_accumulation_state_and_incompatible_windows(tmp_path):
    import mlx.core as mx
    import mlx.optimizers as optim

    from mlx_vlm.tests.test_training_cli import TinyVLM

    path = tmp_path / "trainer_state.safetensors"
    model = TinyVLM()
    trainer = SimpleNamespace(
        model=model,
        args=SimpleNamespace(iters=5, gradient_accumulation_steps=3),
        optimizer=optim.SGD(0.1),
        task=SimpleNamespace(batches=iter),
    )
    meta = {"iteration": "1", "optimizer_step": "0"}
    mx.save_safetensors(str(path), {"optimizer.step": mx.array(0)}, metadata=meta)
    assert train_cli.restore_trainer_state(trainer, path) == (1, 5)
    assert (
        trainer.accumulation.microsteps == 0 and trainer.accumulation.gradients is None
    )
    mx.save_safetensors(
        str(path),
        {"gradient.projection.weight": model.projection.weight},
        metadata={
            **meta,
            "pending_microsteps": "1",
            "gradient_accumulation_steps": "2",
        },
    )
    with pytest.raises(ValueError, match="original gradient_accumulation_steps"):
        train_cli.restore_trainer_state(trainer, path)


def test_stop_and_caps_exit_all_ranks_at_the_same_report(tmp_path):
    import mlx.core as mx

    if not mx.distributed.is_available("ring"):
        pytest.skip("MLX ring backend is unavailable")
    primary = tmp_path / "stop" / "0"
    primary.mkdir(parents=True)
    (primary / "stop").touch()
    hosts = tmp_path / "hosts.json"
    with ExitStack() as stack:
        addresses = []
        for _ in range(2):
            sock = stack.enter_context(socket.socket())
            sock.bind(("127.0.0.1", 0))
            addresses.append([f"127.0.0.1:{sock.getsockname()[1]}"])
        hosts.write_text(json.dumps(addresses))
    command = [
        sys.executable,
        "-c",
        "import sys; from mlx_vlm.tests.test_train_runs import _distributed_stop_worker; "
        "_distributed_stop_worker(sys.argv[1])",
        str(tmp_path),
    ]
    workers = []
    try:
        for rank in range(2):
            workers.append(
                subprocess.Popen(
                    command,
                    cwd=Path(__file__).resolve().parents[2],
                    env={
                        **os.environ,
                        "MLX_RANK": str(rank),
                        "MLX_HOSTFILE": str(hosts),
                    },
                )
            )
        for worker in workers:
            assert worker.wait(timeout=90) == 0
    finally:
        for worker in workers:
            worker.kill()
            worker.wait()
    for case, event in (("stop", "stopped"), ("memory", "capped"), ("time", "capped")):
        primary = tmp_path / case / "0"
        events = [
            json.loads(line)
            for line in (primary / "progress.jsonl").read_text().splitlines()
        ]
        assert [record["event"] for record in events] == ["start", "train", event]
        if case != "stop":
            assert ("max_memory_gb" if case == "memory" else "max_minutes") in events[
                -1
            ]["reason"]
        assert events[-1]["iteration"] == 1
        assert (primary / "adapters.safetensors").is_file()
        assert (primary / "trainer_state.safetensors").is_file()
        assert not (primary.parent / "1").exists()


# -- the run manager ------------------------------------------------------------
def test_check_uses_the_real_training_parser(tmp_path):
    manager = RunManager(tmp_path, Limits(max_iters=20))
    good = manager.check({"model": "m", "algorithm": "dpo", "iters": 20, "beta": 0.2})
    assert good["valid"] and good["config"]["max_iters"] == 20
    over = manager.check({"model": "m", "iters": 21, "adapter_file": "x"})
    assert not over["valid"] and "over the cap" in over["errors"][0]
    assert over["notes"] == ["Ignored 'adapter_file'; the run manager sets it."]
    unknown = manager.check({"model": "m", "no_such_option": 1})
    assert "Unknown config options: no_such_option" in unknown["errors"][0]
    assert "sft" in manager.describe()["tasks"]["vlm"]["algorithms"]


def test_run_lifecycle_completes_and_reports_progress(manager):
    run = manager.start({"model": "m", "algorithm": "sft"}, name="first")
    assert run["status"] == "running"
    config = json.loads((manager.root / run["id"] / "config.json").read_text())
    assert config["adapter_file"].endswith("adapters/adapters.safetensors")
    assert config["progress_file"].endswith("progress.jsonl")
    done = wait_for_end(manager, run["id"])
    assert done["status"] == "completed" and done["exit_code"] == 0
    summary = manager.summary(done)
    assert summary["iteration"] == 10 and summary["iters"] == 10
    assert summary["metrics"]["train_loss"] == 1.5
    assert summary["metrics"]["val_loss"] == 1.2
    assert manager.get("first")["id"] == run["id"]
    assert manager.get(run["id"][:15])["id"] == run["id"]
    # A new manager (the host restarted) still knows the run.
    assert RunManager(manager.root).get()["status"] == "completed"


def test_check_reads_nested_schedule_as_json(tmp_path):
    manager = RunManager(tmp_path)
    schedule = {"name": "cosine_decay", "arguments": [1e-5, 20, 1e-6]}
    checked = manager.check({"model": "m", "iters": 20, "lr_schedule": schedule})
    assert checked["valid"], checked["errors"]
    path = tmp_path / "config.json"
    path.write_text(json.dumps(checked["config"]))
    with patch("sys.stdin", io.StringIO(path.read_text())):
        stdin_args, _ = parse_args(["-c", "-"])
    file_args, _ = parse_args(["-c", str(path)])
    assert stdin_args.lr_schedule == file_args.lr_schedule == schedule


def test_failed_and_capped_runs_are_told_apart(manager, monkeypatch):
    monkeypatch.setenv("FAKE_TRAIN", "fail")
    failed = wait_for_end(manager, manager.start({"model": "m"})["id"])
    assert failed["status"] == "failed" and failed["exit_code"] == 1
    assert manager.summary(failed)["error"] == "ValueError: boom"
    assert "ValueError: boom" in manager.log_tail(failed)
    monkeypatch.setenv("FAKE_TRAIN", "capped")
    capped = wait_for_end(manager, manager.start({"model": "m"})["id"])
    assert capped["status"] == "capped"
    assert manager.summary(capped)["capped"] == "max_minutes (1) reached"


def test_concurrency_cap_and_stop(manager, monkeypatch):
    monkeypatch.setenv("FAKE_TRAIN", "hang")
    run = manager.start({"model": "m"})
    with pytest.raises(RuntimeError, match="1 of 1 concurrent runs"):
        manager.start({"model": "m"})
    assert manager.usage()["active_runs"] == 1
    stopped = manager.stop(run["id"], save_s=0)
    assert stopped["status"] == "stopped"
    with pytest.raises(ProcessLookupError):
        os.killpg(run["pid"], 0)
    assert manager.usage()["active_runs"] == 0
    assert manager.stop(run["id"])["status"] == "stopped"  # stopping twice is fine


def test_restarted_manager_escalates_through_run_fifo(manager, monkeypatch):
    monkeypatch.setenv("FAKE_TRAIN", "ignore-term")
    run = manager.start({"model": "m"})
    run_dir = Path(run["run_dir"])
    supervisor = manager._procs[run["id"]]
    try:
        deadline = time.monotonic() + 5
        while not (run_dir / "ready").exists():
            assert time.monotonic() < deadline
            time.sleep(0.01)
        # The persisted PID is advisory. Even after restart the supervisor
        # addresses its own child, including escalation past ignored SIGTERM.
        run["pid"] = None
        manager._save(run)
        restarted = RunManager(manager.root)
        stopped = restarted.stop(run["id"], save_s=0, wait_s=1)
        assert stopped["status"] == "stopped"
        assert stopped["exit_code"] == 128 + signal.SIGKILL
    finally:
        manager.stop(run["id"], save_s=0, wait_s=1)
        supervisor.wait(timeout=5)


def test_stale_run_cannot_signal_a_persisted_pid(manager):
    run = wait_for_end(manager, manager.start({"model": "m"})["id"])
    (Path(run["run_dir"]) / "exit_code").unlink()
    restarted = RunManager(manager.root)
    with patch("os.killpg") as signal_group:
        for legacy in (False, True):
            if legacy:
                run.pop("control_fifo")
            run.update(status="running", pid=os.getpid())
            manager._save(run)
            assert restarted.stop(run["id"], save_s=0)["status"] == "failed"
        signal_group.assert_not_called()


def test_daily_run_cap(manager):
    manager.limits = Limits(max_runs_per_day=1)
    wait_for_end(manager, manager.start({"model": "m"})["id"])
    with pytest.raises(RuntimeError, match="last 24 hours"):
        manager.start({"model": "m"})
    assert manager.usage()["runs_last_24h"] == 1


@pytest.mark.parametrize("legacy", [False, True])
def test_deleted_run_counts_until_its_launch_expires(manager, legacy):
    from datetime import datetime, timedelta, timezone

    manager.limits = Limits(max_runs_per_day=1)
    launched = datetime(2026, 11, 1, 8, 30, tzinfo=timezone.utc)
    with patch("mlx_vlm.train_runs._now", return_value=launched):
        run = wait_for_end(manager, manager.start({"model": "m"})["id"])
        if legacy:
            (manager.root / "launches.json").unlink()
        manager.delete(run["id"])
    restarted = RunManager(manager.root, manager.limits, python=manager.python)
    with patch(
        "mlx_vlm.train_runs._now",
        return_value=launched + timedelta(hours=23, minutes=59),
    ):
        assert restarted.usage()["runs_last_24h"] == 1
        with pytest.raises(RuntimeError, match="last 24 hours"):
            restarted.start({"model": "m"})
    with patch("mlx_vlm.train_runs._now", return_value=launched + timedelta(hours=24)):
        assert restarted.usage()["runs_last_24h"] == 0
        wait_for_end(restarted, restarted.start({"model": "m"})["id"])
        assert restarted.usage()["runs_last_24h"] == 1


def test_invalid_options_start_nothing(tmp_path):
    manager = RunManager(tmp_path / "runs", Limits())
    with pytest.raises(ValueError, match="Unknown config options"):
        manager.start({"model": "m", "bogus": True})
    assert manager.runs() == []


def test_limits_come_from_environment_and_flags(monkeypatch):
    monkeypatch.setenv("MLX_VLM_TRAIN_MAX_ITERS", "100")
    monkeypatch.setenv("MLX_VLM_TRAIN_MAX_MINUTES", "30")
    limits = Limits.from_env(max_minutes=5.5)
    assert limits == Limits(max_concurrent_runs=1, max_iters=100, max_minutes=5.5)
    assert isinstance(limits.max_iters, int)
    with pytest.raises(ValueError):
        Limits.from_env(max_iters=0)


def test_local_paths_become_absolute(tmp_path):
    (tmp_path / "data").mkdir()
    options = {
        "data": "data",
        "model": "org/name",
        "export_qat_path": "./out",
        "iters": 5,
        "new_reference_file": "~/x.safetensors",
    }
    resolved = resolve_paths(options, tmp_path)
    assert resolved["data"] == str((tmp_path / "data").resolve())
    assert resolved["model"] == "org/name"  # a Hub id stays as it is
    assert resolved["export_qat_path"] == str((tmp_path / "out").resolve())
    assert os.path.isabs(resolved["new_reference_file"])
    assert resolved["iters"] == 5


def _tiny_training_cli():
    import numpy as np

    from mlx_vlm.tests.test_training_cli import TinyVLM

    def prepare(**kwargs):
        return {"input_ids": np.array([[1, 2, 3, 4]])}

    with (
        patch.object(
            train_cli,
            "_load_model",
            return_value=(TinyVLM(), SimpleNamespace(pad_token_id=0)),
        ),
        patch(
            "mlx_vlm.trainer.vlm.cross_entropy.dataset._load_vlm_helpers",
            return_value=({}, None, prepare, None),
        ),
    ):
        train_cli.main()


def test_export_after_continuation_survives_run_deletion(tmp_path, monkeypatch):
    import mlx.core as mx

    monkeypatch.chdir(tmp_path)
    Path("base").mkdir()
    Path("base/config.json").write_text('{"model_type":"tiny"}')
    Path("train.jsonl").write_text('{"text":"sample"}\n')
    manager = RunManager(tmp_path / "runs", Limits(max_minutes=0.000001))
    monkeypatch.setattr(
        manager,
        "_train_command",
        lambda *args: [
            sys.executable,
            "-c",
            "from mlx_vlm.tests.test_train_runs import _tiny_training_cli; _tiny_training_cli()",
            *args,
        ],
    )
    destination = tmp_path / "models" / "final"
    run = manager.start(
        dict(
            model="base",
            data="train.jsonl",
            train_type="full",
            iters=2,
            steps_per_report=1,
            export_qat_path="models/final",
        )
    )
    capped = wait_for_end(manager, run["id"], timeout=90)
    assert capped["status"] == "capped", manager.log_tail(capped)
    assert capped["export_path"] == str(destination) and not destination.exists()
    # The cap also expires during the final iteration of the continuation.
    finished = wait_for_end(manager, manager.resume(run["id"])["id"], timeout=90)
    assert finished["status"] == "completed", manager.log_tail(finished)
    weights = mx.load(str(manager.weights(finished)))
    exported = mx.load(str(destination / "model.safetensors"))
    assert exported.keys() == weights.keys()
    assert all(mx.array_equal(v, exported[k]).item() for k, v in weights.items())
    manager.delete(capped["id"])
    manager.delete(finished["id"])
    assert (destination / "model.safetensors").is_file()


def test_inspect_dataset_reports_splits(tmp_path):
    (tmp_path / "train.jsonl").write_text('{"question": "q", "answer": "a"}\n' * 3)
    (tmp_path / "valid.jsonl").write_text('{"question": "q", "answer": "a"}\n')
    report = inspect_dataset({"data": str(tmp_path)})
    assert report["splits"]["train"] == {
        "rows": 3,
        "columns": ["answer", "question"],
        "format": "question_answer",
        "algorithms": ["sft"],
        "sampled_rows": 3,
        "text_chars": {"median": 2, "p95": 2, "max": 2},
        "rows_with_images": 0,
    }
    assert report["splits"]["valid"]["rows"] == 1
    with pytest.raises(ValueError, match="no non-empty"):
        inspect_dataset({"data": str(tmp_path), "train_split": "missing"})


def test_inspection_fetches_a_bounded_sample_without_retaining_media():
    import weakref

    from mlx_vlm.train_runs import _describe_split

    class Image:
        pass

    class LazyRows:
        def __init__(self):
            self.indices = []
            self.live_images = weakref.WeakSet()

        def __len__(self):
            return 3999

        def __getitem__(self, index):
            self.indices.append(index)
            image = Image()
            self.live_images.add(image)
            assert len(self.live_images) <= 2
            return {"text": "abc", "image": image}

    rows = LazyRows()
    report = _describe_split(rows)
    assert report["sampled_rows"] == report["rows_with_images"] == 2000
    assert report["text_chars"] == {"median": 3, "p95": 3, "max": 3}
    assert rows.indices == list(range(0, 3999, 2))


def test_command_line_prints_json_and_reports_errors(tmp_path, capsys):
    main(["--runs-dir", str(tmp_path), "--max-runs-per-day", "3", "limits"])
    usage = json.loads(capsys.readouterr().out)
    assert usage["limits"]["max_runs_per_day"] == 3 and usage["active_runs"] == 0
    main(["--runs-dir", str(tmp_path), "list"])
    assert json.loads(capsys.readouterr().out) == {"runs": []}
    with pytest.raises(SystemExit) as stopped:
        main(["--runs-dir", str(tmp_path), "status"])
    assert stopped.value.code == 1
    error = json.loads(capsys.readouterr().out)
    assert error == {"error": "No training runs yet.", "type": "LookupError"}
    assert SimpleNamespace(**error).type == "LookupError"


def test_limits_file_binds_every_client_of_the_runs_folder(tmp_path, monkeypatch):
    for name in ("MAX_ITERS", "MAX_MINUTES", "MAX_CONCURRENT_RUNS"):
        monkeypatch.delenv("MLX_VLM_TRAIN_" + name, raising=False)
    assert Limits.load(tmp_path) == Limits()  # no file: defaults
    limits_file = tmp_path / "limits.json"
    limits_file.write_text(
        json.dumps({"max_iters": 100, "max_minutes": 30, "max_concurrent_runs": None})
    )
    assert Limits.load(tmp_path) == Limits(
        max_concurrent_runs=None, max_iters=100, max_minutes=30
    )
    # A flag can tighten a cap from the file but never loosen it.
    assert Limits.load(tmp_path, max_iters=50).max_iters == 50
    assert Limits.load(tmp_path, max_iters=5000).max_iters == 100
    assert Limits.load(tmp_path, max_concurrent_runs=2).max_concurrent_runs == 2
    assert RunManager(tmp_path).limits.max_minutes == 30
    limits_file.write_text(json.dumps({"max_iters": -1}))
    with pytest.raises(ValueError):
        Limits.load(tmp_path)


def test_history_returns_every_reported_metric_as_a_series(manager):
    run = wait_for_end(manager, manager.start({"model": "m", "algorithm": "sft"})["id"])
    progress = manager.root / run["id"] / "progress.jsonl"
    events = [{"event": "start", "iters": 1000}]
    for step in range(1, 1001):
        events.append(
            {
                "event": "train",
                "iteration": step,
                "train_loss": 1 / step,
                "train_reward_margin": step * 0.5,  # an algorithm-specific metric
                "flag": True,
                "note": "text",
                "broken": float("nan"),
            }
        )
        if step % 250 == 0:
            events.append({"event": "validation", "iteration": step, "val_loss": 2.0})
    progress.write_text("\n".join(json.dumps(event) for event in events) + "\n")
    history = manager.history(run, max_points=50)
    assert history["iters"] == 1000 and history["algorithm"] == "sft"
    assert set(history["series"]) == {"train_loss", "train_reward_margin", "val_loss"}
    loss = history["series"]["train_loss"]
    assert len(loss) == 50 and loss[0] == [1, 1.0] and loss[-1][0] == 1000
    assert history["series"]["val_loss"] == [
        [250, 2.0],
        [500, 2.0],
        [750, 2.0],
        [1000, 2.0],
    ]
    json.dumps(history)


def test_concurrent_saves_of_one_run_do_not_collide(manager):
    """Polls from several processes settle a finished run at the same moment."""
    from concurrent.futures import ThreadPoolExecutor

    run = wait_for_end(manager, manager.start({"model": "m", "algorithm": "sft"})["id"])
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(lambda _: manager._save(run), range(400)))
    assert manager.get(run["id"])["status"] == "completed"
    assert not list((manager.root / run["id"]).glob("*.tmp"))


def test_long_progress_log_and_nonfinite_metrics(manager):
    run = wait_for_end(manager, manager.start({"model": "m"})["id"])
    path = Path(run["progress_path"])
    path.write_text("")
    callback = ProgressCallback(path)
    callback.write("start", iters=1000)
    callback.write("train", iteration=1, note="x" * 260_000)
    callback.write("train", iteration=2, loss=float("inf"), perplexity=float("nan"))
    assert json.loads(path.read_text().splitlines()[-1])["loss"] is None
    # Older progress files used Python's nonstandard JSON constants.
    with path.open("a") as log:
        log.write('{"event":"train","iteration":3,"loss":-Infinity,"perplexity":NaN}\n')
    status = manager.summary(run)
    assert status["iters"] == 1000 and manager.remaining(run) == 997
    assert status["metrics"]["loss"] is None and status["metrics"]["perplexity"] is None
    history = manager.history(run)
    assert history["iters"] == 1000 and "loss" not in history["series"]
    json.dumps([status, history], allow_nan=False)


def test_describe_reports_which_optimizers_read_an_option():
    options = {
        option["name"]: option
        for option in describe()["tasks"]["vlm"]["algorithms"]["sft"]["options"]
    }
    assert options["beta1"]["requires"] == {"optimizer": ["adam", "adamw"]}
    assert options["weight_decay"]["requires"] == {"optimizer": ["adamw", "muon"]}
    assert options["learning_rate"]["requires"] is None
    assert set(options["beta1"]["requires"]["optimizer"]) <= set(
        options["optimizer"]["choices"]
    )


def test_stop_lets_the_run_save_and_it_can_be_continued(manager, monkeypatch):
    monkeypatch.setenv("FAKE_TRAIN", "wait")
    first = manager.start({"model": "m", "algorithm": "sft", "iters": 10}, "job")
    stopped = manager.stop(first["id"], save_s=10)
    assert stopped["status"] == "stopped" and stopped["exit_code"] == 0
    summary = manager.summary(stopped)
    assert summary["can_continue"] and summary["iteration"] == 5

    monkeypatch.delenv("FAKE_TRAIN")
    second = manager.resume(first["id"])
    assert (
        second["continued_from"] == first["id"] and second["name"] == "job (continued)"
    )
    config = json.loads((manager.root / second["id"] / "config.json").read_text())
    adapters = manager.root / first["id"] / "adapters"
    # The whole run's iterations are kept; the trainer works out what is left.
    assert config["iters"] == 10
    assert config["resume_adapter_file"] == str(adapters / "adapters.safetensors")
    assert config["resume_state"] == str(adapters / "trainer_state.safetensors")
    assert manager.summary(wait_for_end(manager, second["id"]))["can_continue"] is False


def test_continue_without_saved_state_restarts_from_the_weights(manager, monkeypatch):
    monkeypatch.setenv("FAKE_TRAIN", "fail")
    failed = wait_for_end(
        manager, manager.start({"model": "m", "algorithm": "sft"})["id"]
    )
    with pytest.raises(RuntimeError, match="saved no weights"):
        manager.resume(failed["id"])
    (manager.root / failed["id"] / "adapters" / "adapters.safetensors").touch()
    monkeypatch.delenv("FAKE_TRAIN")
    resumed = manager.resume(failed["id"])
    config = json.loads((manager.root / resumed["id"] / "config.json").read_text())
    assert config["iters"] == 5 and "resume_state" not in config
    assert "fresh optimizer" in resumed["notes"][-1]


def test_options_of_a_run_leave_out_what_the_manager_set(manager):
    run = manager.start({"model": "m", "algorithm": "sft", "iters": 7})
    assert manager.options(run) == {"model": "m", "algorithm": "sft", "iters": 7}


def test_failed_continuation_drops_ancestor_state(manager, monkeypatch):
    from mlx_vlm.tests.test_training_cli import TinyVLM
    from mlx_vlm.trainer.common.utils import save_trainable_weights

    monkeypatch.setenv("FAKE_TRAIN", "wait")
    first = manager.stop(manager.start({"model": "m", "iters": 10})["id"])
    monkeypatch.setenv("FAKE_TRAIN", "fail")
    second = wait_for_end(manager, manager.resume(first["id"])["id"])
    weights = Path(second["adapter_path"]) / "adapters.safetensors"
    save_trainable_weights(TinyVLM(), weights, iteration=2)
    progress = Path(second["progress_path"])
    progress.write_text(
        '{"event":"start","iters":10,"resumed_at":5}\n'
        '{"event":"train","iteration":9}\n'
    )
    monkeypatch.delenv("FAKE_TRAIN")
    third = wait_for_end(manager, manager.resume(second["id"])["id"])
    options = manager.options(third)
    assert options["resume_adapter_file"] == str(weights)
    assert "resume_state" not in options and options["iters"] == 3
    Path(third["progress_path"]).unlink()
    with pytest.raises(RuntimeError, match="already finished"):
        manager.resume(third["id"])


def test_delete_removes_an_ended_run_only(manager, monkeypatch):
    monkeypatch.setenv("FAKE_TRAIN", "hang")
    run = manager.start({"model": "m", "algorithm": "sft"})
    with pytest.raises(RuntimeError, match="still running"):
        manager.delete(run["id"])
    manager.stop(run["id"], save_s=0)
    assert manager.delete(run["id"]) == {"deleted": run["id"]}
    assert not (manager.root / run["id"]).exists() and manager.runs() == []


def test_sample_returns_the_latest_generation(manager, tmp_path):
    from mlx_vlm.train import record_samples

    run = wait_for_end(manager, manager.start({"model": "m", "algorithm": "sft"})["id"])
    assert manager.sample(run)["sample"] is None

    def exact(prompts, completions, answer, **_):
        return [1.0 if c == a else 0.0 for c, a in zip(completions, answer)]

    reward = record_samples(exact, manager.root / run["id"] / "last_generation.json")
    assert reward.__name__ == "exact"
    assert reward(prompts=["q"], completions=["x" * 9000], answer=["y"]) == [0.0]
    assert reward(prompts=["q2"], completions=["a"], answer=["a"]) == [1.0]
    sample = manager.sample(run)["sample"]
    assert sample["prompt"] == "q2" and sample["completion"] == "a"
    assert sample["reward"] == 1.0 and sample["group"] == 2


def test_describe_offers_schedules_as_choices_with_their_options():
    options = {
        option["name"]: option
        for option in describe()["tasks"]["vlm"]["algorithms"]["sft"]["options"]
    }
    assert options["lr_schedule"]["choices"] == ["cosine_decay", "linear_schedule"]
    assert options["min_learning_rate"]["requires"] == {
        "lr_schedule": ["cosine_decay", "linear_schedule"]
    }
    assert options["resume_state"]["group"] == "managed"
    for schedule in options["lr_schedule"]["choices"]:
        parse_args(["--model", "m", "--lr-schedule", schedule, "--warmup-steps", "2"])
