# Copyright © 2026 Neywa Labs Inc.

"""Worker lifecycle tests; no remote hosts, downloads or inference."""

import asyncio
import json
import os
import pty
import select
import signal
import subprocess
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from fastapi.testclient import TestClient

import mlx_vlm.server as server
from mlx_vlm.server import _worker_supervisor, distributed_launch
from mlx_vlm.server.distributed import ClusterConfig, Coordinator, Worker
from mlx_vlm.server.distributed_launch import (
    LaunchConfig,
    WorkerLauncher,
    managed_workers,
)
from mlx_vlm.server.runtime import runtime


@pytest.fixture
def cluster(tmp_path, monkeypatch):
    hosts = tmp_path / "hosts.json"
    hosts.write_text(json.dumps({"hosts": [{"ssh": "127.0.0.1"}, {"ssh": "mac-two"}]}))
    monkeypatch.setenv("MLX_VLM_DISTRIBUTED_API_KEY", "secret-worker-token")
    return ClusterConfig(
        (Worker("http://rank0:8283"), Worker("http://rank1:8283", "mac-two")),
        tmp_path / "models",
        tmp_path / "cluster.lock",
        python=sys.executable,
        launch=LaunchConfig(
            hosts, str(tmp_path), startup_timeout=0.1, shutdown_timeout=0.1
        ),
    )


def _config_file(cluster, **changes):
    config = {
        "workers": [
            {"url": w.url, **({"ssh": w.ssh} if w.ssh else {})} for w in cluster.workers
        ],
        "model_dir": str(cluster.model_dir),
        "shared_lock": str(cluster.shared_lock),
        "python": cluster.python,
        "launch": {"hostfile": "hosts.json", "cwd": cluster.launch.cwd},
        **changes,
    }
    path = cluster.shared_lock.parent / "cluster.json"
    path.write_text(json.dumps(config))
    return path


def test_configuration_resolves_hostfile_and_preserves_external_mode(cluster):
    loaded = ClusterConfig.read(_config_file(cluster))
    assert loaded.launch.hostfile == cluster.launch.hostfile
    assert loaded.launch.max_num_seqs == 8
    path = _config_file(cluster)
    data = json.loads(path.read_text())
    del data["launch"]
    path.write_text(json.dumps(data))
    assert ClusterConfig.read(path).launch is None


@pytest.mark.parametrize(
    "change",
    [
        {"backend": "typo"},
        {"cwd": "relative"},
        {"max_num_seqs": 0},
        {"max_num_seqs": True},
        {"port": 65536},
        {"starting_port": 65536},
        {"starting_port": 0},
        {"startup_timeout": 0},
        {"shutdown_timeout": float("nan")},
        {"extra_setting": "typo"},
        {"env": {"BAD-KEY": "x"}},
        {"env": {"KV_BITS": 4}},
        {"env": {"MLX_VLM_SERVER_API_KEY": "secret"}},
        {"env": {"MLX_VLM_PRELOAD_MODEL": "bad"}},
    ],
)
def test_invalid_launch_configuration_fails_before_start(cluster, change):
    data = {"hostfile": "hosts.json", "cwd": cluster.launch.cwd, **change}
    with pytest.raises(ValueError):
        ClusterConfig.read(_config_file(cluster, launch=data))


def test_hostfile_and_worker_mapping_must_agree(cluster):
    cluster.launch.hostfile.write_text(
        json.dumps([{"ssh": "other"}, {"ssh": "mac-two"}])
    )
    with pytest.raises(ValueError, match="match workers"):
        ClusterConfig.read(_config_file(cluster))


def test_launcher_uses_configured_collective_port(cluster):
    config = replace(cluster, launch=replace(cluster.launch, starting_port=32973))
    command = WorkerLauncher(Coordinator(config))._command()
    assert command[command.index("--starting-port") + 1] == "32973"


def test_worker_payload_is_private_and_local_settings_do_not_leak(cluster, monkeypatch):
    monkeypatch.setenv("MLX_VLM_SERVER_API_KEY", "public-key")
    monkeypatch.setenv("MLX_VLM_PRELOAD_MODEL", "local-only-model")
    monkeypatch.setenv("KV_BITS", "8")
    launch = WorkerLauncher(Coordinator(cluster))
    payload = json.loads(launch._payload())
    assert payload["env"]["MLX_VLM_SERVER_API_KEY"] == "secret-worker-token"
    assert payload["env"]["MLX_VLM_WORKER_LAUNCH_ID"] == launch.launch_id
    assert "secret-worker-token" not in " ".join(launch._command())
    assert "public-key" not in launch._payload().decode()
    env = _worker_supervisor.worker_environment(os.environ)
    assert "MLX_VLM_PRELOAD_MODEL" not in env and "KV_BITS" not in env
    assert _worker_supervisor.worker_environment({"MLX_RANK": "1"}) == {"MLX_RANK": "1"}


@pytest.mark.parametrize("fast_sync", [None, False, True])
def test_fast_sync_configuration_controls_worker_payload(
    cluster, monkeypatch, fast_sync
):
    monkeypatch.setenv("MLX_METAL_FAST_SYNCH", "0" if fast_sync else "1")
    data = {"hostfile": "hosts.json", "cwd": cluster.launch.cwd}
    if fast_sync is not None:
        data["fast_sync"] = fast_sync
    config = ClusterConfig.read(_config_file(cluster, launch=data))
    assert config.launch.fast_sync is (fast_sync is True)
    payload = json.loads(WorkerLauncher(Coordinator(config))._payload())
    assert payload["env"]["MLX_METAL_FAST_SYNCH"] == ("1" if fast_sync else "0")
    assert os.environ["MLX_METAL_FAST_SYNCH"] == ("0" if fast_sync else "1")


@pytest.mark.parametrize("value", [None, 0, 1, "true", "false", "1", [], {}])
def test_fast_sync_requires_an_explicit_boolean(cluster, value):
    data = {"hostfile": "hosts.json", "cwd": cluster.launch.cwd, "fast_sync": value}
    with pytest.raises(ValueError, match="launch.fast_sync must be a boolean"):
        ClusterConfig.read(_config_file(cluster, launch=data))


@pytest.mark.parametrize("value", ["0", "1"])
def test_fast_sync_cannot_be_overridden_by_launch_env(cluster, value):
    data = {
        "hostfile": "hosts.json",
        "cwd": cluster.launch.cwd,
        "env": {"MLX_METAL_FAST_SYNCH": value},
    }
    with pytest.raises(ValueError, match="Use launch.fast_sync instead"):
        ClusterConfig.read(_config_file(cluster, launch=data))


@pytest.mark.parametrize("configured", [False, True])
@pytest.mark.parametrize("override", [None, False, True])
def test_start_fast_sync_override_preserves_configured_default(
    fake_launch, monkeypatch, configured, override
):
    launcher, process, _ = fake_launch
    launcher.config = replace(
        launcher.config, launch=replace(launcher.launch, fast_sync=configured)
    )
    launcher.launch = launcher.config.launch

    def handler(request):
        if launcher.process is None:
            raise httpx.ConnectError("refused", request=request)
        return httpx.Response(
            200, json=_health(int(request.url.host[-1]), launcher.launch_id)
        )

    _transport(monkeypatch, handler)
    asyncio.run(launcher.start(fast_sync=override))
    expected = configured if override is None else override
    assert launcher.launch.fast_sync is expected
    assert launcher.config.launch.fast_sync is configured
    assert json.loads(process.read())["env"]["MLX_METAL_FAST_SYNCH"] == (
        "1" if expected else "0"
    )


def _transport(monkeypatch, handler):
    original = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kw: original(**kw, transport=httpx.MockTransport(handler)),
    )


def _health(rank, launch_id, backend="jaccl"):
    return {
        "distributed_worker": {
            "rank": rank,
            "world_size": 2,
            "backend": backend,
            "fingerprint": "same-build",
            "launch_id": launch_id,
        }
    }


class FakeProcess:
    """A launched process with a real control pipe, avoiding remote execution."""

    def __init__(self, launcher):
        self.returncode = None
        self.pid = 123456789  # killpg is mocked for this test double.
        self.read_fd, write_fd = os.pipe()
        self.stdin = os.fdopen(write_fd, "wb", buffering=0)
        self.stdout = []
        self.launcher = launcher

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.returncode = 0
        return 0

    def read(self):
        return os.read(self.read_fd, 8192)


@pytest.fixture
def fake_launch(cluster, monkeypatch):
    launcher = WorkerLauncher(Coordinator(cluster))
    process = FakeProcess(launcher)
    monkeypatch.setattr(os, "killpg", Mock())

    def popen(*args, **kwargs):
        launcher.ready_ranks.update({0, 1})
        return process

    factory = Mock(side_effect=popen)
    monkeypatch.setattr(distributed_launch.subprocess, "Popen", factory)
    yield launcher, process, factory
    launcher.stop()
    os.close(process.read_fd)


def test_start_waits_for_matching_health_and_sends_key_on_stdin(
    fake_launch, monkeypatch
):
    launcher, process, factory = fake_launch

    def handler(request):
        assert request.headers["Authorization"] == "Bearer secret-worker-token"
        if launcher.process is None:
            raise httpx.ConnectError("refused", request=request)
        return httpx.Response(
            200, json=_health(int(request.url.host[-1]), launcher.launch_id)
        )

    _transport(monkeypatch, handler)
    progress = []

    async def report(**data):
        progress.append(data)

    async def run():
        await launcher.start(progress=report)
        assert launcher.monitor_task is not None and not launcher.monitor_task.done()

    asyncio.run(run())
    factory.assert_called_once()
    assert (
        json.loads(process.read())["env"]["MLX_VLM_SERVER_API_KEY"]
        == "secret-worker-token"
    )
    assert launcher.coordinator.primary == 0
    assert launcher.coordinator.launch_id == launcher.launch_id
    assert progress[0]["stage"] == "checking_endpoints"
    assert progress[0]["workers_ready"] == 0
    assert progress[-1]["stage"] == "ready"
    assert progress[-1]["supervisors_ready"] == progress[-1]["workers_ready"] == 2
    assert all(p["total_workers"] == 2 for p in progress)


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, json={"other_service": True}),
        httpx.Response(401),
        httpx.ReadTimeout("cannot establish absence"),
    ],
)
def test_occupied_or_uncertain_endpoints_are_never_replaced(
    fake_launch, monkeypatch, response
):
    launcher, _, factory = fake_launch

    def handler(request):
        if isinstance(response, Exception):
            raise response
        return response

    _transport(monkeypatch, handler)
    with pytest.raises(RuntimeError, match="occupied"):
        asyncio.run(launcher.start())
    factory.assert_not_called()


@pytest.mark.parametrize(
    "backend,identity", [("jaccl", "other-launch"), ("ring", "correct")]
)
def test_readiness_rejects_wrong_launch_or_backend(
    fake_launch, monkeypatch, backend, identity
):
    launcher, _, _ = fake_launch

    def handler(request):
        if launcher.process is None:
            raise httpx.ConnectError("refused", request=request)
        return httpx.Response(
            200,
            json=_health(
                int(request.url.host[-1]),
                launcher.launch_id if identity == "correct" else identity,
                backend,
            ),
        )

    _transport(monkeypatch, handler)
    with pytest.raises(RuntimeError, match="Timed out waiting"):
        asyncio.run(launcher.start())


def test_second_owner_cannot_launch(cluster, fake_launch, monkeypatch):
    first, _, factory = fake_launch
    first.lock = Path(str(cluster.shared_lock) + ".launcher").open("a+")
    distributed_launch.fcntl.flock(first.lock, distributed_launch.fcntl.LOCK_EX)
    second = WorkerLauncher(Coordinator(cluster))
    try:
        with pytest.raises(RuntimeError, match="Another server owns"):
            asyncio.run(second.start())
        factory.assert_not_called()
    finally:
        second.stop()


@pytest.mark.parametrize(
    "failure", [RuntimeError("startup failed"), asyncio.CancelledError()]
)
def test_lifespan_rolls_back_failed_or_cancelled_startup(cluster, monkeypatch, failure):
    instance = WorkerLauncher(Coordinator(cluster))
    monkeypatch.setattr(instance, "start", AsyncMock(side_effect=failure))
    monkeypatch.setattr(instance, "_stop", Mock())
    monkeypatch.setattr(
        distributed_launch, "WorkerLauncher", Mock(return_value=instance)
    )

    async def run():
        async with managed_workers(Coordinator(cluster)):
            pytest.fail("must not serve before workers are ready")

    with pytest.raises(type(failure)):
        asyncio.run(run())
    instance._stop.assert_called_once()


def test_fastapi_lifespan_starts_and_stops_workers(cluster, monkeypatch):
    instance = WorkerLauncher(Coordinator(cluster))
    monkeypatch.setattr(instance, "start", AsyncMock())
    monkeypatch.setattr(instance, "_stop", Mock())
    monkeypatch.setattr(
        distributed_launch, "WorkerLauncher", Mock(return_value=instance)
    )
    monkeypatch.setattr(runtime, "distributed_coordinator", Coordinator(cluster))
    with TestClient(server.app):
        instance.start.assert_awaited_once()
        instance._stop.assert_not_called()
    instance._stop.assert_called_once()


def test_cancelled_close_waits_for_owned_process_cleanup(cluster, monkeypatch):
    launcher = WorkerLauncher(Coordinator(cluster))
    entered, release, stopped = (threading.Event() for _ in range(3))

    def stop():
        entered.set()
        assert release.wait(timeout=5)
        stopped.set()

    monkeypatch.setattr(launcher, "_stop", stop)

    async def run():
        launcher.monitor_task = asyncio.create_task(asyncio.Event().wait())
        closing = asyncio.create_task(launcher.close())
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            closing.cancel()
            await asyncio.sleep(0)
            assert not closing.done() and launcher.monitor_task.done()
        finally:
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await closing
        assert stopped.is_set()
        await launcher.close()  # Repeated teardown remains safe.

    asyncio.run(run())


def test_external_and_local_servers_do_not_launch_workers(cluster, monkeypatch):
    factory = Mock()
    monkeypatch.setattr(distributed_launch, "WorkerLauncher", factory)

    async def run():
        for coordinator in (None, Coordinator(replace(cluster, launch=None))):
            async with managed_workers(coordinator):
                pass

    asyncio.run(run())
    factory.assert_not_called()


def test_unexpected_launcher_exit_blocks_distributed_requests(fake_launch):
    launcher, process, _ = fake_launch
    launcher.process = process
    process.returncode = 1
    asyncio.run(launcher.monitor())
    assert "worker group exited" in launcher.coordinator.failure.read_text()
    assert launcher.closed


def _wait_file(path, timeout=5):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if path.exists():
            return json.loads(path.read_text())
        time.sleep(0.02)
    raise AssertionError(f"Worker did not start: {path}")


@pytest.mark.parametrize("stop_mode", ["command", "eof", "signal"])
def test_real_supervisor_stops_its_child_even_if_term_is_ignored(tmp_path, stop_mode):
    record = tmp_path / "child.json"
    code = (
        "import os,json,time,signal,pathlib; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        f"pathlib.Path({str(record)!r}).write_text(json.dumps({{'pid':os.getpid(),"
        "'key':os.environ.get('MLX_VLM_SERVER_API_KEY'),"
        "'preload':os.environ.get('MLX_VLM_PRELOAD_MODEL')})); time.sleep(60)"
    )
    payload = {
        "args": ["-c", code],
        "cwd": str(tmp_path),
        "shutdown_timeout": 0.1,
        "env": {"MLX_VLM_SERVER_API_KEY": "worker-secret"},
    }
    proc = subprocess.Popen(
        [sys.executable, str(Path(_worker_supervisor.__file__)), "test", "5"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**os.environ, "MLX_RANK": "0", "MLX_VLM_PRELOAD_MODEL": "local-model"},
    )
    child = None
    try:
        assert select.select([proc.stdout], [], [], 5)[0]
        assert b"SUPERVISOR_READY:test:0" in proc.stdout.readline()
        proc.stdin.write(json.dumps(payload).encode() + b"\r\n")
        proc.stdin.flush()
        child = _wait_file(record)
        assert child["key"] == "worker-secret" and child["preload"] is None
        if stop_mode == "command":
            proc.stdin.write(b"ping\r\nstop\r\n")
            proc.stdin.flush()
        elif stop_mode == "eof":
            proc.stdin.close()
        else:
            proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=5)
        with pytest.raises(ProcessLookupError):
            os.kill(child["pid"], 0)
        assert b"worker-secret" not in proc.stdout.read() + proc.stderr.read()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        if child:
            try:
                os.kill(child["pid"], signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_native_mlx_launcher_broadcasts_control_to_local_supervisors(cluster, tmp_path):
    """Exercise the real mlx.launch protocol, with two local dummy workers.

    Duplicate localhost ranks are only a launcher test; managed server configs
    require distinct hosts because real workers use the same HTTP bind port.
    """
    cluster.launch.hostfile.write_text(
        json.dumps(
            {
                "hosts": [
                    {"ssh": "127.0.0.1", "ips": ["127.0.0.1"]},
                    {"ssh": "127.0.0.1", "ips": ["127.0.0.1"]},
                ]
            }
        )
    )
    launcher = WorkerLauncher(
        Coordinator(replace(cluster, launch=replace(cluster.launch, backend="ring")))
    )
    record_prefix = str(tmp_path / "rank-")
    code = (
        "import os,json,time,pathlib; "
        f"pathlib.Path({record_prefix!r}+os.environ['MLX_RANK']).write_text("
        "json.dumps({'pid':os.getpid(),'key':os.environ['MLX_VLM_SERVER_API_KEY']})); "
        "time.sleep(60)"
    )
    payload = json.loads(launcher._payload())
    payload["args"] = ["-c", code]
    children = []
    output = b""
    proc = subprocess.Popen(
        launcher._command(),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 10
        while not all(
            f"MLX_VLM_SUPERVISOR_READY:{launcher.launch_id}:{rank}".encode() in output
            for rank in range(2)
        ):
            assert time.monotonic() < deadline, output.decode(errors="replace")
            assert proc.poll() is None, output.decode(errors="replace")
            if select.select([proc.stdout], [], [], 0.1)[0]:
                output += os.read(proc.stdout.fileno(), 8192)
        proc.stdin.write(json.dumps(payload).encode() + b"\n")
        proc.stdin.flush()
        for rank in range(2):
            children.append(_wait_file(Path(record_prefix + str(rank))))
        assert all(child["key"] == "secret-worker-token" for child in children)
        proc.stdin.write(b"stop\n")
        proc.stdin.flush()
        proc.wait(timeout=8)
        output += proc.stdout.read()
        assert b"secret-worker-token" not in output
        for child in children:
            with pytest.raises(ProcessLookupError):
                os.kill(child["pid"], 0)
    finally:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        proc.wait(timeout=5)
        for child in children:
            try:
                os.kill(child["pid"], signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.parametrize("stop_mode", ["stop", "lease_expiry"])
def test_supervisor_accepts_long_pty_messages_without_echo_and_expires_lease(
    tmp_path, stop_mode
):
    record = tmp_path / "child.json"
    code = (
        "import os,json,time,pathlib; "
        f"pathlib.Path({str(record)!r}).write_text(json.dumps("
        "{'pid':os.getpid(),'length':len(os.environ['LONG_VALUE'])})); time.sleep(60)"
    )
    master, slave = pty.openpty()
    proc = subprocess.Popen(
        [sys.executable, _worker_supervisor.__file__, "test", "5"],
        stdin=slave,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**os.environ, "MLX_RANK": "0"},
    )
    os.close(slave)
    child = None
    try:
        assert select.select([proc.stdout], [], [], 5)[0]
        assert b"SUPERVISOR_READY:test:0" in proc.stdout.readline()
        payload = {
            "args": ["-c", code],
            "cwd": str(tmp_path),
            "shutdown_timeout": 0.1,
            "env": {"LONG_VALUE": "x" * 2048},
        }
        os.write(master, json.dumps(payload).encode() + b"\n")
        child = _wait_file(record)
        assert child["length"] == 2048
        assert not select.select([master], [], [], 0.05)[0], "PTY echoed private setup"
        if stop_mode == "stop":
            os.write(master, b"stop\n")
        # Leave the PTY open to simulate a stalled/orphaned SSH launcher.
        proc.wait(timeout=35)
        with pytest.raises(ProcessLookupError):
            os.kill(child["pid"], 0)
    finally:
        os.close(master)
        if proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=5)
        if child:
            try:
                os.kill(child["pid"], signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_monitor_does_not_fail_group_after_explicit_stop():
    from threading import Event
    from types import SimpleNamespace
    from unittest.mock import Mock

    launcher = WorkerLauncher.__new__(WorkerLauncher)
    launcher.process = SimpleNamespace(poll=lambda: 0)
    launcher.stopping = Event()
    launcher.stopping.set()
    launcher.coordinator = SimpleNamespace(failed=Mock())
    asyncio.run(launcher.monitor())
    launcher.coordinator.failed.assert_not_called()


@pytest.mark.parametrize("fast_sync", [None, False, True])
def test_native_launcher_sets_fast_sync_before_mlx_import_on_every_rank(
    cluster, tmp_path, fast_sync
):
    inherited = "0" if fast_sync else "1"
    cluster.launch.hostfile.write_text(
        json.dumps(
            {
                "envs": [f"MLX_METAL_FAST_SYNCH={inherited}"],
                "hosts": [
                    {"ssh": "127.0.0.1", "ips": ["127.0.0.1"]},
                    {"ssh": "127.0.0.1", "ips": ["127.0.0.1"]},
                ],
            }
        )
    )
    options = {} if fast_sync is None else {"fast_sync": fast_sync}
    launcher = WorkerLauncher(
        Coordinator(
            replace(cluster, launch=replace(cluster.launch, backend="ring", **options))
        )
    )
    prefix = str(tmp_path / "fast-sync-rank-")
    code = f"""
import json, os, pathlib, sys, time
assert 'mlx.core' not in sys.modules
before_import = os.environ.get('MLX_METAL_FAST_SYNCH')
import mlx.core
pathlib.Path({prefix!r} + os.environ['MLX_RANK']).write_text(json.dumps({{
    'pid': os.getpid(),
    'before_import': before_import,
    'after_import': os.environ.get('MLX_METAL_FAST_SYNCH'),
}}))
time.sleep(60)
"""
    payload = json.loads(launcher._payload())
    payload["args"] = ["-c", code]
    proc = subprocess.Popen(
        launcher._command(),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env={**os.environ, "MLX_METAL_FAST_SYNCH": inherited},
        start_new_session=True,
    )
    children, output = [], b""
    try:
        deadline = time.monotonic() + 10
        while not all(
            f"MLX_VLM_SUPERVISOR_READY:{launcher.launch_id}:{rank}".encode() in output
            for rank in range(2)
        ):
            assert time.monotonic() < deadline, output.decode(errors="replace")
            assert proc.poll() is None, output.decode(errors="replace")
            if select.select([proc.stdout], [], [], 0.1)[0]:
                output += os.read(proc.stdout.fileno(), 8192)
        proc.stdin.write(json.dumps(payload).encode() + b"\n")
        proc.stdin.flush()
        for rank in range(2):
            children.append(_wait_file(Path(prefix + str(rank))))
        expected = "1" if fast_sync else "0"
        assert all(
            child["before_import"] == child["after_import"] == expected
            for child in children
        )
        proc.stdin.write(b"stop\n")
        proc.stdin.flush()
        proc.wait(timeout=8)
        for child in children:
            with pytest.raises(ProcessLookupError):
                os.kill(child["pid"], 0)
    finally:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        proc.wait(timeout=5)
        for child in children:
            try:
                os.kill(child["pid"], signal.SIGKILL)
            except ProcessLookupError:
                pass
