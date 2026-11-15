# Copyright © 2026 Neywa Labs Inc.

"""Own the lifetime of an optional mlx.launch worker group."""

import asyncio
import fcntl
import json
import logging
import math
import os
import re
import secrets
import shlex
import signal
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path

from ._worker_supervisor import worker_environment

logger = logging.getLogger("mlx_vlm.server.distributed")


@dataclass(frozen=True)
class LaunchConfig:
    hostfile: Path
    cwd: str
    backend: str = "jaccl"
    host: str = "0.0.0.0"
    port: int = 8283
    starting_port: int = 32323
    max_num_seqs: int = 8
    startup_timeout: float = 120
    shutdown_timeout: float = 10
    env: dict = field(default_factory=dict)
    fast_sync: bool = False

    @classmethod
    def read(cls, data, base):
        if not isinstance(data, dict):
            raise ValueError("launch must be an object")
        unknown = set(data) - cls.__dataclass_fields__.keys()
        if unknown:
            raise ValueError(f"Unknown launch settings: {sorted(unknown)}")
        hostfile = (base / Path(data["hostfile"]).expanduser()).resolve()
        cwd = data["cwd"]
        if not isinstance(cwd, str) or not Path(cwd).is_absolute():
            raise ValueError("launch.cwd must be an absolute path on every Mac")
        config = cls(**{**data, "hostfile": hostfile})
        if type(config.fast_sync) is not bool:
            raise ValueError("launch.fast_sync must be a boolean")
        if config.backend not in ("jaccl", "ring"):
            raise ValueError("Automatic launch supports jaccl or ring")
        if not isinstance(config.host, str) or not config.host:
            raise ValueError("launch.host must be a nonempty bind address")
        for name in ("port", "starting_port", "max_num_seqs"):
            value = getattr(config, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"launch.{name} must be a positive integer")
        for name in ("port", "starting_port"):
            if getattr(config, name) > 65535:
                raise ValueError(f"launch.{name} must be at most 65535")
        for name in ("startup_timeout", "shutdown_timeout"):
            value = getattr(config, name)
            if isinstance(value, bool) or not isinstance(value, (float, int)):
                raise ValueError(f"launch.{name} must be a positive finite number")
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"launch.{name} must be a positive finite number")
        if not isinstance(config.env, dict) or any(
            not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key)
            or not isinstance(value, str)
            or "\x00" in value
            for key, value in config.env.items()
        ):
            raise ValueError("launch.env must map environment names to strings")
        if "MLX_METAL_FAST_SYNCH" in config.env:
            raise ValueError(
                "Use launch.fast_sync instead of launch.env.MLX_METAL_FAST_SYNCH"
            )
        if any(
            key.startswith(
                ("MLX_VLM_PRELOAD_", "MLX_VLM_DRAFT_", "MLX_JACCL_", "MLX_RING_")
            )
            or key
            in {
                "MLX_VLM_SERVER_API_KEY",
                "MLX_VLM_WORKER_LAUNCH_ID",
                "MLX_RANK",
                "MLX_WORLD_SIZE",
                "MLX_HOSTFILE",
                "MLX_JACCL_COORDINATOR",
                "MLX_JACCL_DEVICES",
            }
            for key in config.env
        ):
            raise ValueError("launch.env cannot override worker identity or preloads")
        return config

    def validate_hosts(self, workers):
        data = json.loads(self.hostfile.read_text())
        hosts = data.get("hosts") if isinstance(data, dict) else data
        if not isinstance(hosts, list) or len(hosts) != len(workers):
            raise ValueError("launch.hostfile must have one entry per worker")
        if not all(isinstance(host, dict) for host in hosts):
            raise ValueError("launch.hostfile entries must be objects")
        names = [host.get("ssh") for host in hosts]
        expected = [worker.ssh or "127.0.0.1" for worker in workers]
        if names != expected or len(set(names)) != len(names):
            raise ValueError(
                "launch.hostfile SSH entries must match workers in order, with "
                "one worker per host; use 127.0.0.1 for the local worker"
            )


class WorkerLauncher:
    def __init__(self, coordinator):
        self.coordinator = coordinator
        self.config = coordinator.config
        self.launch = self.config.launch
        self.process = None
        self.lock = None
        self.heartbeat = None
        self.output = None
        self.ready_ranks = set()
        self.ssh_directory = None
        self.stop_lock = threading.Lock()
        self.closed = False
        self.stopping = threading.Event()
        self.launch_id = secrets.token_hex(16)
        self.monitor_task = None

    def _payload(self):
        key = os.environ.get(self.config.api_key_env)
        if not key:
            raise ValueError(f"Set {self.config.api_key_env} before launching workers")
        env = {
            **self.launch.env,
            # The supervisor applies this before spawning each worker Python.
            # Explicit 0 also overrides inherited host/hostfile tuning.
            "MLX_METAL_FAST_SYNCH": "1" if self.launch.fast_sync else "0",
            "MLX_VLM_SERVER_API_KEY": key,
            "MLX_VLM_WORKER_LAUNCH_ID": self.launch_id,
        }
        if self.config.compaction_key_file:
            env["MLX_VLM_COMPACTION_KEY_FILE"] = str(self.config.compaction_key_file)
        args = [
            "-m",
            "mlx_vlm.server",
            "--distributed-worker",
            "--distributed-backend",
            self.launch.backend,
            "--distributed-model-dir",
            str(self.config.model_dir),
            "--max-num-seqs",
            str(self.launch.max_num_seqs),
            "--host",
            self.launch.host,
            "--port",
            str(self.launch.port),
        ]
        payload = (
            json.dumps(
                {
                    "args": args,
                    "env": env,
                    "cwd": self.launch.cwd,
                    "shutdown_timeout": self.launch.shutdown_timeout,
                }
            ).encode()
            + b"\n"
        )
        # Keep the initial control message small enough for a single pipe write.
        if len(payload) > 4000:
            raise ValueError("Worker launch configuration exceeds the 4000-byte limit")
        return payload

    def _command(self):
        bootstrap = Path(__file__).with_name("_worker_supervisor.py").read_text()
        # Use the installed mlx.launch entry point in this Python environment.
        entrypoint = (
            "from importlib.metadata import distribution; "
            "next(e for e in distribution('mlx').entry_points "
            "if e.name == 'mlx.launch').load()()"
        )
        return [
            sys.executable,
            "-c",
            entrypoint,
            "--hostfile",
            str(self.launch.hostfile),
            "--backend",
            self.launch.backend,
            "--starting-port",
            str(self.launch.starting_port),
            "--python",
            self.config.python,
            "--cwd",
            self.launch.cwd,
            "--",
            "-c",
            bootstrap,
            self.launch_id,
            str(self.launch.startup_timeout),
        ]

    def _environment(self):
        # mlx.launch invokes ssh by name. Scope these options to this launch;
        # never edit the user's SSH config or ask interactively during startup.
        self.ssh_directory = tempfile.TemporaryDirectory(prefix="mlx-vlm-launch-")
        wrapper = Path(self.ssh_directory.name) / "ssh"
        command = [
            "/usr/bin/ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=yes",
            "-o",
            "ConnectTimeout=10",
            *self.config.ssh_options,
        ]
        wrapper.write_text("#!/bin/sh\nexec " + shlex.join(command) + ' "$@"\n')
        wrapper.chmod(0o700)
        env = worker_environment(os.environ)
        env["PATH"] = self.ssh_directory.name + os.pathsep + env.get("PATH", os.defpath)
        return env

    def _read_output(self):
        prefix = f"MLX_VLM_SUPERVISOR_READY:{self.launch_id}:"
        for raw in self.process.stdout:
            line = raw.decode(errors="replace").strip()
            if line.startswith(prefix):
                try:
                    self.ready_ranks.add(int(line[len(prefix) :]))
                except ValueError:
                    pass
            else:
                logger.info("mlx.launch: %s", line)

    def _renew(self):
        # Independent of the event loop: local model loading may block it.
        while not self.stopping.wait(5):
            try:
                os.write(self.process.stdin.fileno(), b"ping\n")
            except (OSError, ValueError):
                return  # Supervisors expire their leases if transport is lost.

    async def start(
        self, *, recover=False, progress=None, fast_sync: bool | None = None
    ):
        import httpx

        if fast_sync is not None:
            # Keep the configured default intact for the next managed launch.
            self.launch = replace(self.launch, fast_sync=fast_sync)
        started = time.monotonic()
        reported = None

        async def report(stage):
            nonlocal reported
            ranks = sorted(self.ready_ranks)
            now = time.monotonic()
            if progress and (
                reported is None
                or (stage, ranks) != reported[:2]
                or now - reported[2] >= 0.5
            ):
                await progress(
                    stage=stage,
                    elapsed_seconds=now - started,
                    total_workers=len(self.config.workers),
                    supervisors_ready=len(ranks),
                    workers_ready=len(ranks) if stage == "ready" else 0,
                    nodes=[
                        {
                            "node_id": worker.node_id,
                            "rank": rank,
                            "stage": (
                                ("ready" if stage == "ready" else "supervisor_ready")
                                if rank in ranks
                                else "waiting_for_supervisor"
                            ),
                        }
                        for rank, worker in enumerate(self.config.workers)
                    ],
                )
                reported = (stage, ranks, now)

        await report("checking_endpoints")
        payload = self._payload()
        self.launch.validate_hosts(self.config.workers)
        if self.coordinator.failure.exists() and not recover:
            raise RuntimeError("Inspect .cluster-failed.json before restarting workers")
        path = Path(str(self.config.shared_lock) + ".launcher")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = path.open("a+")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                "Another server owns this worker group's launcher"
            ) from error

        headers = {"Authorization": f"Bearer {os.environ[self.config.api_key_env]}"}
        async with httpx.AsyncClient(
            headers=headers, timeout=2, trust_env=False
        ) as client:
            # Never adopt or replace an existing service, including a partial group.
            checks = await asyncio.gather(
                *[
                    client.get(worker.url.rstrip("/") + "/health")
                    for worker in self.config.workers
                ],
                return_exceptions=True,
            )
            if any(not isinstance(check, httpx.ConnectError) for check in checks):
                raise RuntimeError(
                    "Worker endpoints are occupied or unreachable without a clear "
                    "connection refusal; inspect them or omit launch for external workers"
                )
            self.coordinator.launch_id = self.launch_id
            logger.info(
                "Starting %d distributed workers via mlx.launch (%s)",
                len(self.config.workers),
                self.launch.backend,
            )
            if self.launch.fast_sync:
                logger.warning(
                    "Experimental distributed fast synchronization is enabled; "
                    "MLX may deadlock. Stop the group and relaunch with "
                    "fast_sync=false if generation stalls."
                )
            self.process = subprocess.Popen(
                self._command(),
                stdin=subprocess.PIPE,
                start_new_session=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=self._environment(),
            )
            self.output = threading.Thread(target=self._read_output, daemon=True)
            self.output.start()
            os.set_blocking(self.process.stdin.fileno(), False)
            deadline = time.monotonic() + self.launch.startup_timeout
            # Do not send secrets while any remote PTY could still echo input.
            expected = set(range(len(self.config.workers)))
            while self.ready_ranks != expected:
                await report("starting_supervisors")
                if self.process.poll() is not None:
                    raise RuntimeError(
                        "mlx.launch exited before supervisors were ready; inspect worker logs"
                    )
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        "Timed out starting worker supervisors; check SSH and Python paths"
                    )
                await asyncio.sleep(0.05)
            os.write(self.process.stdin.fileno(), payload)
            self.heartbeat = threading.Thread(target=self._renew, daemon=True)
            self.heartbeat.start()
            await report("waiting_for_workers")
            last_error = None
            while time.monotonic() < deadline:
                await report("waiting_for_workers")
                if self.process.poll() is not None:
                    raise RuntimeError(
                        "mlx.launch exited before workers were ready; inspect worker logs"
                    )
                try:
                    await asyncio.wait_for(
                        self.coordinator._health(client),
                        timeout=max(0.001, deadline - time.monotonic()),
                    )
                    logger.info("Distributed worker group is ready")
                    await report("ready")
                    self.monitor_task = asyncio.create_task(self.monitor())
                    return
                except Exception as error:
                    last_error = error
                await asyncio.sleep(min(0.25, max(0, deadline - time.monotonic())))
            raise RuntimeError(
                "Timed out waiting for the distributed worker group"
            ) from last_error

    async def monitor(self):
        while self.process.poll() is None:
            await asyncio.sleep(0.5)
        if self.stopping.is_set():
            return  # An explicit API stop is not an unexpected worker failure.
        try:
            self.coordinator.failed(
                RuntimeError("Owned mlx.launch worker group exited")
            )
        finally:
            # Local requests remain available; never restart a collective mid-request.
            await asyncio.to_thread(self.stop)

    async def close(self):
        async def cleanup():
            if self.monitor_task:
                self.monitor_task.cancel()
                await asyncio.gather(self.monitor_task, return_exceptions=True)
            await asyncio.to_thread(self.stop)

        # Shutdown must finish even when startup or an API operation is cancelled.
        cleanup_task = asyncio.create_task(cleanup())
        try:
            await asyncio.shield(cleanup_task)
        except asyncio.CancelledError:
            await cleanup_task
            raise

    def stop(self):
        with self.stop_lock:
            if self.closed:
                return
            self.closed = True
            self._stop()

    def _stop(self):
        self.stopping.set()
        if self.heartbeat:
            self.heartbeat.join(timeout=1)
        try:
            if self.process is not None:
                try:
                    os.write(self.process.stdin.fileno(), b"stop\n")
                except (OSError, ValueError):
                    pass
                try:
                    self.process.wait(timeout=self.launch.shutdown_timeout + 5)
                except subprocess.TimeoutExpired:
                    logger.warning(
                        "Stopping unresponsive owned mlx.launch process group"
                    )
                # Also close any launcher descendants left after an early exit.
                try:
                    os.killpg(self.process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(self.process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    self.process.wait()
                self.process.stdin.close()
                if self.output:
                    self.output.join(timeout=2)
                    if not self.output.is_alive() and hasattr(
                        self.process.stdout, "close"
                    ):
                        self.process.stdout.close()
        finally:
            if self.ssh_directory:
                self.ssh_directory.cleanup()
            if self.lock is not None:
                self.lock.close()
                self.lock = None


@asynccontextmanager
async def managed_workers(coordinator):
    if coordinator is None or coordinator.config.launch is None:
        yield
        return
    launcher = WorkerLauncher(coordinator)
    coordinator.launcher = launcher
    try:
        await launcher.start()
        yield
    finally:
        await launcher.close()
