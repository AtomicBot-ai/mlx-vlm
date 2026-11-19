# Copyright © 2026 Neywa Labs Inc.

"""Standalone stdlib bootstrap, sent as Python source through mlx.launch.

Do not import mlx_vlm here: this supervisor must remain responsive even when a
worker is blocked in a native collective. Configuration (including the worker
key) arrives on stdin, followed by lease renewals or a stop message. EOF, lost
renewals and signals stop only this supervisor's child process group.
"""

import json
import os
import select
import signal
import subprocess
import sys
import termios
import time


def worker_environment(environment):
    # Local preloads, drafts and tuning must not leak into collective workers.
    prefixes = ("MLX_VLM_", "APC_", "KV_")
    names = {
        "PREFILL_STEP_SIZE",
        "MAX_KV_SIZE",
        "QUANTIZED_KV_START",
        "EXPERT_CACHE_GB",
        "TOP_LOGPROBS_K",
        "MLX_TRUST_REMOTE_CODE",
    }
    return {
        key: value
        for key, value in environment.items()
        if not key.startswith(prefixes) and key not in names
    }


def main():
    stopped = False

    def stop(*_):
        nonlocal stopped
        stopped = True

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, stop)

    child = None
    buffer = b""
    deadline = time.monotonic() + float(sys.argv[2])
    shutdown_timeout = 10
    exit_code = 1
    if os.isatty(sys.stdin.fileno()):
        # macOS PTYs have a short canonical line buffer. Read the control stream
        # without line editing or echo, before acknowledging readiness.
        attrs = termios.tcgetattr(sys.stdin.fileno())
        attrs[3] &= ~(termios.ICANON | termios.ECHO)
        attrs[6][termios.VMIN] = 1
        attrs[6][termios.VTIME] = 0
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSANOW, attrs)
    print(
        f"MLX_VLM_SUPERVISOR_READY:{sys.argv[1]}:{os.environ['MLX_RANK']}",
        flush=True,
    )
    try:
        while not stopped and time.monotonic() < deadline:
            if child is not None and child.poll() is not None:
                exit_code = child.returncode or 1  # A worker must stay running.
                break
            if not select.select([sys.stdin], [], [], 0.2)[0]:
                continue
            data = os.read(sys.stdin.fileno(), 4096)
            if not data:
                break
            buffer += data
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                line = line.rstrip(b"\r")  # SSH PTYs may translate newlines.
                if line == b"stop":
                    exit_code = 0
                    stopped = True
                    break
                elif child is None:
                    config = json.loads(line)
                    shutdown_timeout = config["shutdown_timeout"]
                    environment = worker_environment(os.environ)
                    environment.update(config["env"])
                    child = subprocess.Popen(
                        [sys.executable, *config["args"]],
                        cwd=config["cwd"],
                        env=environment,
                        stdin=subprocess.DEVNULL,
                        start_new_session=True,
                    )
                elif line != b"ping":
                    raise ValueError("Invalid worker supervisor message")
                deadline = time.monotonic() + 30
    finally:
        if child is not None:
            # The child owns a fresh session. Never signal a name or other job.
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=shutdown_timeout)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait()
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
