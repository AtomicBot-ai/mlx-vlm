"""Smoke test of a frozen mlx-server, as Atomic Chat's core uses it.

    python3 -I atomic/ci/smoke.py --binary dist/mlx-server --models-dir ~/mlx-smoke-models

1. `--help` exits 0, prints argparse's `usage: mlx-server` (the core's probe) and
   lists every flag and choice the core passes (atomic-chat-core
   src/runtime/mlx/args.ts).
2. `--selftest-structured` exits 0: llguidance loads inside the bundle.
3. With --models-dir: the server starts on a tiny pinned model with
   MLX_VLM_SINGLE_MODEL=1, as the core starts it, answers a chat completion
   that names another model label, and answers a json_schema response_format
   with JSON. This is what catches a module PyInstaller left out: mlx-vlm
   imports model code lazily, at load and request time.

The pipeline's agent may not edit this file: weakening it would let a broken
binary through.
"""

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# Flags and choices the core passes to mlx-server.
CORE_FLAGS = [
    "--model",
    "--host",
    "--port",
    "--max-kv-size",
    "--draft-model",
    "--draft-kind",
    "--draft-block-size",
    "--kv-bits",
    "--kv-quant-scheme",
]
CORE_CHOICES = ["dflash", "mtp", "eagle3", "uniform", "turboquant"]

# A 135M llama-architecture model, 141 MB: small enough for any runner.
MODEL_REPO = "mlx-community/SmolLM2-135M-Instruct-8bit"
MODEL_REVISION = "0f0d9b8218915bc34d401e1a340b8c049d300d5e"
MODEL_FILES = ["*.json", "*.safetensors", "*.txt"]

START_TIMEOUT = 300
REQUEST_TIMEOUT = 180


def fail(message: str) -> None:
    print(f"::error title=mlx-server smoke::{message}")
    sys.exit(1)


def run(binary: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(binary), *args], capture_output=True, text=True, timeout=REQUEST_TIMEOUT
    )


def check_help(binary: Path) -> None:
    out = run(binary, "--help")
    if out.returncode != 0:
        fail(f"--help exited {out.returncode}: {out.stderr[-2000:]}")
    text = out.stdout
    if "usage: mlx-server" not in text:
        fail("--help does not print 'usage: mlx-server', the core's probe")
    missing = [f for f in CORE_FLAGS + CORE_CHOICES if f not in text]
    if missing:
        fail(f"--help does not list what the core passes: {missing}")
    print(f"--help OK: {len(CORE_FLAGS)} core flags, {len(CORE_CHOICES)} choices")


def check_selftest(binary: Path) -> None:
    out = run(binary, "--selftest-structured")
    print(out.stdout.strip())
    if out.returncode != 0:
        fail(
            f"--selftest-structured exited {out.returncode}: {out.stdout}{out.stderr[-2000:]}"
        )


def download(models_dir: Path) -> Path:
    from huggingface_hub import snapshot_download

    dest = models_dir / MODEL_REPO.replace("/", "--") / MODEL_REVISION
    snapshot_download(
        MODEL_REPO, revision=MODEL_REVISION, local_dir=dest, allow_patterns=MODEL_FILES
    )
    return dest


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def post(url: str, body: dict) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        fail(
            f"POST {url} answered {exc.code}: {exc.read().decode(errors='replace')[:2000]}"
        )
    raise AssertionError


def check_serving(binary: Path, model: Path, log_path: Path) -> None:
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    env = {**os.environ, "MLX_VLM_SINGLE_MODEL": "1", "HF_HUB_OFFLINE": "1"}
    with open(log_path, "w") as log:
        server = subprocess.Popen(
            [
                str(binary),
                "--model",
                str(model),
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
        )
    try:
        deadline = time.monotonic() + START_TIMEOUT
        while True:
            if server.poll() is not None:
                fail(
                    f"server exited {server.returncode} while starting:\n{log_path.read_text()[-4000:]}"
                )
            try:
                with urllib.request.urlopen(f"{base}/health", timeout=5) as response:
                    if response.status == 200:
                        break
            except (urllib.error.URLError, OSError):
                pass
            if time.monotonic() > deadline:
                fail(
                    f"server did not answer /health in {START_TIMEOUT}s:\n{log_path.read_text()[-4000:]}"
                )
            time.sleep(2)

        messages = [{"role": "user", "content": "Say hello in one word."}]
        # Another label than the loaded model: single-model mode must keep serving it.
        chat = post(
            f"{base}/v1/chat/completions",
            {
                "model": "atomic-smoke",
                "messages": messages,
                "max_tokens": 16,
                "temperature": 0,
            },
        )
        text = chat["choices"][0]["message"].get("content") or ""
        if not text.strip():
            fail(f"empty chat completion: {json.dumps(chat)[:2000]}")
        print(f"chat OK: {text.strip()[:80]!r}")

        schema = {
            "type": "object",
            "properties": {"word": {"type": "string"}},
            "required": ["word"],
        }
        structured = post(
            f"{base}/v1/chat/completions",
            {
                "model": "atomic-smoke",
                "messages": messages,
                "max_tokens": 32,
                "temperature": 0,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "answer", "schema": schema},
                },
            },
        )
        content = structured["choices"][0]["message"].get("content") or ""
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            fail(f"json_schema response is not JSON: {content[:500]!r}")
        if not isinstance(parsed, dict) or "word" not in parsed:
            fail(f"json_schema response does not follow the schema: {content[:500]!r}")
        print(f"structured OK: {content[:80]}")
    finally:
        server.terminate()
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server.kill()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument(
        "--models-dir", type=Path, help="Download the pinned model here and serve it"
    )
    parser.add_argument("--log", type=Path, default=Path("mlx-server-smoke.log"))
    a = parser.parse_args()

    binary = a.binary.resolve()
    if not binary.is_file():
        fail(f"no binary at {binary}")
    check_help(binary)
    check_selftest(binary)
    if a.models_dir:
        check_serving(binary, download(a.models_dir.expanduser()), a.log)
    print("smoke OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
