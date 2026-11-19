# Copyright © 2026 Neywa Labs Inc.
"""Stable installation identity and measured, explicitly optional telemetry."""

import fcntl
import hashlib
import os
import platform
import shutil
import subprocess
import time
import uuid
from pathlib import Path

from .cluster_types import Memory, NodeIdentity


def state_directory():
    return Path(
        os.environ.get("MLX_VLM_CLUSTER_STATE_DIR", "~/.cache/mlx-vlm/cluster")
    ).expanduser()


def node_id():
    explicit = os.environ.get("MLX_VLM_NODE_ID")
    if explicit:
        return explicit
    root = state_directory()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = root / "node-id"
    with (root / "node-id.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            value = path.read_text().strip()
            if not value:
                raise ValueError(
                    "Empty persistent node identity; restore node-id before starting"
                )
            return value
        temporary = root / (".node-id-" + uuid.uuid4().hex)
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        value = uuid.uuid4().hex
        with os.fdopen(fd, "w") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        return value


def physical_id():
    """Hash a host identifier, independent of endpoint/rank/installation path."""
    if platform.system() == "Darwin":
        output = subprocess.check_output(
            ["ioreg", "-rd1", "-c", "IOPlatformExpertDevice"], text=True, timeout=3
        )
        raw = next(
            line.split("=")[1].strip()
            for line in output.splitlines()
            if '"IOPlatformUUID"' in line
        )
    else:
        raw = Path("/etc/machine-id").read_text().strip()
    return hashlib.sha256(raw.encode()).hexdigest()


def memory_info():
    import mlx.core as mx

    memory = Memory()
    # psutil is optional: absent measurements are never reported as zero usage.
    try:
        import psutil

        measured = psutil.virtual_memory()
        memory.total_bytes = int(measured.total)
        memory.used_bytes = int(measured.total - measured.available)
    except (ImportError, OSError):
        if platform.system() == "Darwin":
            try:
                memory.total_bytes = int(
                    subprocess.check_output(["sysctl", "-n", "hw.memsize"], timeout=3)
                )
                output = subprocess.check_output(["vm_stat"], text=True, timeout=3)
                page_size = int(output.split("page size of ")[1].split(" bytes")[0])
                pages = {
                    line.split(":")[0]: int(line.split(":")[1].strip().rstrip("."))
                    for line in output.splitlines()[1:]
                    if ":" in line
                }
                available = (
                    sum(
                        pages.get(key, 0)
                        for key in ("Pages free", "Pages inactive", "Pages speculative")
                    )
                    * page_size
                )
                memory.used_bytes = max(0, memory.total_bytes - available)
            except (OSError, ValueError, IndexError, subprocess.SubprocessError):
                pass
    for name, getter in (
        ("mlx_active_bytes", "get_active_memory"),
        ("mlx_cache_bytes", "get_cache_memory"),
        ("mlx_peak_bytes", "get_peak_memory"),
    ):
        try:
            setattr(memory, name, int(getattr(mx, getter)()))
        except (AttributeError, RuntimeError):
            pass
    return memory


def link_info():
    executable = shutil.which("ibv_devinfo")
    if not executable:
        return []
    try:
        output = subprocess.check_output(
            [executable], text=True, timeout=3, stderr=subprocess.DEVNULL
        )
    except (OSError, subprocess.SubprocessError):
        return []
    links, current = [], None
    for line in output.splitlines():
        key, separator, value = line.strip().partition(":")
        if not separator:
            continue
        if key == "hca_id":
            current = {
                "interface": value.strip(),
                "source": "ibv_devinfo",
                "measured_at": time.time(),
                "negotiated_rate": None,
                "rdma_validated": False,
            }
            links.append(current)
        elif current is not None and key in {
            "state",
            "active_mtu",
            "link_layer",
            "active_speed",
            "active_width",
        }:
            current[key] = value.strip()
            if key == "active_speed":
                current["negotiated_rate"] = value.strip()
    return links


def identity(worker=None):
    import mlx.core as mx

    from ..version import __version__

    try:
        physical = physical_id()
    except (OSError, StopIteration, subprocess.SubprocessError):
        physical = "installation:" + node_id()
    backends = []
    for backend in ("jaccl", "ring", "mpi", "nccl"):
        try:
            if mx.distributed.is_available(backend):
                backends.append(backend)
        except (TypeError, ValueError, RuntimeError):
            pass
    info = worker.info() if worker else None
    if info and info["backend"] not in backends:
        backends.append(info["backend"])
    return NodeIdentity(
        id=node_id(),
        name=platform.node(),
        physical_id=physical,
        os=platform.platform(),
        mlx_version=mx.__version__,
        mlx_vlm_version=__version__,
        memory=memory_info(),
        measured_at=time.time(),
        backends=backends,
        checkpoint_transfer=["ssh_rsync"]
        + (["jaccl_rdma"] if "jaccl" in backends else []),
        worker=info,
        model_dir=(
            str(worker.model_dir)
            if worker
            else os.environ.get("MLX_VLM_DISTRIBUTED_MODEL_DIR")
        ),
        links=link_info(),
        rdma_validated=bool(
            worker
            and worker.backend == "jaccl"
            and getattr(worker, "inference_validated_at", None)
        ),
    )
