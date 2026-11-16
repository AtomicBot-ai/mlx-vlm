# Copyright © 2026 Neywa Labs Inc.

import ctypes
import errno
import hashlib
import io
import json
import os
import queue
import struct
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest

from mlx_vlm.server import checkpoint_store as store
from mlx_vlm.server import checkpoint_transfer as transfer
from mlx_vlm.server._checkpoint_worker import RDMAUnavailable, connection_call
from mlx_vlm.server.checkpoint_transfer import (
    Helper,
    Pair,
    TransferPool,
    TransferUnavailable,
    bootstrap,
)
from mlx_vlm.server.cluster_types import PrepareModel, RDMAPair
from mlx_vlm.server.distributed import ClusterConfig, ModelStager, Worker


@pytest.mark.parametrize("interactive", [False, True])
def test_terminal_progress_reuse_verification_and_cancellation(
    monkeypatch, interactive
):
    output = io.StringIO()
    output.isatty = lambda: interactive
    monkeypatch.setattr(sys, "stderr", output)
    with pytest.raises(InterruptedError), transfer.TransferProgress() as progress:
        progress.update(
            "peer",
            "transferring",
            dict(
                bytes_transferred=25,
                bytes_reused=25,
                total_bytes=100,
                transport="jaccl_rdma",
                throughput_bytes_per_second=10,
            ),
        )
        if interactive:
            assert progress.bars["peer"].n == 50
            assert "50%" in output.getvalue() and "ETA 00:05" in output.getvalue()
        progress.update(
            "peer",
            "verifying_worker",
            dict(
                bytes_transferred=75,
                bytes_reused=25,
                bytes_verified=60,
                total_bytes=100,
            ),
        )
        if interactive:
            assert progress.bars["peer"].n == 60
        raise InterruptedError("cancelled")
    if interactive:
        assert "peer: cancelled" in output.getvalue()
        assert "worker ready" not in output.getvalue()
        assert all(bar.disable for bar in progress.bars.values())
    else:
        assert output.getvalue() == ""


@pytest.fixture
def pool():
    value = TransferPool()
    yield value
    value.close()


@pytest.fixture
def snapshot(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text("{}")
    (source / "weights.safetensors").write_bytes(bytes(range(256)) * 35 + b"tail")
    names = ["config.json", "weights.safetensors"]
    root = tmp_path / "cache"
    final, manifest = store.prepare_snapshot(source, names, root, chunk=1024)
    return source, names, root, final, manifest


@pytest.fixture
def darwin_clone(monkeypatch):
    original = store._clone
    clone = Mock()

    def call(source, target):
        # Simulate Darwin only during clonefile. Copying and flushing must use
        # the host platform, which may not provide macOS's F_FULLFSYNC.
        with monkeypatch.context() as patch:
            patch.setattr(store, "sys", SimpleNamespace(platform="darwin"))
            patch.setattr(
                store.ctypes, "CDLL", lambda *a, **kw: SimpleNamespace(clonefile=clone)
            )
            return original(source, target)

    monkeypatch.setattr(store, "_clone", call)
    return clone


def test_manifest_reuse_does_not_read_weight_bytes(snapshot, monkeypatch):
    source, names, root, final, manifest = snapshot
    original = Path.open

    def guarded(path, *args, **kwargs):
        if path.suffix == ".safetensors":
            raise AssertionError("Warm preparation must not rescan weights")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)
    assert store.prepare_snapshot(source, names, root, chunk=1024) == (final, manifest)
    assert store.SnapshotStore(root).existing(manifest) == final
    assert not final.stat().st_mode & 0o222


def test_snapshot_does_not_modify_or_hardlink_source(snapshot):
    source, names, root, final, manifest = snapshot
    for name in names:
        assert (source / name).stat().st_ino != (final / name).stat().st_ino
        assert (source / name).stat().st_mode & 0o200
        assert not (final / name).stat().st_mode & 0o222
    (source / names[1]).write_bytes(b"updated")
    new_final, new_manifest = store.prepare_snapshot(source, names, root, chunk=1024)
    assert new_final != final and new_manifest != manifest
    assert (final / names[1]).read_bytes().endswith(b"tail")


def test_same_size_mutation_with_restored_mtime_cannot_reuse_receipt(snapshot):
    _, _, root, final, manifest = snapshot
    path = final / "weights.safetensors"
    before = path.stat()
    path.chmod(0o644)
    path.write_bytes(b"X" * before.st_size)
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    path.chmod(0o444)
    with pytest.raises(ValueError, match="checksum mismatch"):
        store.SnapshotStore(root).existing(manifest)


def test_resume_checks_partial_chunks_and_tail(snapshot, tmp_path):
    _, _, _, final, manifest = snapshot
    destination = store.SnapshotStore(tmp_path / "receiver")
    partial = destination.partial(manifest)
    data = (final / "weights.safetensors").read_bytes()
    (partial / "weights.safetensors").write_bytes(
        data[:1024] + b"X" * 1024 + data[2048:3072]
    )
    missing, reused = destination.missing(manifest)
    assert reused == 2048
    assert missing["weights.safetensors"] == [
        1,
        *range(3, len(manifest["files"]["weights.safetensors"]["chunks"])),
    ]
    assert missing["config.json"] == [0]


@pytest.mark.parametrize(
    "name", ["../escape", "/escape", "a/../escape", ".hidden", "a//b"]
)
def test_manifest_paths_cannot_escape(name, snapshot):
    manifest = json.loads(json.dumps(snapshot[-1]))
    manifest["files"][name] = manifest["files"].pop("config.json")
    with pytest.raises(ValueError):
        store.validate(manifest)


def test_symlinked_partial_is_not_followed(snapshot, tmp_path):
    manifest = snapshot[-1]
    destination = store.SnapshotStore(tmp_path / "receiver")
    partial = destination.partial(manifest)
    outside = tmp_path / "outside"
    outside.write_text("untouched")
    (partial / "config.json").symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        destination.missing(manifest)
    assert outside.read_text() == "untouched"


def test_readonly_rsync_partial_can_be_resumed(snapshot, tmp_path):
    manifest = snapshot[-1]
    destination = store.SnapshotStore(tmp_path / "receiver")
    partial = destination.partial(manifest)
    file = partial / "config.json"
    file.write_bytes((snapshot[-2] / "config.json").read_bytes())
    file.chmod(0o444)
    partial.chmod(0o555)
    destination.writable_partial(manifest)
    assert partial.stat().st_mode & 0o200 and file.stat().st_mode & 0o200
    missing, reused = destination.missing(manifest)
    assert not missing["config.json"] and reused == file.stat().st_size


@pytest.mark.parametrize(
    "name,error",
    [("config.json", "changed after"), ("unexpected.py", "Unexpected file")],
)
def test_publish_rejects_unverified_changes(snapshot, tmp_path, name, error):
    manifest = snapshot[-1]
    destination = store.SnapshotStore(tmp_path / "receiver")
    partial = destination.partial(manifest)
    signatures = {}
    for filename in manifest["files"]:
        (partial / filename).write_bytes((snapshot[-2] / filename).read_bytes())
        signatures[filename] = store.signature(partial / filename)
    (partial / name).write_text("unverified content")
    with pytest.raises(ValueError, match=error):
        destination.publish(manifest, signatures)
    assert not (destination.root / store.snapshot_id(manifest)).exists()


def test_snapshot_root_symlink_is_rejected(snapshot, tmp_path):
    manifest = snapshot[-1]
    destination = store.SnapshotStore(tmp_path / "receiver")
    (destination.root / store.snapshot_id(manifest)).symlink_to(snapshot[-2])
    with pytest.raises(ValueError, match="symlink"):
        destination.existing(manifest)


def test_cancelled_preparation_does_not_publish(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "weights.safetensors").write_bytes(b"weights")
    stop = threading.Event()
    stop.set()
    with pytest.raises(InterruptedError):
        store.prepare_snapshot(
            source, ["weights.safetensors"], tmp_path / "cache", stop=stop
        )
    assert not list((tmp_path / "cache").glob(".building-*"))


def test_helper_identity_mismatch_closes_owned_process():
    with pytest.raises(RuntimeError, match="identity"):
        Helper(
            [sys.executable, "-u", "-c", bootstrap()],
            timeout=5,
            physical_id="wrong-device",
        )


def test_explicit_rdma_never_silently_falls_back(snapshot, tmp_path, pool):
    _, _, root, final, manifest = snapshot
    worker = Worker("http://local-test", model_dir=str(tmp_path / "receiver"))
    config = ClusterConfig((worker,), root, tmp_path / "lock", python=sys.executable)
    with pytest.raises(RuntimeError, match="No direct RDMA"):
        pool.deliver(
            config, worker, final, manifest, "jaccl_rdma", lambda *a, **k: None
        )
    assert not pool.pairs
    assert not (Path(worker.model_dir) / store.snapshot_id(manifest)).exists()


@pytest.mark.parametrize("transport", ["auto", "ssh_rsync"])
def test_preparation_transports_share_verified_snapshots_and_pool(
    tmp_path, transport, pool
):
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text("{}")
    (source / "weights.safetensors").write_bytes(b"actual checkpoint")
    worker = Worker("http://local-test", model_dir=str(tmp_path / "worker"))
    config = ClusterConfig(
        (worker,), tmp_path / "cache", tmp_path / "lock", python=sys.executable
    )
    stager = ModelStager(config, transfers=pool)
    updates = []
    report = lambda phase, **data: updates.append((phase, data))
    first = stager.prepare(
        str(source), validate_sharding=False, transport=transport, progress=report
    )
    assert stager.worker_transfers[worker.url]["transport"] == "ssh_rsync"
    assert any(p == "transport_fallback" for p, _ in updates) is (transport == "auto")
    assert any(p == "verifying_worker" for p, _ in updates)
    path = Path(stager.worker_paths[(first, worker.url)])
    assert path.parent == Path(worker.model_dir) and path != Path(first)
    assert (path / "weights.safetensors").read_bytes() == (
        source / "weights.safetensors"
    ).read_bytes()
    assert not list(Path(worker.model_dir).glob(".partial-*"))
    pair = next(iter(pool.pairs.values()))
    pid = pair.remote.pid
    second = ModelStager(config, transfers=pool)
    assert (
        second.prepare(str(source), validate_sharding=False, transport="jaccl_rdma")
        == first
    )
    assert second.worker_transfers[worker.url]["transport"] == "verified_cache"
    # Switching transport, including ordinary inference staging, uses the
    # same snapshot without a connection check or checkpoint retransmission.
    assert (
        second.prepare(str(source), validate_sharding=False, progress=report) == first
    )
    assert pair.remote.pid == pid
    assert updates[-1][1]["bytes_transferred"] == 0
    assert updates[-1][1]["bytes_reused"] == sum(
        p.stat().st_size for p in source.iterdir()
    )
    pool.close()
    assert pair.remote.process.poll() is not None


def test_transport_contract_and_topology_validation():
    assert PrepareModel(model="org/model").transport == "ssh_rsync"
    with pytest.raises(ValueError):
        PrepareModel(model="org/model", transport="http")
    with pytest.raises(ValueError):
        RDMAPair(
            local_address="not-an-ip", local_device="rdma_en4", remote_device="rdma_en5"
        )


def test_chunk_hashes_and_whole_file_identity(snapshot):
    manifest = snapshot[-1]
    data = (snapshot[-2] / "weights.safetensors").read_bytes()
    record = manifest["files"]["weights.safetensors"]
    assert record["sha256"] == hashlib.sha256(data).hexdigest()
    assert record["chunks"] == [
        hashlib.sha256(data[i : i + 1024]).hexdigest()
        for i in range(0, len(data), 1024)
    ]


@pytest.mark.parametrize("code", [errno.EXDEV, errno.ENOTSUP, errno.ENOSYS])
def test_unsupported_clone_uses_verified_copy(tmp_path, darwin_clone, code):
    darwin_clone.side_effect = lambda *args: (ctypes.set_errno(code), -1)[1]
    source = tmp_path / "source"
    source.mkdir()
    data = b"checkpoint content"
    (source / "weights.safetensors").write_bytes(data)
    final, manifest = store.prepare_snapshot(
        source, ["weights.safetensors"], tmp_path / "cache"
    )
    assert (final / "weights.safetensors").read_bytes() == data
    assert (
        manifest["files"]["weights.safetensors"]["sha256"]
        == hashlib.sha256(data).hexdigest()
    )
    darwin_clone.assert_called_once()


@pytest.mark.parametrize(
    "code", [errno.EACCES, errno.EPERM, errno.ENOSPC, errno.EIO, errno.EINVAL]
)
def test_clone_errors_abort_instead_of_copying(tmp_path, darwin_clone, code):
    darwin_clone.side_effect = lambda *args: (ctypes.set_errno(code), -1)[1]
    source = tmp_path / "source"
    source.mkdir()
    (source / "weights.safetensors").write_bytes(b"checkpoint content")
    root = tmp_path / "cache"
    with pytest.raises(OSError) as caught:
        store.prepare_snapshot(source, ["weights.safetensors"], root)
    assert caught.value.errno == code
    assert not list(root.glob(".building-*"))
    assert not list(root.rglob("weights.safetensors"))


@pytest.mark.parametrize(
    "message",
    [
        "[jaccl] Recv failed with errno=2",
        "[jaccl] Send failed with errno=54",
        "[jaccl] Couldn't connect (error: 61)",
        "[jaccl] Accept failed (error: 60)",
        "[jaccl] Could not find device rdma_nonexistent (6 available)",
    ],
)
def test_native_connection_failures_are_classified(message):
    error = RuntimeError(message)
    with pytest.raises(RDMAUnavailable, match="jaccl") as caught:
        connection_call(Mock(side_effect=error))
    assert caught.value.__cause__ is error


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("[jaccl] Unsupported dtype for JACCL operation"),
        RuntimeError(
            "[jaccl] The configuration should define a valid mesh or a valid ring."
        ),
        RuntimeError("[jaccl] Register memory region failed"),
        RuntimeError("unrecognized future backend failure"),
        ValueError("RDMA connection challenge mismatch"),
        TypeError("unexpected backend keyword"),
        ImportError("missing dependency"),
        OSError(errno.ENOSPC, "disk full"),
    ],
)
def test_native_defects_are_not_classified_as_connection_failures(error):
    with pytest.raises(type(error)) as caught:
        connection_call(Mock(side_effect=error))
    assert caught.value is error


@pytest.mark.parametrize(
    "checking,code,expected",
    [
        (True, "rdma_unavailable", TransferUnavailable),
        (False, "rdma_unavailable", RuntimeError),
        (True, "helper_error", RuntimeError),
        (True, "unknown_error", RuntimeError),
    ],
)
def test_helper_fallback_requires_explicit_connection_error(checking, code, expected):
    helper = object.__new__(Helper)
    helper.timeout = 1
    helper.events = queue.Queue()
    helper.events.put(
        {"event": "error", "id": "test", "message": "failure", "code": code}
    )
    with pytest.raises(expected) as caught:
        helper.wait("test", connection_check=checking)
    assert type(caught.value) is expected


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("Incompatible checkpoint helper protocol"),
        TimeoutError("Helper bootstrap timed out"),
        TypeError("Unexpected implementation error"),
    ],
)
def test_local_helper_startup_errors_do_not_become_fallback(
    connecting_pair, error, monkeypatch
):
    pair, _ = connecting_pair
    monkeypatch.setattr(transfer, "Helper", Mock(side_effect=error))
    with pytest.raises(type(error)) as caught:
        pair.connect()
    assert caught.value is error


@pytest.fixture
def connecting_pair(monkeypatch):
    pair = object.__new__(Pair)
    pair.validated = False
    pair.config = SimpleNamespace(timeout=1)
    pair.worker = SimpleNamespace(
        ssh="peer",
        rdma={
            "local_address": "127.0.0.1",
            "local_device": "rdma_en4",
            "remote_device": "rdma_en5",
        },
    )
    pair.remote = Mock()
    pair.local = None
    local = Mock()
    monkeypatch.setattr(transfer, "Helper", Mock(return_value=local))
    return pair, local


@pytest.mark.parametrize(
    "error,expected",
    [
        (TimeoutError("Connection timed out"), TransferUnavailable),
        (TransferUnavailable("JACCL unavailable"), TransferUnavailable),
        (RuntimeError("Incompatible checkpoint helper protocol"), RuntimeError),
        (ValueError("RDMA connection challenge mismatch"), ValueError),
        (TypeError("unexpected backend keyword"), TypeError),
        (InterruptedError("Checkpoint preparation cancelled"), InterruptedError),
    ],
)
def test_connection_check_only_converts_expected_failures(
    connecting_pair, error, expected
):
    pair, local = connecting_pair

    def wait(*args, **kwargs):
        if kwargs.get("event") == "initializing":
            return {}
        raise error

    local.wait.side_effect = wait
    with pytest.raises(expected) as caught:
        pair.connect()
    assert type(caught.value) is expected
    assert not pair.validated
    local.close.assert_called_once()
    pair.remote.close.assert_called_once()


def test_initialization_ack_timeout_does_not_fallback(connecting_pair):
    pair, local = connecting_pair
    error = TimeoutError("Helper did not acknowledge initialization")
    local.wait.side_effect = error
    try:
        with pytest.raises(TimeoutError) as caught:
            pair.connect()
        assert caught.value is error
        assert not pair.validated
    finally:
        pair.close()


@pytest.mark.parametrize(
    "code,expected",
    [
        (errno.EADDRNOTAVAIL, TransferUnavailable),
        (errno.ENETDOWN, TransferUnavailable),
        (errno.EACCES, PermissionError),
        (errno.EMFILE, OSError),
    ],
)
def test_bind_only_falls_back_for_unavailable_network(
    connecting_pair, monkeypatch, code, expected
):
    pair, _ = connecting_pair
    socket = MagicMock()
    socket.__enter__.return_value.bind.side_effect = OSError(code, os.strerror(code))
    monkeypatch.setattr(transfer.socket, "socket", Mock(return_value=socket))
    try:
        with pytest.raises(expected) as caught:
            pair.connect()
        assert type(caught.value) is expected
        assert not pair.validated
    finally:
        pair.close()


@pytest.mark.parametrize(
    "error",
    [
        TransferUnavailable("Connection refused"),
        RuntimeError("Unexpected checkpoint helper command sequence"),
        ValueError("RDMA connection challenge mismatch"),
        TypeError("unexpected backend keyword"),
        TimeoutError("Helper bootstrap timed out"),
        InterruptedError("Checkpoint preparation cancelled"),
    ],
)
def test_auto_falls_back_only_on_classified_connection_errors(
    snapshot, tmp_path, monkeypatch, error, pool
):
    _, _, root, final, manifest = snapshot
    worker = Worker(
        "http://local-test",
        model_dir=str(tmp_path / "receiver"),
        rdma={
            "local_address": "127.0.0.1",
            "local_device": "rdma_en4",
            "remote_device": "rdma_en5",
        },
    )
    config = ClusterConfig((worker,), root, tmp_path / "lock", python=sys.executable)
    monkeypatch.setattr(Pair, "connect", Mock(side_effect=error))
    updates = []

    def deliver():
        return pool.deliver(
            config,
            worker,
            final,
            manifest,
            "auto",
            lambda phase, **data: updates.append((phase, data)),
        )

    if isinstance(error, TransferUnavailable):
        result = deliver()
        assert result["transport"] == "ssh_rsync" and result["verified"]
        assert result["fallback_reason"] == str(error)
        assert sum(phase == "transport_fallback" for phase, _ in updates) == 1
        assert (Path(result["path"]) / "weights.safetensors").read_bytes() == (
            final / "weights.safetensors"
        ).read_bytes()
    else:
        with pytest.raises(type(error)) as caught:
            deliver()
        assert caught.value is error
        assert not any(phase == "transport_fallback" for phase, _ in updates)
        assert not pool.pairs
        assert not (Path(worker.model_dir) / store.snapshot_id(manifest)).exists()


@pytest.mark.parametrize("corrupt", [False, True])
def test_hub_blob_checksum_is_verified_before_publication(tmp_path, corrupt):
    data = b"published Hugging Face weight" * 200
    digest = hashlib.sha256(data).hexdigest()
    repo = tmp_path / "models--org--model"
    blobs = repo / "blobs"
    blobs.mkdir(parents=True)
    blob = blobs / digest
    blob.write_bytes(b"X" * len(data) if corrupt else data)
    source = repo / "snapshots" / ("a" * 40)
    source.mkdir(parents=True)
    (source / "weights.safetensors").symlink_to(blob)
    root = tmp_path / "cache"
    if corrupt:
        with pytest.raises(ValueError, match="Hugging Face checksum mismatch"):
            store.prepare_snapshot(source, ["weights.safetensors"], root, chunk=1024)
        assert not list(root.glob("*/weights.safetensors"))
        assert not list(root.glob(".building-*"))
    else:
        _, manifest = store.prepare_snapshot(
            source, ["weights.safetensors"], root, chunk=1024
        )
        assert manifest["files"]["weights.safetensors"]["sha256"] == digest
    # A Git blob ID is not a SHA-256 checksum, nor is an arbitrary local filename.
    assert store._hub_sha256(blobs / ("a" * 40)) is None
    assert store._hub_sha256(tmp_path / digest) is None


def test_multiple_files_hash_concurrently_with_monotonic_progress(
    tmp_path, monkeypatch
):
    source = tmp_path / "source"
    source.mkdir()
    names = ["one.safetensors", "two.safetensors"]
    for n in names:
        (source / n).write_bytes(n.encode() * 300)
    monkeypatch.setattr(store.os, "cpu_count", lambda: 2)
    barrier = threading.Barrier(2)
    clone = store._clone

    def synchronized_clone(src, dst):
        barrier.wait(timeout=5)
        return clone(src, dst)

    monkeypatch.setattr(store, "_clone", synchronized_clone)
    updates = []
    _, manifest = store.prepare_snapshot(
        source,
        names,
        tmp_path / "cache",
        chunk=1024,
        progress=lambda **data: updates.append(data["bytes_verified"]),
    )
    assert list(manifest["files"]) == names
    assert updates == sorted(updates)
    assert updates[-1] == sum((source / n).stat().st_size for n in names)
    for n in names:
        assert (
            manifest["files"][n]["sha256"]
            == hashlib.sha256((source / n).read_bytes()).hexdigest()
        )


@pytest.mark.parametrize("corrupt", [False, True])
def test_receiver_pipeline_preserves_tail_and_resumed_chunks(
    snapshot, tmp_path, monkeypatch, corrupt
):
    import mlx.core as mx
    import numpy as np

    from mlx_vlm.server import _checkpoint_worker as worker

    _, _, _, final, manifest = snapshot
    destination = store.SnapshotStore(tmp_path / "received")
    partial = destination.partial(manifest)
    original = (final / "weights.safetensors").read_bytes()
    (partial / "weights.safetensors").write_bytes(original[:1024])
    missing, reused = destination.missing(manifest)
    assert reused == 1024
    payloads = []
    for name, record in manifest["files"].items():
        data = (final / name).read_bytes()
        for index in missing[name]:
            payloads.append(data[index * 1024 : (index + 1) * 1024])
    if corrupt:
        payloads[-1] = b"X" * len(payloads[-1])
    chunks = iter(payloads)
    monkeypatch.setattr(
        mx.distributed,
        "recv",
        lambda *a, **k: mx.array(np.frombuffer(next(chunks), dtype=np.uint8)),
    )
    command = {"manifest": manifest, "path": str(partial), "missing": missing}
    group = SimpleNamespace(rank=lambda: 1)
    if corrupt:
        with pytest.raises(ValueError, match="checksum mismatch"):
            worker.copy_chunks(group, command, lambda **data: None)
        assert not (destination.root / store.snapshot_id(manifest)).exists()
    else:
        signatures = worker.copy_chunks(group, command, lambda **data: None)
        published = destination.publish(manifest, signatures)
        assert (published / "weights.safetensors").read_bytes() == original


def test_parallel_preparation_cancellation_leaves_no_snapshot(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    names = ["one.safetensors", "two.safetensors"]
    for name in names:
        (source / name).write_bytes(b"w" * 8192)
    stop = threading.Event()
    root = tmp_path / "cache"
    with pytest.raises(InterruptedError):
        store.prepare_snapshot(
            source,
            names,
            root,
            chunk=1024,
            stop=stop,
            progress=lambda **data: stop.set(),
        )
    assert not list(root.glob(".building-*"))
    assert not list(root.glob("*/one.safetensors"))


@pytest.mark.parametrize("mutation", [None, "source", "snapshot", "manifest"])
def test_streamed_snapshot_checks_mutation_before_publication(
    snapshot, tmp_path, mutation
):
    source, names, _, _, manifest = snapshot
    root = tmp_path / "streamed"

    def streaming(path, layout):
        assert all((path / n).read_bytes() == (source / n).read_bytes() for n in names)
        assert not list(root.glob("*/.mlx-vlm-distributed.json"))
        if mutation in {"source", "snapshot"}:
            ((source if mutation == "source" else path) / names[0]).write_bytes(
                b"mutation"
            )
        if mutation == "manifest":
            return {**manifest, "chunk_bytes": 2048}
        return manifest

    if mutation:
        with pytest.raises(ValueError):
            store.prepare_snapshot(source, names, root, chunk=1024, stream=streaming)
        assert not list(root.glob("*/.mlx-vlm-distributed.json"))
    else:
        final, actual = store.prepare_snapshot(
            source, names, root, chunk=1024, stream=streaming
        )
        assert actual == manifest
        assert (final / names[1]).read_bytes() == (source / names[1]).read_bytes()
        store.prepare_snapshot(
            source,
            names,
            root,
            chunk=1024,
            stream=lambda *a: pytest.fail("Warm source must not stream"),
        )
    assert not list(root.glob(".building-*"))


@pytest.mark.parametrize(
    "transport,connection_error,fallback",
    [
        ("auto", TransferUnavailable("link unavailable"), True),
        ("jaccl_rdma", TransferUnavailable("link unavailable"), False),
        ("auto", ValueError("backend defect"), False),
    ],
)
def test_pipeline_fallback_only_for_auto_connection_failure(
    snapshot, tmp_path, pool, monkeypatch, transport, connection_error, fallback
):
    source, names, _, _, _ = snapshot
    config = ClusterConfig((), tmp_path / "fresh", tmp_path / "lock")
    worker = Worker("http://peer")
    pair = Mock()
    pair.connect.side_effect = connection_error
    monkeypatch.setattr(transfer, "Pair", lambda *args: pair)
    result = {"transport": "ssh_rsync", "path": "prepared"}
    pool.deliver = Mock(return_value=result)
    call = lambda: pool.prepare_and_deliver(
        config,
        worker,
        source,
        names,
        transport,
        lambda *a, **kw: None,
        lambda **kw: None,
    )
    if fallback:
        final, manifest, actual = call()
        assert store.SnapshotStore(config.model_dir).existing(manifest) == final
        assert actual["fallback_reason"] == "link unavailable"
        assert pool.deliver.call_args.args[4] == "ssh_rsync"
    else:
        with pytest.raises(type(connection_error), match=str(connection_error)):
            call()
        pool.deliver.assert_not_called()
        assert not list(config.model_dir.glob("*/.mlx-vlm-distributed.json"))
    pair.close.assert_called()
    assert not pool.pairs
    assert not list(config.model_dir.glob(".building-*"))


def test_pipeline_helper_rejects_commit_until_every_file_is_received(
    snapshot, tmp_path
):
    helper = Helper([sys.executable, "-u", "-c", bootstrap()], timeout=5)
    try:
        helper.call(
            "begin_stream",
            root=str(tmp_path / "receiver"),
            partial_id="a" * 64,
            layout={
                "chunk_bytes": 1024,
                "files": {
                    n: {"size": r["size"]} for n, r in snapshot[-1]["files"].items()
                },
            },
        )
        with pytest.raises(RuntimeError, match="Incomplete or changed"):
            helper.call("commit", manifest=snapshot[-1])
        assert not list((tmp_path / "receiver").glob("*/.mlx-vlm-distributed.json"))
    finally:
        helper.close()


def test_pipeline_warm_source_resumes_legacy_partial(snapshot, tmp_path):
    _, _, _, final, manifest = snapshot
    destination = store.SnapshotStore(tmp_path / "receiver")
    partial = destination.partial(manifest)
    (partial / "config.json").write_bytes((final / "config.json").read_bytes())
    helper = Helper([sys.executable, "-u", "-c", bootstrap()], timeout=5)
    try:
        state = helper.call(
            "begin", root=str(destination.root), manifest=manifest, partial_id="b" * 64
        )
        assert state["path"] == str(partial)
        assert state["bytes_reused"] == manifest["files"]["config.json"]["size"]
    finally:
        helper.close()


def test_stager_pipelines_remote_peer_when_local_rank_is_first(
    snapshot, tmp_path, pool
):
    source, _, root, final, manifest = snapshot
    local = Worker("http://local", node_id="local")
    remote = Worker(
        "http://remote",
        node_id="remote",
        rdma={
            "local_address": "192.0.2.1",
            "local_device": "rdma_en1",
            "remote_device": "rdma_en2",
        },
    )
    config = ClusterConfig((local, remote), root, tmp_path / "lock")
    remote_result = {"transport": "jaccl_rdma", "path": "/remote/snapshot"}
    pool.prepare_and_deliver = Mock(return_value=(final, manifest, remote_result))
    pool.deliver = Mock(
        return_value={"transport": "verified_cache", "path": str(final)}
    )
    stager = ModelStager(config, transfers=pool)
    assert stager.prepare(
        str(source), validate_sharding=False, transport="auto"
    ) == str(final)
    assert pool.prepare_and_deliver.call_args.args[1] is remote
    assert pool.deliver.call_args.args[1] is local
    assert stager.worker_transfers[remote.url] == remote_result


@pytest.mark.parametrize("corrupt", [None, "source", "payload"])
def test_stream_hashes_each_source_byte_once_and_verifies_receiver(
    tmp_path, monkeypatch, corrupt
):
    import mlx.core as mx
    import numpy as np

    from mlx_vlm.server import _checkpoint_worker as worker

    source, partial = tmp_path / "source", tmp_path / "partial"
    source.mkdir()
    partial.mkdir()
    original = {f"part{i}": bytes([i]) * (2048 + i * 53) for i in range(9)}
    original["empty"] = b""
    for name, data in original.items():
        (source / name).write_bytes(data)
    (partial / "part0").write_bytes(original["part0"])  # Reuse actual verified chunks.
    (partial / "part1").write_bytes(b"wrong" * 400)  # Repair a corrupt partial.
    layout = {
        "chunk_bytes": 1024,
        "files": {
            n: {"size": len(v), "expected_sha256": hashlib.sha256(v).hexdigest()}
            for n, v in original.items()
        },
    }
    if corrupt == "source":
        (source / "part0").write_bytes(b"X" * len(original["part0"]))
    partial_hashes = worker.partial_hashes(partial, layout, lambda **kw: None)
    packets, reads = [], {}
    open_file = Path.open

    class CountReads:
        def __init__(self, file, name):
            self.file, self.name = file, name

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.file.close()

        def fileno(self):
            return self.file.fileno()

        def flush(self):
            return self.file.flush()

        def readinto(self, view):
            n = self.file.readinto(view)
            reads[self.name] = reads.get(self.name, 0) + n
            return n

    def opening(path, mode="r", *args, **kwargs):
        file = open_file(path, mode, *args, **kwargs)
        return (
            CountReads(file, path.name)
            if path.parent == source and mode == "rb"
            else file
        )

    def sending(value, *args, **kwargs):
        mx.eval(value)
        packets.append(np.asarray(value).tobytes())
        return value

    monkeypatch.setattr(Path, "open", opening)
    monkeypatch.setattr(mx.distributed, "send", sending)
    command = {"layout": layout, "path": str(source), "partial_hashes": partial_hashes}
    if corrupt == "source":
        with pytest.raises(ValueError, match="Hugging Face checksum mismatch"):
            worker.stream_chunks(
                SimpleNamespace(rank=lambda: 0), command, lambda **kw: None
            )
        return
    sent = worker.stream_chunks(
        SimpleNamespace(rank=lambda: 0), command, lambda **kw: None
    )
    assert sum(reads.values()) == sum(map(len, original.values()))
    assert (
        sent["manifest"]["files"]["part0"]["sha256"]
        == layout["files"]["part0"]["expected_sha256"]
    )
    if corrupt == "payload":
        index = next(i for i, data in enumerate(packets) if len(data) >= 1024)
        packets[index] = b"X" * len(packets[index])
    packets = iter(packets)

    def receiving(shape, *args, **kwargs):
        data = next(packets)
        assert len(data) == shape[0]
        return mx.array(np.frombuffer(data, dtype=np.uint8))

    monkeypatch.setattr(mx.distributed, "recv", receiving)
    events = []
    receive = lambda: worker.stream_chunks(
        SimpleNamespace(rank=lambda: 1),
        {"layout": layout, "path": str(partial)},
        lambda **kw: events.append(kw),
    )
    if corrupt == "payload":
        with pytest.raises(ValueError, match="checksum mismatch"):
            receive()
    else:
        got = receive()
        assert {n: (partial / n).read_bytes() for n in original} == original
        assert got["manifest"]["files"] == {
            n: {"size": r["size"], "chunks": r["chunks"]}
            for n, r in sent["manifest"]["files"].items()
        }
        assert events[-1]["bytes_reused"] == len(original["part0"])
        assert events[-1]["bytes_verified"] == sum(map(len, original.values()))
        assert (
            events[-1]["bytes_received"] + events[-1]["bytes_reused"]
            == events[-1]["bytes_verified"]
        )


@pytest.mark.parametrize(
    "platform,result,length,volume",
    [
        ("darwin", 0, 20, bytes(range(16))),
        ("darwin", -1, 20, bytes(range(16))),
        ("darwin", 0, 4, bytes(range(16))),
        ("darwin", 0, 20, bytes(16)),
        ("linux", 0, 20, bytes(range(16))),
    ],
)
def test_volume_identity_falls_back_when_uuid_is_unavailable(
    monkeypatch, platform, result, length, volume
):
    def getattrlist(path, attrs, output, size, options):
        payload = struct.pack("=I", length) + volume
        ctypes.memmove(output, payload, len(payload))
        return result

    call = Mock(side_effect=getattrlist)
    monkeypatch.setattr(store, "sys", SimpleNamespace(platform=platform))
    monkeypatch.setattr(
        store.ctypes, "CDLL", lambda *args: SimpleNamespace(getattrlist=call)
    )
    expected = (
        "volume-uuid:" + volume.hex()
        if platform == "darwin" and result == 0 and length == 20 and any(volume)
        else 1234
    )
    assert store._volume_identity("/checkpoint", 1234) == expected
    assert call.call_count == (1 if platform == "darwin" else 0)


def test_volume_uuid_survives_mount_number_change(snapshot, monkeypatch):
    source, names, root, final, manifest = snapshot
    original = store._volume_identity
    if not isinstance(original(source, source.stat().st_dev), str):
        pytest.skip("Persistent volume identity is Darwin-specific")
    real_stat = Path.lstat

    def remounted(path):
        s = real_stat(path)
        return SimpleNamespace(
            **{
                k: getattr(s, k)
                for k in [
                    "st_mode",
                    "st_ino",
                    "st_size",
                    "st_mtime_ns",
                    "st_ctime_ns",
                    "st_uid",
                ]
            },
            st_dev=s.st_dev + 100,
        )

    monkeypatch.setattr(Path, "lstat", remounted)
    monkeypatch.setattr(
        store,
        "verify_chunks",
        Mock(side_effect=AssertionError("Must reuse the persistent receipt")),
    )
    stream = Mock(side_effect=AssertionError("Must not transfer unchanged weights"))
    assert store.prepare_snapshot(source, names, root, chunk=1024, stream=stream) == (
        final,
        manifest,
    )
    stream.assert_not_called()
