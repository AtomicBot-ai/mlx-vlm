# Copyright © 2026 Neywa Labs Inc.
"""Private, sealed checkpoint snapshots and bounded chunk verification.

Receipts are written only after verification. Reuse requires the same inode,
ctime, size, mode and mtime for every read-only file and directory. This protects
against ordinary cache mutation, not an attacker controlling the server's uid.
No source file is chmod'ed or hard-linked into the cache.
"""

import concurrent.futures
import ctypes
import errno
import hashlib
import json
import os
import re
import stat
import struct
import sys
import threading
import uuid
from collections import deque
from pathlib import Path

CHUNK_BYTES = 16 * 1024 * 1024
MANIFEST = ".mlx-vlm-distributed.json"
# Darwin getattrlist constants from <sys/attr.h>.
ATTR_BIT_MAP_COUNT = 5
ATTR_VOL_UUID = 0x00040000
ATTR_VOL_INFO = 0x80000000


def cancelled(stop):
    if stop is not None and stop.is_set():
        raise InterruptedError("Checkpoint preparation cancelled")


def _volume_identity(path, device):
    # Darwin assigns st_dev at mount time; it is not a persistent volume ID.
    # getattrlist's ATTR_VOL_UUID is stable across reboots. Unsupported filesystems
    # retain the conservative device-number check (and reverify on a cache miss).
    if sys.platform == "darwin":
        attrs = ctypes.create_string_buffer(
            struct.pack(
                "=HHIIIII",
                ATTR_BIT_MAP_COUNT,
                0,
                0,
                ATTR_VOL_INFO | ATTR_VOL_UUID,
                0,
                0,
                0,
            )
        )
        result = ctypes.create_string_buffer(20)
        if ctypes.CDLL(None).getattrlist(os.fsencode(path), attrs, result, 20, 0) == 0:
            length = struct.unpack_from("=I", result.raw)[0]
            volume = result.raw[4:20]
            if length == 20 and any(volume):
                return "volume-uuid:" + volume.hex()
    return device


def signature(path):
    s = Path(path).lstat()
    if stat.S_ISLNK(s.st_mode):
        raise ValueError("Snapshot symlinks are not permitted")
    return [
        _volume_identity(path, s.st_dev),
        s.st_ino,
        s.st_size,
        s.st_mtime_ns,
        s.st_ctime_ns,
        s.st_mode,
    ]


def safe_path(root, name):
    if root.is_symlink():
        raise ValueError("Snapshot symlinks are not permitted")
    rel = Path(name)
    if (
        not name
        or rel.is_absolute()
        or any(p in {"", "..", "."} or p.startswith(".") for p in name.split("/"))
    ):
        raise ValueError("Invalid checkpoint path")
    path = root
    for part in rel.parts:
        path = path / part
        if path.is_symlink():
            raise ValueError("Snapshot symlinks are not permitted")
    return path


def validate(manifest):
    if manifest.get("version") != 1:
        raise ValueError("Unsupported checkpoint manifest version")
    chunk = manifest.get("chunk_bytes")
    if type(chunk) is not int or not 0 < chunk <= 64 * 1024 * 1024:
        raise ValueError("Invalid checkpoint chunk size")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Empty checkpoint manifest")
    for name, record in files.items():
        safe_path(Path("/nonexistent-mlx-manifest-validation"), name)
        size = record.get("size")
        hashes = record.get("chunks")
        if type(size) is not int or size < 0 or not isinstance(hashes, list):
            raise ValueError("Invalid checkpoint file size or chunks")
        if len(hashes) != (size + chunk - 1) // chunk:
            raise ValueError("Invalid checkpoint chunk count")
        if any(
            not isinstance(h, str) or not re.fullmatch(r"[a-f0-9]{64}", h)
            for h in [record.get("sha256"), *hashes]
        ):
            raise ValueError("Invalid checkpoint checksum")
    return manifest


def validate_layout(layout):
    """Validate sizes before the sender has calculated the final manifest."""
    chunk = layout.get("chunk_bytes")
    files = layout.get("files")
    if type(chunk) is not int or not 0 < chunk <= 64 * 1024 * 1024:
        raise ValueError("Invalid checkpoint chunk size")
    if not isinstance(files, dict) or not files:
        raise ValueError("Empty checkpoint layout")
    for name, record in files.items():
        safe_path(Path("/nonexistent-mlx-manifest-validation"), name)
        if type(record.get("size")) is not int or record["size"] < 0:
            raise ValueError("Invalid checkpoint file size")
        expected = record.get("expected_sha256")
        if expected is not None and (
            not isinstance(expected, str) or not re.fullmatch(r"[a-f0-9]{64}", expected)
        ):
            raise ValueError("Invalid expected checkpoint checksum")
    return layout


def snapshot_id(manifest):
    validate(manifest)
    return hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    tmp = path.with_name(".write-" + uuid.uuid4().hex)
    fd = os.open(tmp, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
        sync_directory(path.parent)
    finally:
        tmp.unlink(missing_ok=True)


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def flush_file(stream):
    import fcntl

    stream.flush()
    os.fsync(stream.fileno())
    if sys.platform == "darwin":
        fcntl.fcntl(stream.fileno(), fcntl.F_FULLFSYNC)


def private_directory(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    s = path.lstat()
    if not stat.S_ISDIR(s.st_mode) or s.st_uid != os.getuid() or s.st_mode & 0o022:
        raise ValueError(
            "Checkpoint cache must be owned by this user and not writable by others"
        )
    return path


def tree_signature(path):
    result = {".": signature(path)}
    for child in sorted(path.rglob("*")):
        value = signature(child)
        if not (stat.S_ISREG(value[-1]) or stat.S_ISDIR(value[-1])):
            raise ValueError("Snapshot contains a non-regular entry")
        if value[-1] & 0o222:
            raise ValueError("Snapshot is not sealed")
        result[child.relative_to(path).as_posix()] = value
    if result["."][-1] & 0o222:
        raise ValueError("Snapshot is not sealed")
    return result


def verify_chunks(path, record, chunk, *, stop=None, progress=None):
    before = signature(path)
    if not stat.S_ISREG(before[-1]) or before[2] != record["size"]:
        raise ValueError("Checkpoint size mismatch")
    with path.open("rb", buffering=0) as stream:
        for expected in record["chunks"]:
            cancelled(stop)
            data = stream.read(chunk)
            if hashlib.sha256(data).hexdigest() != expected:
                raise ValueError("Checkpoint chunk checksum mismatch")
            if progress:
                progress(len(data))
    if signature(path) != before:
        raise ValueError("Checkpoint changed during verification")


class SnapshotStore:
    def __init__(self, root, *, partial_id=None):
        self.root = private_directory(root)
        self.receipts = private_directory(self.root / ".verified")
        if partial_id is not None and not re.fullmatch(r"[a-f0-9]{64}", partial_id):
            raise ValueError("Invalid checkpoint transaction ID")
        self.partial_id = partial_id

    def existing(self, manifest, *, stop=None, progress=None):
        digest = snapshot_id(manifest)
        final = self.root / digest
        if final.is_symlink():
            raise ValueError("Snapshot symlinks are not permitted")
        if not final.exists():
            return None
        receipt = self.receipts / (digest + ".json")
        try:
            saved = json.loads(receipt.read_text())
            if saved["manifest"] == manifest and saved["signature"] == tree_signature(
                final
            ):
                return final
        except (OSError, ValueError, KeyError):
            pass
        # Changed, unsealed or legacy cache: never trust its old receipt.
        for name, record in manifest["files"].items():
            cancelled(stop)
            verify_chunks(
                safe_path(final, name),
                record,
                manifest["chunk_bytes"],
                stop=stop,
                progress=progress,
            )
        allowed = set(manifest["files"]) | {MANIFEST}
        if any(
            p.is_file() and p.relative_to(final).as_posix() not in allowed
            for p in final.rglob("*")
        ):
            raise ValueError("Unexpected file in checkpoint snapshot")
        self._seal(final, manifest)
        return final

    def _seal(self, path, manifest):
        for child in sorted(path.rglob("*"), reverse=True):
            signature(child)  # Reject symlinks before chmod.
            child.chmod(0o555 if child.is_dir() else 0o444)
        path.chmod(0o555)
        atomic_json(
            self.receipts / (snapshot_id(manifest) + ".json"),
            {"manifest": manifest, "signature": tree_signature(path)},
        )

    def partial(self, manifest):
        digest = (
            "pipeline-" + self.partial_id
            if self.partial_id is not None
            else snapshot_id(manifest)
        )
        path = self.root / (".partial-" + digest)
        private_directory(path)
        return path

    def writable_partial(self, manifest):
        """rsync can leave source read-only modes on an interrupted partial copy."""
        path = self.partial(manifest)
        path.chmod(0o700)
        for child in path.rglob("*"):
            value = signature(child)
            if not (stat.S_ISDIR(value[-1]) or stat.S_ISREG(value[-1])):
                raise ValueError("Non-regular checkpoint entry")
            child.chmod(0o700 if child.is_dir() else 0o600)
        return path

    def missing(self, manifest, *, stop=None, progress=None):
        """Resume checks actual partial bytes; a persisted bitmap alone is unsafe."""
        partial = self.partial(manifest)
        chunk = manifest["chunk_bytes"]
        missing, reused = {}, 0
        for name, record in manifest["files"].items():
            path = safe_path(partial, name)
            indexes = []
            if path.exists() and not path.is_file():
                raise ValueError("Non-file checkpoint target")
            stream = path.open("rb", buffering=0) if path.exists() else None
            try:
                for index, expected in enumerate(record["chunks"]):
                    cancelled(stop)
                    size = min(chunk, record["size"] - index * chunk)
                    data = stream.read(size) if stream else b""
                    if (
                        len(data) == size
                        and hashlib.sha256(data).hexdigest() == expected
                    ):
                        reused += size
                    else:
                        indexes.append(index)
                    if progress:
                        progress(len(data))
            finally:
                if stream:
                    stream.close()
            missing[name] = indexes
        return missing, reused

    def publish(self, manifest, signatures):
        partial = self.partial(manifest)
        for path in partial.rglob("*"):
            value = signature(path)
            if not (stat.S_ISDIR(value[-1]) or stat.S_ISREG(value[-1])):
                raise ValueError("Non-regular checkpoint entry")
            if path.is_file() and path.relative_to(partial).as_posix() not in {
                *manifest["files"],
                MANIFEST,
            }:
                raise ValueError("Unexpected file in checkpoint snapshot")
        for name in manifest["files"]:
            if signature(safe_path(partial, name)) != signatures.get(name):
                raise ValueError("Checkpoint changed after chunk verification")
        # Old worker loaders require this conventional manifest marker.
        atomic_json(
            partial / MANIFEST,
            {
                n: {k: r[k] for k in ("size", "sha256")}
                for n, r in manifest["files"].items()
            },
        )
        final = self.root / snapshot_id(manifest)
        if final.exists():
            raise ValueError("Preserving existing snapshot; refusing replacement")
        os.rename(partial, final)
        sync_directory(self.root)
        self._seal(final, manifest)
        return final


def _clone(source, target):
    # APFS clonefile is copy-on-write, unlike a hard link. Fall back on other FSes.
    if sys.platform == "darwin":
        libc = ctypes.CDLL(None, use_errno=True)
        clone = libc.clonefile
        clone.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
        clone.restype = ctypes.c_int
        if clone(os.fsencode(source), os.fsencode(target), 0) == 0:
            return True
        error = ctypes.get_errno()
        if error not in {errno.EXDEV, errno.ENOTSUP, errno.EOPNOTSUPP, errno.ENOSYS}:
            raise OSError(error, os.strerror(error), os.fspath(target))
    return False


def _hub_sha256(path):
    # Hub cache blob names are their ETags: SHA-256 for LFS, Git SHA-1 otherwise.
    # These are expected hashes, never proof that the cached bytes are intact.
    if (
        path.parent.name == "blobs"
        and path.parent.parent.name.startswith("models--")
        and re.fullmatch(r"[a-f0-9]{64}", path.name)
    ):
        return path.name
    return None


def prepare_snapshot(
    source,
    names,
    root,
    *,
    stop=None,
    progress=None,
    chunk=CHUNK_BYTES,
    stream=None,
):
    """Hash once, create an independent sealed snapshot, persist its chunk manifest."""
    store = SnapshotStore(root)
    source = Path(source).resolve()
    # Follow Hub blob links for the input only; receipts always describe private copies.
    inputs = {n: signature((source / n).resolve()) for n in names}
    expected = {n: _hub_sha256((source / n).resolve()) for n in names}
    key = hashlib.sha256(str(source).encode()).hexdigest()
    index = store.receipts / ("source-" + key + ".json")
    try:
        cached = json.loads(index.read_text())
        if (
            cached["inputs"] == inputs
            and cached["manifest"]["chunk_bytes"] == chunk
            and all(
                not h or cached["manifest"]["files"][n]["sha256"] == h
                for n, h in expected.items()
            )
        ):
            final = store.existing(cached["manifest"], stop=stop)
            if final:
                return final, cached["manifest"]
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        pass
    temporary = private_directory(store.root / (".building-" + uuid.uuid4().hex))
    read = 0
    verified_signatures = {}
    progress_lock, failed = threading.Lock(), threading.Event()

    def prepare_file(name):
        nonlocal read
        cancelled(stop)
        cancelled(failed)
        src = (source / name).resolve()
        target = safe_path(temporary, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        cloned = _clone(src, target)
        digest, chunks, pending = hashlib.sha256(), [], deque()
        output = None if cloned else target.open("xb", buffering=0)
        try:
            with (target if cloned else src).open("rb", buffering=0) as stream:
                while data := stream.read(chunk):
                    cancelled(stop)
                    cancelled(failed)
                    if output:
                        view = memoryview(data)
                        while view:
                            count = output.write(view)
                            if not count:
                                raise OSError("Short checkpoint write")
                            view = view[count:]
                    pending.append(hash_pool.submit(hashlib.sha256, data))
                    digest.update(data)
                    if len(pending) >= 2:
                        chunks.append(pending.popleft().result().hexdigest())
                    # Serialize callbacks as well as counters; consumers see monotonic
                    # totals even when several files complete chunks simultaneously.
                    with progress_lock:
                        read += len(data)
                        if progress:
                            progress(bytes_read=read, bytes_verified=read)
            chunks.extend(f.result().hexdigest() for f in pending)
            if expected[name] and digest.hexdigest() != expected[name]:
                raise ValueError(f"Hugging Face checksum mismatch: {name}")
            if output:
                flush_file(output)
            else:
                with target.open("rb") as cloned_file:
                    flush_file(cloned_file)
        finally:
            if output:
                output.close()
        if signature(src) != inputs[name]:
            raise ValueError("Source checkpoint changed during preparation")
        verified_signatures[name] = signature(target)
        return {
            "size": target.stat().st_size,
            "sha256": expected[name] or digest.hexdigest(),
            "chunks": chunks,
        }

    try:
        if stream:
            # Clone privately before sending. On APFS this does not read the bytes;
            # other filesystems retain a bounded copy to isolate mutable sources.
            for name in names:
                cancelled(stop)
                target = safe_path(temporary, name)
                target.parent.mkdir(parents=True, exist_ok=True)
                if not _clone((source / name).resolve(), target):
                    with (source / name).open("rb") as src, target.open("xb") as dst:
                        while data := src.read(chunk):
                            cancelled(stop)
                            dst.write(data)
                        flush_file(dst)
                verified_signatures[name] = signature(target)
            layout = {
                "chunk_bytes": chunk,
                "files": {
                    n: {"size": inputs[n][2], "expected_sha256": expected[n]}
                    for n in names
                },
            }
            manifest = validate(stream(temporary, layout))
            if manifest["chunk_bytes"] != chunk or set(manifest["files"]) != set(names):
                raise ValueError("Unexpected streamed checkpoint manifest")
            for name in names:
                record = manifest["files"][name]
                if record["size"] != inputs[name][2] or (
                    expected[name] and record["sha256"] != expected[name]
                ):
                    raise ValueError("Streamed checkpoint does not match source")
        else:
            # Whole-file SHA-256 is serial within a file; use bounded file parallelism.
            with (
                concurrent.futures.ThreadPoolExecutor(max_workers=4) as hash_pool,
                concurrent.futures.ThreadPoolExecutor(
                    max_workers=min(8, os.cpu_count() or 1)
                ) as pool,
            ):
                futures = {pool.submit(prepare_file, n): n for n in names}
                files = {}
                try:
                    for future in concurrent.futures.as_completed(futures):
                        files[futures[future]] = future.result()
                except BaseException:
                    failed.set()
                    for future in futures:
                        future.cancel()
                    raise
            manifest = {
                "version": 1,
                "chunk_bytes": chunk,
                "files": {n: files[n] for n in names},
            }
        cancelled(stop)
        if any(signature((source / n).resolve()) != inputs[n] for n in names):
            raise ValueError("Source checkpoint changed during preparation")
        if any(signature(temporary / n) != verified_signatures[n] for n in names):
            raise ValueError("Prepared checkpoint changed before publication")
        final = store.existing(manifest, stop=stop)
        if final is None:
            partial = store.root / (".partial-" + snapshot_id(manifest))
            if partial.exists():
                raise ValueError(
                    "Local partial checkpoint exists; inspect it before rebuilding"
                )
            os.rename(temporary, partial)
            final = store.publish(manifest, verified_signatures)
        atomic_json(index, {"inputs": inputs, "manifest": manifest})
        return final, manifest
    finally:
        # Only this operation's unpublished scratch directory can be removed.
        if temporary.exists():
            import shutil

            shutil.rmtree(temporary)
