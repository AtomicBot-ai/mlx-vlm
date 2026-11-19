# Copyright © 2026 Neywa Labs Inc.
"""SSH-controlled, leased transfer helper. Never joins an inference group.

This file and checkpoint_store are bootstrapped as source, allowing approved
workers to use their own Python/MLX installation without identical repo paths.
Weight bytes use JACCL; commands, manifests and progress use the SSH channel.
"""

import concurrent.futures
import fcntl
import hashlib
import json
import os
import platform
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from pathlib import Path

try:
    from . import checkpoint_store as store
except ImportError:  # Standalone, authenticated source bootstrap.
    import mlx_checkpoint_store as store

PREFIX = "MLX_TRANSFER "


class RDMAUnavailable(RuntimeError):
    """A recognized JACCL connection failure, before checkpoint delivery."""


def connection_call(function, *args, **kwargs):
    # JACCL currently exposes native connection errors as RuntimeError strings.
    # Match only known connection failures, never arbitrary backend exceptions.
    try:
        return function(*args, **kwargs)
    except RuntimeError as error:
        if re.fullmatch(
            r"\[jaccl\] (?:(?:Send|Recv) failed with errno=\d+|"
            r"(?:Couldn't connect|Accept failed) \(error: \d+\)|"
            r"Could not find device rdma_\w+ \(\d+ available\))",
            str(error),
        ):
            raise RDMAUnavailable(str(error)) from error
        raise


def emit(event, **data):
    print(PREFIX + json.dumps({"event": event, **data}), flush=True)


def physical_id():
    # Same physical identity contract as cluster_identity, with stdlib-only startup.
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


def copy_chunks(group, command, report):
    import mlx.core as mx
    import numpy as np

    manifest = store.validate(command["manifest"])
    root = Path(command["path"])
    sending = group.rank() == 0
    stream = mx.new_stream(mx.cpu)
    chunk, depth = manifest["chunk_bytes"], 2
    idle, ready = queue.Queue(depth), queue.Queue(depth)
    errors, signatures = [], {}
    halted = threading.Event()
    arrays = []
    read = received = verified = 0
    started = time.monotonic()
    last = 0

    def progress(force=False):
        nonlocal last
        now = time.monotonic()
        if force or now - last >= 0.2:
            last = now
            report(
                bytes_read=read,
                bytes_received=received,
                bytes_verified=verified,
                elapsed_seconds=now - started,
            )

    def put(q, value):
        while not halted.is_set():
            try:
                q.put(value, timeout=0.1)
                return
            except queue.Full:
                pass
        raise RuntimeError("Transfer pipeline failed")

    def get(q):
        while not halted.is_set():
            try:
                return q.get(timeout=0.1)
            except queue.Empty:
                pass
        raise RuntimeError("Transfer pipeline failed")

    # CPU buffers must remain alive and unmodified through explicit send eval.
    if sending:
        for slot in range(depth):
            value = mx.zeros((chunk,), mx.uint8, stream=mx.cpu)
            mx.eval(value)
            view = memoryview(np.asarray(value))
            if view.readonly:
                raise RuntimeError("MLX CPU buffers are not writable in this version")
            arrays.append((value, view))
            idle.put(slot)

    def reader():
        nonlocal read
        try:
            for name, record in manifest["files"].items():
                path = store.safe_path(root, name)
                before = store.signature(path)
                with path.open("rb", buffering=0) as source:
                    for index in command["missing"][name]:
                        n = min(chunk, record["size"] - index * chunk)
                        slot = get(idle)
                        view = arrays[slot][1]
                        source.seek(index * chunk)
                        got = 0
                        while got < n:
                            count = source.readinto(view[got:n])
                            if not count:
                                raise OSError("Short checkpoint read")
                            got += count
                        read += n
                        progress()
                        put(ready, (slot, n))
                if store.signature(path) != before:
                    raise ValueError("Source checkpoint changed during transfer")
        except BaseException as error:
            errors.append(error)
            halted.set()

    def writer():
        nonlocal verified
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                pending = deque()

                def check(view, expected):
                    if hashlib.sha256(view).hexdigest() != expected:
                        raise ValueError("Checkpoint chunk checksum mismatch")
                    return len(view)

                for name, record in manifest["files"].items():
                    path = store.safe_path(root, name)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with path.open(
                        "r+b" if path.exists() else "xb", buffering=0
                    ) as destination:
                        if sys.platform == "darwin":
                            # Streaming checkpoints should not evict the active model
                            # or accumulate hundreds of GB in the receiver's file cache.
                            fcntl.fcntl(destination.fileno(), fcntl.F_NOCACHE, 1)
                        destination.truncate(record["size"])
                        for index in command["missing"][name]:
                            value, view = get(ready)
                            pending.append(
                                pool.submit(check, view, record["chunks"][index])
                            )
                            destination.seek(index * chunk)
                            remaining = view
                            while remaining:
                                count = destination.write(remaining)
                                if not count:
                                    raise OSError("Short checkpoint write")
                                remaining = remaining[count:]
                            if len(pending) >= 4:
                                verified += pending.popleft().result()
                            progress()
                            del view, value
                        while pending:
                            verified += pending.popleft().result()
                        store.flush_file(destination)
                    signatures[name] = store.signature(path)
        except BaseException as error:
            errors.append(error)
            halted.set()

    thread = threading.Thread(target=reader if sending else writer, daemon=True)
    thread.start()
    try:
        if sending:
            for name, record in manifest["files"].items():
                for index in command["missing"][name]:
                    slot, n = get(ready)
                    value = arrays[slot][0]
                    mx.eval(
                        mx.distributed.send(
                            value if n == chunk else value[:n],
                            1,
                            group=group,
                            stream=stream,
                        )
                    )
                    put(idle, slot)
        else:
            pending = deque()
            for name, record in manifest["files"].items():
                for index in command["missing"][name]:
                    n = min(chunk, record["size"] - index * chunk)
                    value = mx.distributed.recv(
                        (n,), mx.uint8, 0, group=group, stream=stream
                    )
                    mx.async_eval(value)
                    pending.append(value)
                    if len(pending) >= depth:
                        value = pending.popleft()
                        mx.eval(value)
                        view = memoryview(np.asarray(value))
                        received += len(view)
                        put(ready, (value, view))
            while pending:
                value = pending.popleft()
                mx.eval(value)
                view = memoryview(np.asarray(value))
                received += len(view)
                put(ready, (value, view))
        thread.join()
        if errors:
            raise errors[0]
        progress(True)
        return signatures
    finally:
        halted.set()


def stream_chunks(group, command, report):
    """Hash each source byte as it enters a bounded, striped RDMA pipeline."""
    import mlx.core as mx
    import numpy as np

    layout = store.validate_layout(command["layout"])
    root, chunk = Path(command["path"]), layout["chunk_bytes"]
    sending = group.rank() == 0
    stream = mx.new_stream(mx.cpu)
    halted, lock = threading.Event(), threading.Lock()
    ready = queue.Queue(2)
    errors, records, signatures = [], {}, {}
    counts = dict(bytes_read=0, bytes_received=0, bytes_verified=0, bytes_reused=0)
    last = 0

    def progress(force=False, **data):
        nonlocal last
        with lock:
            for key, value in data.items():
                counts[key] += value
            now = time.monotonic()
            if force or now - last >= 0.2:
                report(**counts)
                last = now

    def exchange(q, value=None, *, put=False):
        while not halted.is_set():
            try:
                return q.put(value, timeout=0.1) if put else q.get(timeout=0.1)
            except (queue.Empty, queue.Full):
                pass
        raise errors[0] if errors else RuntimeError("Transfer pipeline failed")

    def reader(name, buffers, idle, outgoing):
        try:
            path = store.safe_path(root, name)
            before = store.signature(path)
            record = layout["files"][name]
            if before[2] != record["size"]:
                raise ValueError("Source checkpoint size changed")
            digest, hashes = hashlib.sha256(), []
            with path.open("rb", buffering=0) as source:
                for offset in range(0, record["size"], chunk):
                    slot = exchange(idle)
                    n = min(chunk, record["size"] - offset)
                    view = buffers[slot][1][:n]
                    got = 0
                    while got < n:
                        count = source.readinto(view[got:])
                        if not count:
                            raise OSError("Short checkpoint read")
                        got += count
                    digest.update(view)
                    hashed = hashlib.sha256(view).hexdigest()
                    hashes.append(hashed)
                    progress(bytes_read=n)
                    exchange(outgoing, (slot, hashed), put=True)
                if (
                    record.get("expected_sha256")
                    and digest.hexdigest() != record["expected_sha256"]
                ):
                    raise ValueError(f"Hugging Face checksum mismatch: {name}")
                store.flush_file(source)
            if store.signature(path) != before:
                raise ValueError("Source checkpoint changed during transfer")
            records[name] = {
                "size": record["size"],
                "sha256": digest.hexdigest(),
                "chunks": hashes,
            }
        except BaseException as error:
            errors.append(error)
            halted.set()
            raise

    def writer():
        destinations, pending = {}, deque()
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as hash_pool:

                def check(view, expected):
                    if hashlib.sha256(view).hexdigest() != expected:
                        raise ValueError("Checkpoint chunk checksum mismatch")
                    return len(view)

                while True:
                    item = exchange(ready)
                    if item is None:
                        break
                    name, index, expected, value, view = item
                    record = layout["files"][name]
                    if name not in destinations:
                        path = store.safe_path(root, name)
                        path.parent.mkdir(parents=True, exist_ok=True)
                        destination = path.open(
                            "r+b" if path.exists() else "xb", buffering=0
                        )
                        destinations[name] = destination
                        if sys.platform == "darwin":
                            fcntl.fcntl(destination.fileno(), fcntl.F_NOCACHE, 1)
                        destination.truncate(record["size"])
                        records[name] = {"size": record["size"], "chunks": []}
                    destination = destinations[name]
                    destination.seek(index * chunk)
                    if view is None:
                        # Recheck reused bytes while consuming the sender's final hash.
                        view = memoryview(
                            destination.read(min(chunk, record["size"] - index * chunk))
                        )
                        progress(bytes_reused=len(view))
                    else:
                        remaining = view
                        while remaining:
                            count = destination.write(remaining)
                            if not count:
                                raise OSError("Short checkpoint write")
                            remaining = remaining[count:]
                    pending.append(hash_pool.submit(check, view, expected))
                    records[name]["chunks"].append(expected)
                    if len(pending) >= 4:
                        progress(bytes_verified=pending.popleft().result())
                while pending:
                    progress(bytes_verified=pending.popleft().result())
            for name, record in layout["files"].items():
                path = store.safe_path(root, name)
                if name not in destinations:  # Empty files have no payload chunks.
                    path.parent.mkdir(parents=True, exist_ok=True)
                    destinations[name] = path.open("wb", buffering=0)
                    records[name] = {"size": 0, "chunks": []}
                store.flush_file(destinations[name])
                signatures[name] = store.signature(path)
        except BaseException as error:
            errors.append(error)
            halted.set()
        finally:
            for destination in destinations.values():
                destination.close()

    thread = None
    if not sending:
        thread = threading.Thread(target=writer, daemon=True)
        thread.start()
    names = sorted(layout["files"], key=lambda n: (-layout["files"][n]["size"], n))
    readers = concurrent.futures.ThreadPoolExecutor(max_workers=8)
    try:
        # Eight files in parallel preserve whole-file SHA-256 throughput. Each lane
        # owns two 16 MiB CPU buffers; source size never determines memory usage.
        for start in range(0, len(names), 8):
            lanes = names[start : start + 8]
            buffers, idles, outgoing, futures = {}, {}, {}, []
            if sending:
                for name in lanes:
                    buffers[name], idles[name], outgoing[name] = (
                        [],
                        queue.Queue(2),
                        queue.Queue(2),
                    )
                    for slot in range(2):
                        value = mx.zeros((chunk,), mx.uint8, stream=mx.cpu)
                        mx.eval(value)
                        view = memoryview(np.asarray(value))
                        if view.readonly:
                            raise RuntimeError("MLX CPU buffers are not writable")
                        buffers[name].append((value, view))
                        idles[name].put(slot)
                    futures.append(
                        readers.submit(
                            reader, name, buffers[name], idles[name], outgoing[name]
                        )
                    )
            rounds = max(
                (layout["files"][n]["size"] + chunk - 1) // chunk for n in lanes
            )
            for index in range(rounds):
                active = [
                    n for n in lanes if index * chunk < layout["files"][n]["size"]
                ]
                if sending:
                    batch, header = [], bytearray()
                    for name in active:
                        slot, hashed = exchange(outgoing[name])
                        reused = command["partial_hashes"][name][index] == hashed
                        header.extend(bytes([int(reused)]) + bytes.fromhex(hashed))
                        batch.append((name, slot, reused))
                    mx.eval(
                        mx.distributed.send(
                            mx.array(np.frombuffer(header, dtype=np.uint8)),
                            1,
                            group=group,
                            stream=stream,
                        )
                    )
                    for name, slot, reused in batch:
                        n = min(chunk, layout["files"][name]["size"] - index * chunk)
                        if not reused:
                            value = buffers[name][slot][0]
                            mx.eval(
                                mx.distributed.send(
                                    value if n == chunk else value[:n],
                                    1,
                                    group=group,
                                    stream=stream,
                                )
                            )
                        exchange(idles[name], slot, put=True)
                else:
                    header = mx.distributed.recv(
                        (33 * len(active),), mx.uint8, 0, group=group, stream=stream
                    )
                    mx.eval(header)
                    header = np.asarray(header).tobytes()
                    pending = deque()

                    def consume():
                        name, expected, value = pending.popleft()
                        if value is None:
                            view = None
                        else:
                            mx.eval(value)
                            view = memoryview(np.asarray(value))
                            progress(bytes_received=len(view))
                        exchange(ready, (name, index, expected, value, view), put=True)

                    for lane, name in enumerate(active):
                        reused = header[lane * 33]
                        if reused not in (0, 1):
                            raise ValueError("Invalid checkpoint chunk header")
                        expected = header[lane * 33 + 1 : lane * 33 + 33].hex()
                        n = min(chunk, layout["files"][name]["size"] - index * chunk)
                        value = None
                        if not reused:
                            value = mx.distributed.recv(
                                (n,), mx.uint8, 0, group=group, stream=stream
                            )
                            mx.async_eval(value)
                        pending.append((name, expected, value))
                        if len(pending) >= 2:
                            consume()
                    while pending:
                        consume()
            for future in futures:
                future.result()
        if thread:
            exchange(ready, None, put=True)
            thread.join()
        if errors:
            raise errors[0]
        progress(True)
        return {
            "manifest": {"version": 1, "chunk_bytes": chunk, "files": records},
            "signatures": signatures,
        }
    finally:
        halted.set()
        readers.shutdown(wait=True, cancel_futures=True)
        if thread:
            thread.join()


def partial_hashes(path, layout, report):
    """Inspect actual partial bytes; a resume bitmap alone cannot establish reuse."""
    chunk, checked, hashes = layout["chunk_bytes"], 0, {}
    for name, record in layout["files"].items():
        target = store.safe_path(path, name)
        values = []
        source = target.open("rb", buffering=0) if target.exists() else None
        try:
            for offset in range(0, record["size"], chunk):
                n = min(chunk, record["size"] - offset)
                data = source.read(n) if source else b""
                values.append(
                    hashlib.sha256(data).hexdigest() if len(data) == n else None
                )
                checked += len(data)
                if data:
                    report(bytes_checked=checked)
        finally:
            if source:
                source.close()
        hashes[name] = values
    return hashes


def check_partial(current, manifest, report):
    checked, last_report = 0, 0

    def checking(n):
        nonlocal checked, last_report
        checked += n
        now = time.monotonic()
        if now - last_report >= 0.2:
            report(bytes_checked=checked)
            last_report = now

    missing, reused = current.missing(manifest, progress=checking)
    report(bytes_checked=checked)
    return {
        "ready": False,
        "missing": missing,
        "bytes_reused": reused,
        "path": str(current.partial(manifest)),
    }


def main():
    commands = queue.Queue()
    renewed = [time.monotonic()]

    # Native RDMA may block. EOF/lease expiry must kill this helper independently.
    def control():
        try:
            for line in sys.stdin:
                message = json.loads(line)
                renewed[0] = time.monotonic()
                if message["op"] == "shutdown":
                    os._exit(0)
                if message["op"] != "ping":
                    commands.put(message)
        finally:
            os._exit(0)

    def watchdog():
        while True:
            time.sleep(1)
            if time.monotonic() - renewed[0] > 30:
                os._exit(124)

    threading.Thread(target=control, daemon=True).start()
    threading.Thread(target=watchdog, daemon=True).start()
    group = None
    lock = None
    current = None
    signatures = None
    pipeline = None
    layout = None
    emit("hello", physical_id=physical_id(), pid=os.getpid(), protocol=1)
    with tempfile.TemporaryDirectory(prefix="mlx-transfer-") as scratch:
        while True:
            command = commands.get()
            request = command["id"]
            op = command["op"]
            report = lambda **data: emit("progress", id=request, **data)
            try:
                result = {}
                if op == "init":
                    import mlx.core as mx
                    import numpy as np

                    if group is not None:
                        raise ValueError("Transfer group is already initialized")
                    if not mx.distributed.is_available("jaccl"):
                        raise RDMAUnavailable("JACCL is unavailable on this worker")
                    for name in list(os.environ):
                        if name.startswith(
                            ("MLX_JACCL_", "MLX_RING_", "JACCL_")
                        ) or name in {"MLX_HOSTFILE", "MLX_RANK", "MLX_IBV_DEVICES"}:
                            del os.environ[name]
                    devices = Path(scratch) / "devices.json"
                    devices.write_text(json.dumps(command["devices"]))
                    os.environ.update(
                        MLX_RANK=str(command["rank"]),
                        MLX_IBV_DEVICES=str(devices),
                        MLX_JACCL_COORDINATOR=command["coordinator"],
                    )
                    emit("initializing", id=request)
                    group = connection_call(
                        mx.distributed.init, backend="jaccl", strict=True
                    )
                    if group.size() != 2 or group.rank() != command["rank"]:
                        raise ValueError("Unexpected transfer group membership")
                    # Verify this pair can communicate using a session-specific challenge.
                    token = np.frombuffer(
                        bytes.fromhex(command["challenge"]), dtype=np.uint8
                    ).astype(np.int32)
                    probe = mx.distributed.all_sum(
                        mx.array(token), group=group, stream=mx.cpu
                    )
                    connection_call(mx.eval, probe)
                    if not np.array_equal(np.asarray(probe), token * 2):
                        raise ValueError("RDMA connection challenge mismatch")
                    result = {"rdma_validated": True, "mlx_version": mx.__version__}
                elif op in {"begin", "begin_stream"}:
                    if lock is not None:
                        raise ValueError("A checkpoint transaction is already active")
                    current = store.SnapshotStore(
                        command["root"], partial_id=command.get("partial_id")
                    )
                    pipeline = {} if op == "begin_stream" else None
                    layout = (
                        store.validate_layout(command["layout"])
                        if pipeline is not None
                        else None
                    )
                    signatures = {}
                    if pipeline is not None:
                        if current.partial_id is None:
                            raise ValueError("Missing checkpoint transaction ID")
                        manifest = None
                    else:
                        manifest = store.validate(command["manifest"])
                        # Resume pre-pipeline transfers when the complete source
                        # manifest was already cached before this operation.
                        if (
                            current.partial_id is not None
                            and (
                                current.root
                                / (".partial-" + store.snapshot_id(manifest))
                            ).exists()
                        ):
                            current = store.SnapshotStore(command["root"])
                    lock_id = current.partial_id or store.snapshot_id(manifest)
                    lock = (current.receipts / (lock_id + ".lock")).open("a")
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    if pipeline is not None:
                        partial = current.writable_partial(None)
                        emit(
                            "result",
                            id=request,
                            path=str(partial),
                            partial_hashes=partial_hashes(partial, layout, report),
                        )
                        continue
                    final = current.existing(manifest)
                    if final:
                        lock.close()
                        lock = None
                        result = {"ready": True, "path": str(final)}
                    else:
                        current.writable_partial(manifest)
                        result = check_partial(current, manifest, report)
                elif op in {"send_stream", "receive_stream"}:
                    if group is None:
                        raise ValueError("No validated transfer group")
                    if op == "receive_stream" and (
                        lock is None
                        or layout != command["layout"]
                        or str(current.partial(None)) != command["path"]
                    ):
                        raise ValueError("No matching checkpoint transaction")
                    result = stream_chunks(group, command, report)
                    if op == "receive_stream":
                        signatures = result["signatures"]
                        pipeline = result["manifest"]["files"]
                        result = {}
                elif op in {"send", "receive"}:
                    if group is None:
                        raise ValueError("No validated transfer group")
                    if op == "send":
                        source = store.SnapshotStore(command["root"])
                        final = source.existing(command["manifest"])
                        if final is None or str(final) != command["path"]:
                            raise ValueError("Unverified transfer source")
                    elif (
                        lock is None
                        or str(current.partial(command["manifest"])) != command["path"]
                    ):
                        raise ValueError("No matching checkpoint transaction")
                    signatures = copy_chunks(group, command, report)
                elif op == "commit":
                    if lock is None or signatures is None:
                        raise ValueError("No verified transfer to commit")
                    manifest = store.validate(command["manifest"])
                    if pipeline is not None and (
                        manifest["chunk_bytes"] != layout["chunk_bytes"]
                        or pipeline
                        != {
                            n: {"size": r["size"], "chunks": r["chunks"]}
                            for n, r in manifest["files"].items()
                        }
                    ):
                        raise ValueError("Incomplete or changed pipeline manifest")
                    # Another preparation may already have published the same bytes.
                    final = current.existing(manifest) if pipeline is not None else None
                    if final:
                        shutil.rmtree(current.partial(manifest))
                    else:
                        final = current.publish(manifest, signatures)
                    result = {"path": str(final)}
                    lock.close()
                    lock = None
                    signatures = None
                    pipeline = None
                elif op == "verify":
                    if lock is None:
                        raise ValueError("No checkpoint transaction")
                    manifest = store.validate(command["manifest"])
                    partial = current.writable_partial(manifest)
                    checked = 0
                    last_report = 0

                    def verified(n):
                        nonlocal checked, last_report
                        checked += n
                        now = time.monotonic()
                        if now - last_report >= 0.2:
                            report(bytes_verified=checked)
                            last_report = now

                    signatures = {}
                    for name, record in manifest["files"].items():
                        path = store.safe_path(partial, name)
                        store.verify_chunks(
                            path, record, manifest["chunk_bytes"], progress=verified
                        )
                        with path.open("rb") as stream:
                            store.flush_file(stream)
                        signatures[name] = store.signature(path)
                    report(bytes_verified=checked)
                else:
                    raise ValueError("Unknown transfer command")
                emit("result", id=request, **result)
            except BaseException as error:
                emit(
                    "error",
                    id=request,
                    message=str(error),
                    code=(
                        "rdma_unavailable"
                        if isinstance(error, RDMAUnavailable)
                        else "helper_error"
                    ),
                )
                # Never reuse a collective after an incomplete send/recv sequence.
                os._exit(1)


if __name__ == "__main__":
    main()
