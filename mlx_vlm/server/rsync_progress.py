# Copyright © 2026 Neywa Labs Inc.
"""Stream rsync file progress without requiring rsync 3's --info=progress2."""

import os
import re
import selectors
import signal
import subprocess
import tempfile
import time

FILE_FORMAT = "MLX_VLM_FILE:%i:%l"
_FILE = re.compile(rb"^MLX_VLM_FILE:([^:]+):(\d+)$")
_PROGRESS = re.compile(rb"^\s*([\d,]+)\s+\d{1,3}%\s+")


class RsyncProgress:
    """Sum logical file bytes, keeping checksum-reused files separate."""

    def __init__(self, total_bytes, callback, interval=0.5):
        self.total_bytes = total_bytes
        self.callback = callback
        self.interval = interval
        self.pending = b""
        self.completed = 0
        self.current = 0
        self.size = None
        self.reused = 0
        self.last_report = 0
        self.last_counts = (0, 0)

    def feed(self, data):
        lines = re.split(rb"[\r\n]", self.pending + data)
        self.pending = lines.pop()
        # Diagnostics or unexpected output must not grow memory indefinitely.
        if len(self.pending) > 8192:
            self.pending = b""
        for line in lines:
            self.line(line)
        self.report()

    def line(self, line):
        item = _FILE.fullmatch(line)
        if item:
            self.completed += self.current
            self.current, self.size = 0, None
            flags, size = item.group(1), int(item.group(2))
            if len(flags) < 2 or flags[1:2] != b"f":
                return  # Directory metadata is not checkpoint payload.
            if flags[:1] in (b">", b"<"):
                self.size = size
            elif flags[:1] == b".":
                self.reused += size  # --checksum established identical content.
        elif self.size is not None:
            progress = _PROGRESS.match(line)
            if progress:
                count = int(progress.group(1).replace(b",", b""))
                self.current = max(self.current, min(count, self.size))

    def report(self, *, force=False):
        counts = (self.completed + self.current, self.reused)
        now = time.monotonic()
        if counts != self.last_counts and (
            force or now - self.last_report >= self.interval
        ):
            self.callback(
                bytes_transferred=counts[0],
                bytes_reused=counts[1],
                total_bytes=self.total_bytes,
            )
            self.last_report, self.last_counts = now, counts

    def finish(self):
        if self.pending:
            self.line(self.pending)
            self.pending = b""
        self.report(force=True)


def run_rsync(args, *, input, timeout, total_bytes, progress, stop=None):
    """Drain both pipes live, retaining only a bounded diagnostic tail.

    Double itemization includes unchanged files on rsync 2.6.7+, including
    macOS's bundled openrsync. No filenames are needed in the progress protocol.
    """
    args = [
        args[0],
        "--progress",
        "-ii",
        "--out-format=" + FILE_FORMAT,
        *args[1:],
    ]
    tracker = RsyncProgress(total_bytes, progress)
    stderr = b""
    deadline = time.monotonic() + timeout
    # A file avoids a pipe deadlock for manifests larger than the stdin buffer.
    with tempfile.TemporaryFile() as manifest, selectors.DefaultSelector() as selector:
        manifest.write(input.encode())
        manifest.seek(0)
        with subprocess.Popen(
            args,
            stdin=manifest,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={**os.environ, "LC_ALL": "C"},
            start_new_session=True,
        ) as process:
            try:
                for stream in (process.stdout, process.stderr):
                    selector.register(stream, selectors.EVENT_READ)
                while selector.get_map():
                    if stop is not None and stop.is_set():
                        raise InterruptedError("Checkpoint preparation cancelled")
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise subprocess.TimeoutExpired(args, timeout, stderr=stderr)
                    for key, _ in selector.select(min(remaining, 0.5)):
                        data = os.read(key.fd, 8192)
                        if not data:
                            selector.unregister(key.fileobj)
                        elif key.fileobj is process.stdout:
                            tracker.feed(data)
                        else:
                            stderr = (stderr + data)[-4096:]
                code = process.wait(timeout=max(0.001, deadline - time.monotonic()))
                tracker.finish()
                return subprocess.CompletedProcess(
                    args, code, "", stderr.decode("utf-8", errors="replace")
                )
            except BaseException:
                # This session contains only our rsync and its SSH descendants.
                # Never leave a copy running after timeout or callback failure.
                if process.returncode is None:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
                raise
