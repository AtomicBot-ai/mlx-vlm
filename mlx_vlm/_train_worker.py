"""Supervise a detached training command. This script only needs the stdlib."""

import os
import select
import signal
import subprocess
import sys
from contextlib import suppress
from pathlib import Path


def main():
    # Keep an exited child waitable even if the launching host ignored SIGCHLD.
    signal.signal(signal.SIGCHLD, signal.SIG_DFL)
    child = subprocess.Popen(
        sys.argv[1:], stdin=subprocess.DEVNULL, start_new_session=True
    )
    while child.poll() is None:
        if select.select([sys.stdin], [], [], 0.2)[0]:
            for sig in os.read(sys.stdin.fileno(), 128):
                if sig in (signal.SIGTERM, signal.SIGKILL):
                    # Only this process reaps the group leader, so its PID
                    # cannot be reused between poll() and the signal.
                    with suppress(ProcessLookupError):
                        os.killpg(child.pid, sig)
    code = child.returncode if child.returncode >= 0 else 128 - child.returncode
    Path("exit_code").write_text(str(code))
    return code


if __name__ == "__main__":
    sys.exit(main())
