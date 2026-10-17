"""Own the lifetime of one GR00T policy server process started by `evaluation/serve_policy.sh`.

`PolicyServer` is a context manager: on entry it starts the server in its own process group
(`conda run` spawns children, so the whole group is signalled), logs its output to a file,
and waits until the server prints that it is listening, failing loudly with the log tail if
the process exits or the wait times out. On exit, including Ctrl-C, SIGTERM, SIGHUP and
exceptions, it terminates the whole group, escalating to SIGKILL if it does not stop.
"""

import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from types import FrameType, TracebackType

from evaluation.config import PROJECT_ROOT

SERVE_SCRIPT = PROJECT_ROOT / "evaluation" / "serve_policy.sh"
READY_MESSAGE = "Server is ready and listening"
POLL_INTERVAL_S = 2.0
STOP_GRACE_S = 30.0


class PolicyServerError(RuntimeError):
    """The policy server could not be started or exited unexpectedly."""


def _exit_on_signal(signal_number: int, frame: FrameType | None) -> None:
    sys.exit(128 + signal_number)


def _check_port_free(port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # as ZMQ does, so TIME_WAIT is not "in use"
        try:
            probe.bind(("", port))
        except OSError as error:
            raise PolicyServerError(
                f"Port {port} is already in use ({error}). Stop the process holding it "
                "(`ps aux | grep inference_service`) or choose another --port."
            ) from error


class PolicyServer:
    """Context manager that starts the policy server on `port` and always stops it."""

    def __init__(self, port: int, checkpoint: Path, log_path: Path, ready_timeout_s: float = 900.0) -> None:
        self.port = port
        self.checkpoint = checkpoint
        self.log_path = log_path
        self.ready_timeout_s = ready_timeout_s
        self.process: subprocess.Popen | None = None
        self.previous_handlers: dict[int, object] = {}

    def __enter__(self) -> "PolicyServer":
        try:
            self.start()
        except BaseException:
            self.stop()
            raise
        return self

    def __exit__(self, exc_type: type | None, exc_value: BaseException | None, traceback: TracebackType | None) -> None:
        self.stop()

    def log_tail(self, num_lines: int = 20) -> str:
        if not self.log_path.exists():
            return "(no log written)"
        return "\n".join(self.log_path.read_text(errors="replace").splitlines()[-num_lines:])

    def start(self) -> None:
        """Launch the server and block until it is listening."""
        _check_port_free(self.port)
        for signal_number in (signal.SIGTERM, signal.SIGHUP):
            self.previous_handlers[signal_number] = signal.signal(signal_number, _exit_on_signal)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        command = ["bash", str(SERVE_SCRIPT), "--port", str(self.port), "--checkpoint", str(self.checkpoint)]
        print(f"Starting policy server on port {self.port} (log: {self.log_path})", flush=True)
        with open(self.log_path, "w") as log_file:
            self.process = subprocess.Popen(
                command, stdout=log_file, stderr=subprocess.STDOUT, cwd=PROJECT_ROOT, start_new_session=True
            )
        start_time = time.monotonic()
        while READY_MESSAGE not in self.log_path.read_text(errors="replace"):
            if self.process.poll() is not None:
                raise PolicyServerError(
                    f"Policy server exited with code {self.process.returncode} before it was ready. "
                    f"Last log lines:\n{self.log_tail()}\nFull log: {self.log_path}"
                )
            if time.monotonic() - start_time > self.ready_timeout_s:
                raise PolicyServerError(
                    f"Policy server was not ready after {self.ready_timeout_s:.0f} s. "
                    f"Last log lines:\n{self.log_tail()}\nFull log: {self.log_path}"
                )
            time.sleep(POLL_INTERVAL_S)
        print(f"Policy server ready on port {self.port} after {time.monotonic() - start_time:.0f} s", flush=True)

    def _group_alive(self) -> bool:
        self.process.poll()
        try:
            os.killpg(self.process.pid, 0)
        except ProcessLookupError:
            return False
        return True

    def _signal_group(self, signal_number: int) -> None:
        try:
            os.killpg(self.process.pid, signal_number)
        except ProcessLookupError:
            return

    def stop(self) -> None:
        """Terminate the server's process group (SIGTERM, then SIGKILL) and restore signal handlers."""
        if self.process is not None:
            if self.process.poll() is not None:
                print(
                    f"WARNING: policy server had already exited with code {self.process.returncode}. "
                    f"Last log lines:\n{self.log_tail()}",
                    file=sys.stderr,
                )
            self._signal_group(signal.SIGTERM)
            deadline = time.monotonic() + STOP_GRACE_S
            while self._group_alive() and time.monotonic() < deadline:
                time.sleep(0.5)
            if self._group_alive():
                self._signal_group(signal.SIGKILL)
                self.process.wait()
            print(f"Policy server on port {self.port} stopped", flush=True)
            self.process = None
        for signal_number, handler in self.previous_handlers.items():
            signal.signal(signal_number, handler)
        self.previous_handlers = {}
