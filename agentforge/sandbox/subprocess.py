"""Subprocess sandbox backend — the default, dependency-free sandbox.

Isolation model:
- Runs untrusted code in a child process using ``sys.executable``.
- Wall-clock timeout is enforced via ``subprocess.Popen``; on expiry the entire
  process *group* is killed via ``os.killpg`` (POSIX) so no orphan threads survive.
- Resource limits (CPU time, virtual memory, file size) are applied via a
  ``preexec_fn`` on POSIX systems using ``resource.setrlimit``. On non-POSIX
  platforms (pure Windows) these limits silently degrade — the process still
  runs, but without hard kernel-level caps.
- **Network isolation is best-effort**: real namespace isolation requires root or
  ``CAP_NET_ADMIN`` and is not available to unprivileged processes. Instead, when
  ``network=False`` a small preamble is prepended to the executed code that
  monkeypatches ``socket.socket`` to raise ``OSError`` on construction. This
  blocks the common case (requests, httpx, urllib) but determined code could work
  around it. For hard network isolation use the Docker or E2B backends.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
import time
from typing import Optional

from agentforge.sandbox.base import ExecResult, Sandbox

# ---------------------------------------------------------------------------
# Network-isolation preamble injected when network=False
# ---------------------------------------------------------------------------

_NETWORK_BLOCK_PREAMBLE = """\
import socket as _socket_module
_original_socket = _socket_module.socket

class _BlockedSocket(_original_socket):
    def __init__(self, *args, **kwargs):
        raise OSError(
            "Network access is disabled in this sandbox (network=False). "
            "Use the Docker or E2B backend for hard network isolation."
        )

_socket_module.socket = _BlockedSocket
del _BlockedSocket, _original_socket, _socket_module
"""

# ---------------------------------------------------------------------------
# Resource limit helper (POSIX only)
# ---------------------------------------------------------------------------

_IS_POSIX = hasattr(os, "setsid")


def _make_preexec(cpu_seconds: int, memory_mb: int):  # type: ignore[return]
    """Return a ``preexec_fn`` that sets rlimits and creates a new process group.

    Guarded behind a POSIX check; returns ``None`` on non-POSIX platforms so
    ``subprocess.Popen`` runs without a ``preexec_fn``.
    """
    if not _IS_POSIX:
        return None

    import resource  # only available on POSIX

    def _preexec() -> None:
        # New process group so we can kill all descendants on timeout.
        os.setsid()

        # CPU time: RLIMIT_CPU (seconds). Hard limit = soft + 1 so the kernel
        # sends SIGKILL shortly after SIGXCPU if the process ignores it.
        soft_cpu = max(1, cpu_seconds)
        try:
            resource.setrlimit(resource.RLIMIT_CPU, (soft_cpu, soft_cpu + 1))
        except (ValueError, resource.error):
            pass  # degrade gracefully if the limit can't be set

        # Virtual memory: RLIMIT_AS (bytes).
        mem_bytes = memory_mb * 1024 * 1024
        try:
            resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
        except (ValueError, resource.error):
            pass

        # File size cap: 64 MiB — prevents runaway file writes.
        fsize_bytes = 64 * 1024 * 1024
        try:
            resource.setrlimit(resource.RLIMIT_FSIZE, (fsize_bytes, fsize_bytes))
        except (ValueError, resource.error):
            pass

    return _preexec


# ---------------------------------------------------------------------------
# Sandbox implementation
# ---------------------------------------------------------------------------


class SubprocessSandbox(Sandbox):
    """Default sandbox backend: runs code in a child subprocess.

    Uses ``sys.executable`` so the code runs inside the same Python
    installation as AgentForge, picking up any installed third-party packages.
    Resource limits are applied via ``resource.setrlimit`` on POSIX; a
    best-effort network block is injected via monkeypatching when ``network``
    is ``False``.
    """

    backend: str = "subprocess"

    def run_python(self, code: str, *, stdin: Optional[str] = None) -> ExecResult:
        """Write *code* to a temp file and execute it as a child process.

        Parameters
        ----------
        code:
            Python source code to execute.
        stdin:
            Optional text passed to the child process's standard input.

        Returns
        -------
        ExecResult
            Always returned, never raised. ``timed_out`` is set when the
            wall-clock limit is exceeded; ``exit_code`` reflects the process
            exit status otherwise.
        """
        # Optionally prepend the network block preamble.
        full_code = (_NETWORK_BLOCK_PREAMBLE + "\n" + code) if not self.network else code

        # Ensure the workdir exists.
        os.makedirs(self.workdir, exist_ok=True)

        preexec_fn = _make_preexec(self.cpu_seconds, self.memory_mb)

        # Write code to a temp file inside workdir so it's visible as a path in
        # tracebacks and the CWD is already jailed to workdir.
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            dir=self.workdir,
            delete=False,
            encoding="utf-8",
        ) as tmp:
            tmp.write(full_code)
            script_path = tmp.name

        stdout_data = ""
        stderr_data = ""
        exit_code = -1
        timed_out = False
        start = time.monotonic()

        try:
            proc = subprocess.Popen(
                [sys.executable, script_path],
                cwd=self.workdir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
                preexec_fn=preexec_fn,
            )

            try:
                raw_out, raw_err = proc.communicate(
                    input=stdin.encode() if stdin is not None else None,
                    timeout=self.timeout_s,
                )
                stdout_data = raw_out.decode(errors="replace")
                stderr_data = raw_err.decode(errors="replace")
                exit_code = proc.returncode
            except subprocess.TimeoutExpired:
                timed_out = True
                # Kill the entire process group (POSIX) or just the process.
                if _IS_POSIX:
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                else:
                    proc.kill()
                # Drain pipes after killing so communicate() doesn't block.
                raw_out, raw_err = proc.communicate()
                stdout_data = raw_out.decode(errors="replace")
                stderr_data = raw_err.decode(errors="replace")
                exit_code = proc.returncode if proc.returncode is not None else -1

        finally:
            # Clean up the temp script file.
            try:
                os.unlink(script_path)
            except OSError:
                pass

        duration_ms = int((time.monotonic() - start) * 1000)

        return ExecResult(
            stdout=stdout_data,
            stderr=stderr_data,
            exit_code=exit_code,
            timed_out=timed_out,
            duration_ms=duration_ms,
        )
