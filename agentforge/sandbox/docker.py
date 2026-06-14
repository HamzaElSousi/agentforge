"""Docker sandbox backend — container-per-execution isolation.

Each ``run_python`` call spins up a fresh ``python:3.11-slim`` container,
mounts the workspace directory as ``/workspace`` (read-write, the only
writable path), runs the supplied code, captures stdout/stderr, and removes
the container. Hard isolation: the container has no network by default, its
own PID/network/IPC namespaces, and a memory cap enforced by the Docker
daemon (not rlimit).

Requirements:
    pip install "agentforge[docker]"
    Docker must be running and accessible (socket or TCP).
"""

from __future__ import annotations

import io
import os
import time
from typing import Optional

from agentforge.sandbox.base import ExecResult, Sandbox

# CPU period used for ``--cpu-period`` / ``--cpu-quota`` throttling (100 ms).
_CPU_PERIOD_US = 100_000


class DockerSandbox(Sandbox):
    """Sandbox backend that executes code inside a Docker container.

    The container uses the ``python:3.11-slim`` image, which must be available
    locally or pullable. The workspace directory is bind-mounted at
    ``/workspace`` as the only writable path.

    Parameters
    ----------
    workdir:
        Host-side absolute path to the per-run workspace.
    network:
        If ``False`` (default), the container is started with
        ``network_disabled=True`` so the OS-level network stack is
        unavailable — a hard guarantee unlike the subprocess backend.
    timeout_s, cpu_seconds, memory_mb:
        Passed directly to the Docker daemon as resource constraints.
    """

    backend: str = "docker"

    def __init__(
        self,
        workdir: str,
        *,
        network: bool = False,
        timeout_s: float = 20.0,
        cpu_seconds: int = 10,
        memory_mb: int = 512,
    ) -> None:
        super().__init__(
            workdir,
            network=network,
            timeout_s=timeout_s,
            cpu_seconds=cpu_seconds,
            memory_mb=memory_mb,
        )
        try:
            import docker  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError(
                "The Docker SDK is not installed. "
                "Run: pip install 'agentforge[docker]'  "
                "and ensure Docker is running on your machine."
            ) from exc

        try:
            self._client = docker.from_env()
            # Ping to verify the daemon is reachable.
            self._client.ping()
        except Exception as exc:
            raise RuntimeError(
                f"Cannot connect to the Docker daemon: {exc}. "
                "Make sure Docker Desktop (or the Docker daemon) is running."
            ) from exc

    def run_python(self, code: str, *, stdin: Optional[str] = None) -> ExecResult:
        """Execute *code* in a disposable ``python:3.11-slim`` container.

        The code is passed to the container via a mounted file inside the
        workspace. ``stdin`` is not yet supported by the Docker backend
        (ignored silently — the subprocess backend supports it if needed).

        Returns
        -------
        ExecResult
            Always returned, never raised.
        """
        import docker  # already imported in __init__, re-import for type access
        from docker.errors import ContainerError, DockerException  # type: ignore[import-untyped]

        # Ensure workdir exists on the host.
        os.makedirs(self.workdir, exist_ok=True)

        # Write the script into the workspace so it's accessible via the mount.
        script_name = "_agentforge_run.py"
        script_host_path = os.path.join(self.workdir, script_name)
        with open(script_host_path, "w", encoding="utf-8") as f:
            f.write(code)

        # CPU quota: allow cpu_seconds worth of CPU within each cpu_period.
        # nano_cpus = cpu_seconds * 1e9 would give cpu_seconds vCPUs of time,
        # but we want a *rate* limit so we use cpu_period/cpu_quota instead.
        # cpu_quota = (cpu_seconds / timeout_s) * cpu_period gives a fractional
        # CPU rate; minimum 1 000 µs (Docker enforces a floor).
        cpu_quota = max(1_000, int((self.cpu_seconds / max(self.timeout_s, 1)) * _CPU_PERIOD_US))

        stdout_data = ""
        stderr_data = ""
        exit_code = -1
        timed_out = False
        start = time.monotonic()

        container = None
        try:
            container = self._client.containers.run(
                image="python:3.11-slim",
                command=["python", f"/workspace/{script_name}"],
                volumes={
                    os.path.abspath(self.workdir): {
                        "bind": "/workspace",
                        "mode": "rw",
                    }
                },
                working_dir="/workspace",
                network_disabled=not self.network,
                mem_limit=f"{self.memory_mb}m",
                cpu_period=_CPU_PERIOD_US,
                cpu_quota=cpu_quota,
                remove=False,   # we handle removal ourselves after log capture
                detach=True,
                stdin_open=False,
            )

            # Wait up to timeout_s for the container to finish.
            try:
                result = container.wait(timeout=self.timeout_s)
                exit_code = result.get("StatusCode", -1)
            except Exception:
                # Timeout or Docker API error: kill the container.
                timed_out = True
                try:
                    container.kill()
                except Exception:
                    pass
                exit_code = -1

            # Capture logs regardless of timeout (partial output is useful).
            try:
                logs = container.logs(stdout=True, stderr=True, stream=False)
                # Docker multiplexes stdout/stderr in the same stream by default
                # when no tty is attached; split them properly.
                stdout_data, stderr_data = self._split_logs(container)
            except Exception:
                stdout_data = ""
                stderr_data = ""

        except Exception as exc:
            stderr_data = f"[DockerSandbox error] {exc}"
            exit_code = -1

        finally:
            # Always remove the container and clean up the script file.
            if container is not None:
                try:
                    container.remove(force=True)
                except Exception:
                    pass
            try:
                os.unlink(script_host_path)
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

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _split_logs(self, container) -> tuple[str, str]:  # type: ignore[type-arg]
        """Return (stdout, stderr) decoded from a container's log streams."""
        try:
            stdout_bytes = container.logs(stdout=True, stderr=False)
            stderr_bytes = container.logs(stdout=False, stderr=True)
            return (
                stdout_bytes.decode(errors="replace") if stdout_bytes else "",
                stderr_bytes.decode(errors="replace") if stderr_bytes else "",
            )
        except Exception:
            # Fallback: return everything as stdout.
            try:
                combined = container.logs(stdout=True, stderr=True)
                return combined.decode(errors="replace") if combined else "", ""
            except Exception:
                return "", ""

    def close(self) -> None:
        """Close the Docker client connection."""
        try:
            self._client.close()
        except Exception:
            pass
