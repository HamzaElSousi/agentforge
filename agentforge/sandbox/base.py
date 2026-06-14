"""Abstract ``Sandbox`` — isolation as a provider abstraction.

Like the LLM layer, the sandbox is a swappable backend chosen in YAML. Every
backend runs an untrusted Python snippet under the same contract: no network
(unless explicitly allowed), a workspace-jailed working directory, resource
limits, and a wall-clock timeout. The result is always captured, never thrown.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class ExecResult:
    """Outcome of running code in a sandbox. Non-raising by contract."""

    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    timed_out: bool = False
    duration_ms: int = 0

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out

    def summary(self) -> str:
        if self.timed_out:
            return f"[timed out after {self.duration_ms} ms]\n{self.stdout}"
        if self.exit_code != 0:
            return f"[exit {self.exit_code}]\n{self.stdout}\n{self.stderr}".strip()
        return self.stdout


class Sandbox(ABC):
    """Run untrusted code with isolation guarantees.

    Parameters
    ----------
    workdir:
        Absolute path to the per-run workspace; the only writable directory and
        the process CWD.
    network:
        Whether the executed code may reach the network. Default False.
    timeout_s:
        Wall-clock limit; on expiry the process is killed and
        ``ExecResult.timed_out`` is set.
    cpu_seconds / memory_mb:
        Soft resource caps where the backend supports them (subprocess rlimits,
        Docker ``--cpus``/``--memory``).
    """

    backend: str = "base"

    def __init__(
        self,
        workdir: str,
        *,
        network: bool = False,
        timeout_s: float = 20.0,
        cpu_seconds: int = 10,
        memory_mb: int = 512,
    ) -> None:
        self.workdir = workdir
        self.network = network
        self.timeout_s = timeout_s
        self.cpu_seconds = cpu_seconds
        self.memory_mb = memory_mb

    @abstractmethod
    def run_python(self, code: str, *, stdin: Optional[str] = None) -> ExecResult:
        """Execute a Python snippet and capture its output. Never raises for
        normal failures (timeouts, non-zero exit) — those are reported in the
        :class:`ExecResult`."""
        raise NotImplementedError

    def __enter__(self) -> "Sandbox":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:  # pragma: no cover - optional cleanup
        """Tear down any backend resources (containers, remote VMs)."""
