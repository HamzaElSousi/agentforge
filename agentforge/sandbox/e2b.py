"""E2B sandbox backend — cloud micro-VM execution (optional).

Each ``run_python`` call executes the code inside an E2B cloud sandbox, a
fully isolated micro-VM managed by the E2B platform. This backend provides the
strongest isolation available (separate kernel, real network namespace control)
at the cost of a network round-trip and an E2B API key.

Requirements:
    pip install "agentforge[e2b]"
    Set the environment variable ``E2B_API_KEY`` to your E2B API key.
    Sign up at https://e2b.dev to obtain a key.
"""

from __future__ import annotations

import os
import time
from typing import Optional

from agentforge.sandbox.base import ExecResult, Sandbox


class E2BSandbox(Sandbox):
    """Sandbox backend that runs code inside an E2B cloud micro-VM.

    The E2B platform provisions a fresh micro-VM per session. This sandbox
    creates one session in ``__init__`` and reuses it across ``run_python``
    calls (the session is closed by :meth:`close` or the context manager).

    Parameters
    ----------
    workdir:
        A logical workspace label used as the execution directory inside the
        E2B sandbox (e.g. ``/home/user/workspace``). Files are not
        automatically synced from the host — the sandbox is cloud-hosted.
    network:
        Whether the sandbox may reach the internet. Passed to the E2B sandbox
        configuration where the API supports it.
    timeout_s:
        Per-execution wall-clock limit in seconds.
    cpu_seconds, memory_mb:
        Advisory limits — E2B enforces its own resource caps per plan tier;
        these values are surfaced in the config for consistency but may not be
        directly honoured by the E2B API.
    """

    backend: str = "e2b"

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
            from e2b_code_interpreter import Sandbox as _E2BSandboxClass  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError(
                "The e2b_code_interpreter package is not installed. "
                "Run: pip install 'agentforge[e2b]'  "
                "and set the E2B_API_KEY environment variable."
            ) from exc

        api_key = os.environ.get("E2B_API_KEY")
        if not api_key:
            raise RuntimeError(
                "E2B_API_KEY environment variable is not set. "
                "Obtain a key from https://e2b.dev and set it before using the E2B backend."
            )

        try:
            self._sandbox = _E2BSandboxClass(api_key=api_key)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to create an E2B sandbox session: {exc}. "
                "Check your E2B_API_KEY and network connectivity."
            ) from exc

    def run_python(self, code: str, *, stdin: Optional[str] = None) -> ExecResult:
        """Execute *code* inside the E2B cloud micro-VM.

        ``stdin`` is not forwarded — E2B's code-run API does not support
        interactive stdin. Use subprocess or Docker if stdin is required.

        Returns
        -------
        ExecResult
            Always returned, never raised.
        """
        stdout_data = ""
        stderr_data = ""
        exit_code = -1
        timed_out = False
        start = time.monotonic()

        try:
            # The E2B SDK's execute method returns a structured result.
            result = self._sandbox.run_code(code, timeout=int(self.timeout_s))

            # Map E2B result fields to ExecResult.
            # The SDK returns an object with .stdout, .stderr, .error, .results.
            stdout_parts: list[str] = []
            stderr_parts: list[str] = []

            # Collect text output from execution results.
            if hasattr(result, "results") and result.results:
                for r in result.results:
                    if hasattr(r, "text") and r.text:
                        stdout_parts.append(r.text)

            # Explicit stdout/stderr logs if the SDK provides them.
            if hasattr(result, "logs"):
                logs = result.logs
                if hasattr(logs, "stdout") and logs.stdout:
                    stdout_parts.extend(logs.stdout)
                if hasattr(logs, "stderr") and logs.stderr:
                    stderr_parts.extend(logs.stderr)

            # Error field signals an execution exception.
            if hasattr(result, "error") and result.error:
                err = result.error
                err_text = (
                    f"{getattr(err, 'name', 'Error')}: {getattr(err, 'value', str(err))}"
                )
                stderr_parts.append(err_text)
                exit_code = 1
            else:
                exit_code = 0

            stdout_data = "\n".join(stdout_parts)
            stderr_data = "\n".join(stderr_parts)

        except TimeoutError:
            timed_out = True
            exit_code = -1
        except Exception as exc:
            # E2B timeout exceptions may have different types depending on SDK version.
            exc_name = type(exc).__name__.lower()
            if "timeout" in exc_name:
                timed_out = True
                exit_code = -1
            else:
                stderr_data = f"[E2BSandbox error] {exc}"
                exit_code = -1

        duration_ms = int((time.monotonic() - start) * 1000)

        return ExecResult(
            stdout=stdout_data,
            stderr=stderr_data,
            exit_code=exit_code,
            timed_out=timed_out,
            duration_ms=duration_ms,
        )

    def close(self) -> None:
        """Close the E2B sandbox session and release remote resources."""
        try:
            self._sandbox.close()
        except Exception:
            pass
