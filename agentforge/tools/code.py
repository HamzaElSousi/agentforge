"""Code execution tool: ``run_python`` (opt-in).

``run_python`` is classified ``dangerous`` and is **disabled by default**. It
only becomes available when the pipeline explicitly sets
``sandbox.allow_run_python: true`` in its YAML configuration. Even then it is
always routed through the configured sandbox backend (subprocess / Docker /
E2B) — it never runs code directly in the AgentForge process.

The opt-in gate and sandbox selection live in the orchestrator (which is
responsible for constructing the :class:`~agentforge.tools.registry.ToolContext`
with the right ``sandbox`` value). This module only:

1. Implements the ``@tool``-decorated function.
2. Guards against a ``None`` sandbox (disabled or misconfigured).
3. Formats and truncates the :class:`~agentforge.sandbox.base.ExecResult`
   for safe inclusion in the agent's history.
"""

from __future__ import annotations

from agentforge.sandbox.base import ExecResult
from agentforge.tools.registry import ToolContext, tool

_MAX_OUTPUT_CHARS = 6_000


def _truncate(s: str, limit: int = _MAX_OUTPUT_CHARS) -> str:
    """Truncate *s* to *limit* characters with an omission marker."""
    if len(s) <= limit:
        return s
    omitted = len(s) - limit
    return s[:limit] + f" [... {omitted} chars omitted ...]"


@tool(risk="dangerous", needs_network=False)
def run_python(ctx: ToolContext, code: str) -> str:
    """Execute a Python code snippet inside the configured sandbox.

    This tool is **opt-in** — it is only available when the pipeline
    configuration sets ``sandbox.allow_run_python: true``. Even when enabled,
    the code runs inside the sandbox backend (subprocess, Docker, or E2B) with
    workspace jailing, resource limits, and (optionally) network isolation.

    The result (stdout, stderr, exit code, timeout flag) is captured and
    returned as a short formatted summary — never raised as an exception.

    Parameters
    ----------
    code:
        The Python source code to execute.

    Returns
    -------
    str
        The execution result summary: stdout/stderr and exit status if the
        code ran, a timeout message if the wall-clock limit was hit, or a
        disabled message if no sandbox is configured.
    """
    if ctx.sandbox is None:
        return (
            "run_python is disabled: no sandbox is configured for this run, "
            "or allow_run_python is false in the pipeline config. "
            "Set 'sandbox.allow_run_python: true' and configure a backend to enable it."
        )

    result: ExecResult = ctx.sandbox.run_python(code)
    return _truncate(result.summary())
