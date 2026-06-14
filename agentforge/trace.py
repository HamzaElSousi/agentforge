"""Execution trace: a complete, human-readable record of a run written to
``trace.json`` — every tool call, every permission decision, token counts, and
exactly where the money went, so a run is fully debuggable without re-running.

Secrets are redacted on write: any registered secret value (an API key read
from the environment) is replaced with ``***`` anywhere it appears.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class ToolCallRecord:
    """One tool invocation and its outcome, as it appears in the trace."""

    tool: str
    args: dict[str, Any]
    result: str  # already truncated for context
    outcome: str = "approved"  # approved | denied | edited (permission outcome)
    auto: bool = True  # resolved without a human?
    reason: str = ""
    edited_args: Optional[dict[str, Any]] = None
    error: Optional[str] = None


@dataclass
class AgentTrace:
    """Per-agent rollup."""

    name: str
    model: str
    branch_id: Optional[str] = None  # DAG/parallel grouping (V2); defaults to the agent name
    depends_on: list[str] = field(default_factory=list)  # DAG edges this agent waited on
    iterations: int = 0
    cost_usd: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    handoff_context: Optional[str] = None
    stopped_reason: Optional[str] = None


@dataclass
class Trace:
    """The whole-run trace, mirroring the schema in the PRD."""

    pipeline: str
    goal: str
    duration_ms: int = 0
    cost: dict[str, Any] = field(default_factory=dict)
    stopped_reason: str = "completed"
    agents: list[AgentTrace] = field(default_factory=list)
    final_output: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "pipeline": self.pipeline,
            "goal": self.goal,
            "duration_ms": self.duration_ms,
            "cost": self.cost,
            "stopped_reason": self.stopped_reason,
            "agents": [asdict(a) for a in self.agents],
            "final_output": self.final_output,
        }


def _redact(obj: Any, secrets: list[str]) -> Any:
    """Recursively replace any secret substring with ``***``."""
    if isinstance(obj, str):
        out = obj
        for s in secrets:
            if s:
                out = out.replace(s, "***")
        return out
    if isinstance(obj, dict):
        return {k: _redact(v, secrets) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_redact(v, secrets) for v in obj]
    return obj


class TraceRecorder:
    """Accumulates trace data during a run and writes ``trace.json``.

    The orchestrator pushes a new :class:`AgentTrace` as each agent starts and
    mutates it through the agent's ReAct loop; cost/token rollups are computed
    at write time.
    """

    def __init__(self, pipeline: str, goal: str, *, secrets: Optional[list[str]] = None) -> None:
        self.trace = Trace(pipeline=pipeline, goal=goal)
        self._secrets = [s for s in (secrets or []) if s]
        self._t0 = time.monotonic()

    def add_agent(self, name: str, model: str) -> AgentTrace:
        at = AgentTrace(name=name, model=model)
        self.trace.agents.append(at)
        return at

    def finalize(self, *, stopped_reason: str, final_output: str) -> Trace:
        self.trace.stopped_reason = stopped_reason
        self.trace.final_output = final_output
        self.trace.duration_ms = int((time.monotonic() - self._t0) * 1000)
        self.trace.cost = {
            "total_usd": round(sum(a.cost_usd for a in self.trace.agents), 6),
            "prompt_tokens": sum(a.prompt_tokens for a in self.trace.agents),
            "completion_tokens": sum(a.completion_tokens for a in self.trace.agents),
        }
        return self.trace

    def write(self, path: str | Path) -> Path:
        data = _redact(self.trace.to_dict(), self._secrets)
        p = Path(path)
        p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return p

    def as_json(self) -> str:
        return json.dumps(_redact(self.trace.to_dict(), self._secrets), indent=2, ensure_ascii=False)
