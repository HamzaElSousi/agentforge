"""Pipeline config: YAML → validated Pydantic v2 models.

The pipeline is *data, not code*. Validation happens at load time so malformed
configs fail before any money is spent, with a clear message pointing at the
offending field. The model is deliberately DAG-ready: ``handoff_to`` is the v1
sequential edge, and a future ``depends_on`` graph slots in beside it without a
rewrite (see PRD V2 roadmap).
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator


class ConfigError(ValueError):
    """Raised for invalid pipeline configuration with a human-readable message."""


# --- Enums ------------------------------------------------------------------ #


class Provider(str, Enum):
    openrouter = "openrouter"
    anthropic = "anthropic"
    openai = "openai"
    ollama = "ollama"


class SandboxBackend(str, Enum):
    subprocess = "subprocess"
    docker = "docker"
    e2b = "e2b"


class PermissionMode(str, Enum):
    auto = "auto"
    prompt = "prompt"
    strict = "strict"


class NonInteractivePolicy(str, Enum):
    deny = "deny"
    allow_auto_approved = "allow_auto_approved"


# --- Section models --------------------------------------------------------- #


class LLMConfig(BaseModel):
    provider: Provider = Provider.openrouter
    model: str = "deepseek/deepseek-v4-flash"
    api_key_env: str = "OPENROUTER_API_KEY"
    base_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    timeout_s: float = 120.0

    model_config = {"extra": "forbid"}


class BudgetConfig(BaseModel):
    max_usd_per_run: float = Field(default=0.25, gt=0)
    max_total_iterations: int = Field(default=30, gt=0)
    max_parallel: int = Field(default=4, gt=0)  # DAG mode: max agent branches to run concurrently (1 = sequential)
    wall_clock_s: Optional[float] = Field(default=None, gt=0)

    model_config = {"extra": "forbid"}


class SandboxConfig(BaseModel):
    backend: SandboxBackend = SandboxBackend.subprocess
    network: bool = False
    timeout_s: float = Field(default=20.0, gt=0)
    cpu_seconds: int = Field(default=10, gt=0)
    memory_mb: int = Field(default=512, gt=0)
    allow_run_python: bool = False  # run_python is opt-in

    model_config = {"extra": "forbid"}


class PermissionsConfig(BaseModel):
    mode: PermissionMode = PermissionMode.prompt
    auto_approve: list[str] = Field(default_factory=list)
    require_approval: list[str] = Field(default_factory=list)
    deny: list[str] = Field(default_factory=list)
    non_interactive: NonInteractivePolicy = NonInteractivePolicy.deny

    model_config = {"extra": "forbid"}


class FanOutConfig(BaseModel):
    """V2 fan-out: this agent produces a list, and the runtime spawns one
    instance of ``to`` per item (capped at ``max``), running them in parallel.
    The ``to`` agent is a *template* — it must ``depends_on`` this agent and is
    never run on its own; a join agent (``depends_on: [to]``) merges the results.
    """

    to: str
    max: int = Field(default=8, gt=0)

    model_config = {"extra": "forbid"}


class AgentConfig(BaseModel):
    role: str
    tools: list[str] = Field(default_factory=list)
    model: Optional[str] = None  # per-agent override
    handoff_to: Optional[str] = None  # v1 sequential edge
    depends_on: list[str] = Field(default_factory=list)  # V2 DAG dependencies
    fan_out: Optional[FanOutConfig] = None  # V2 fan-out: spawn N instances of fan_out.to
    terminal: bool = False
    max_iterations: int = Field(default=10, gt=0)
    wall_clock_s: Optional[float] = Field(default=None, gt=0)

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _terminal_xor_handoff(self) -> "AgentConfig":
        if self.terminal and self.handoff_to:
            raise ValueError("an agent cannot be both 'terminal' and have 'handoff_to'")
        return self


class PipelineConfig(BaseModel):
    name: str
    start: str
    agents: dict[str, AgentConfig]
    llm: LLMConfig = Field(default_factory=LLMConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)
    permissions: PermissionsConfig = Field(default_factory=PermissionsConfig)

    model_config = {"extra": "forbid"}

    @field_validator("agents")
    @classmethod
    def _non_empty(cls, v: dict[str, AgentConfig]) -> dict[str, AgentConfig]:
        if not v:
            raise ValueError("pipeline must define at least one agent")
        return v

    @model_validator(mode="after")
    def _validate_graph(self) -> "PipelineConfig":
        names = set(self.agents)
        if self.start not in names:
            raise ValueError(f"start agent {self.start!r} is not defined in 'agents'")

        for agent_name, agent in self.agents.items():
            if agent.handoff_to and agent.handoff_to not in names:
                raise ValueError(
                    f"agent {agent_name!r} hands off to unknown agent {agent.handoff_to!r}"
                )
            for dep in agent.depends_on:
                if dep not in names:
                    raise ValueError(
                        f"agent {agent_name!r} depends_on unknown agent {dep!r}"
                    )
            if agent.fan_out:
                to = agent.fan_out.to
                if to not in names:
                    raise ValueError(
                        f"agent {agent_name!r} fans out to unknown agent {to!r}"
                    )
                if agent_name not in self.agents[to].depends_on:
                    raise ValueError(
                        f"fan-out target {to!r} must 'depends_on: [{agent_name}]' "
                        f"(it is a template spawned by {agent_name!r})"
                    )

        # DAG mode (V2): if any agent uses depends_on, the graph must be acyclic.
        dag_mode = any(a.depends_on for a in self.agents.values())
        if dag_mode:
            indeg = {n: len(a.depends_on) for n, a in self.agents.items()}
            adj: dict[str, list[str]] = {n: [] for n in self.agents}
            for n, a in self.agents.items():
                for d in a.depends_on:
                    adj[d].append(n)
            ready = [n for n, d in indeg.items() if d == 0]
            seen = 0
            while ready:
                node = ready.pop()
                seen += 1
                for m in adj[node]:
                    indeg[m] -= 1
                    if indeg[m] == 0:
                        ready.append(m)
            if seen != len(self.agents):
                raise ValueError("'depends_on' graph has a cycle — agents cannot all run")
            return self

        # v1 sequential: there must be a reachable terminal (or a dangling
        # handoff that ends the chain). Guard against an all-handoff cycle with
        # no terminal so the pipeline can actually stop.
        has_terminal = any(a.terminal for a in self.agents.values())
        has_chain_end = any(a.handoff_to is None and not a.terminal for a in self.agents.values())
        if not has_terminal and not has_chain_end:
            raise ValueError(
                "no terminal agent and every agent hands off — pipeline would never stop; "
                "mark one agent 'terminal: true'"
            )
        return self


def load_pipeline(path: str | Path) -> PipelineConfig:
    """Load and validate a pipeline YAML file into a :class:`PipelineConfig`.

    Raises :class:`ConfigError` with a readable message on any problem.
    """
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"pipeline file not found: {p}")
    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise ConfigError(f"invalid YAML in {p}: {e}") from e
    if not isinstance(raw, dict):
        raise ConfigError(f"pipeline file {p} must be a YAML mapping at the top level")
    try:
        return PipelineConfig.model_validate(raw)
    except ValidationError as e:
        raise ConfigError(_format_validation_error(p, e)) from e


def _format_validation_error(path: Path, e: ValidationError) -> str:
    lines = [f"invalid pipeline config in {path}:"]
    for err in e.errors():
        loc = ".".join(str(x) for x in err["loc"]) or "(root)"
        lines.append(f"  - {loc}: {err['msg']}")
    return "\n".join(lines)
