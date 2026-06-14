"""The orchestrator: load a pipeline, arm the guards, run agents in sequence,
enforce the budget, and write the trace.

v1 is sequential: start at ``start``, follow each agent's ``handoff_to`` until a
terminal agent concludes or a guard trips. Every stop — budget, iteration cap,
handoff cycle, repeated action, wall-clock — is **non-fatal**: it logs a reason,
finalizes a partial result, and writes ``trace.json``. The data structures
(per-agent context packaging, orchestrator-level budget) are deliberately
DAG-shaped so V2 can schedule independent agents in parallel without a rewrite.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

from rich.console import Console

from agentforge.agent import AgentResult, AgentRunner
from agentforge.config import Provider, load_pipeline
from agentforge.cost import PricingCatalog, fetch_openrouter_models
from agentforge.guards import BudgetExceeded, BudgetGuard, StopReason
from agentforge.permissions import PermissionManager
from agentforge.tools.registry import REGISTRY, ToolContext


class PipelineStop(Exception):
    """Raised to stop the whole pipeline for a non-budget reason (iteration cap,
    handoff cycle, wall-clock). Carries the canonical stop reason."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


def _import_builtin_tools() -> None:
    """Importing these modules registers the built-in tools on REGISTRY."""
    import agentforge.tools.files  # noqa: F401
    import agentforge.tools.web  # noqa: F401
    import agentforge.tools.code  # noqa: F401


def _build_client(provider: Provider, model: str, *, api_key: Optional[str], base_url, timeout):
    if provider == Provider.openrouter:
        from agentforge.llm.openrouter import OpenRouterClient
        return OpenRouterClient(model, api_key=api_key, base_url=base_url, timeout=timeout)
    if provider == Provider.anthropic:
        from agentforge.llm.anthropic import AnthropicClient
        return AnthropicClient(model, api_key=api_key, base_url=base_url, timeout=timeout)
    if provider == Provider.openai:
        from agentforge.llm.openai import OpenAIClient
        return OpenAIClient(model, api_key=api_key, base_url=base_url, timeout=timeout)
    if provider == Provider.ollama:
        from agentforge.llm.ollama import OllamaClient
        return OllamaClient(model, base_url=base_url, timeout=timeout)
    raise ValueError(f"unsupported provider: {provider}")


def _build_catalog(provider: Provider, api_key: Optional[str], console: Console) -> PricingCatalog:
    """Live OpenRouter pricing when possible; always merged over the fallback.

    ``from_openrouter`` seeds with the built-in FALLBACK table and ``.get()``
    falls back to it too, so unknown slugs still cost-account correctly.
    """
    if provider == Provider.openrouter:
        try:
            return PricingCatalog.from_openrouter(fetch_openrouter_models(api_key))
        except Exception:
            console.print("[yellow]⚠ couldn't fetch live pricing; using built-in fallback table[/yellow]")
    return PricingCatalog.from_openrouter([])  # seeds with FALLBACK_PRICING


def run_pipeline(
    *,
    pipeline_path: Path,
    goal: str,
    trace_path: Optional[Path],
    estimate_only: bool = False,
    assume_yes: bool = False,
    console: Optional[Console] = None,
    client=None,
    catalog: Optional[PricingCatalog] = None,
) -> dict:
    """Execute a pipeline against ``goal``. Returns the trace dict.

    ``client`` and ``catalog`` are injection seams: pass a scripted client (and
    a pricing catalog) to drive the pipeline deterministically in tests or to
    embed AgentForge without its default provider wiring. When ``client`` is
    given, the API-key requirement and live pricing fetch are skipped.
    """
    console = console or Console()
    cfg = load_pipeline(pipeline_path)
    _import_builtin_tools()

    # Resolve API key (Ollama needs none; --estimate never spends, so it can run keyless).
    api_key = os.environ.get(cfg.llm.api_key_env)
    if catalog is None:
        catalog = (
            PricingCatalog.from_openrouter([]) if client is not None
            else _build_catalog(cfg.llm.provider, api_key, console)
        )

    if estimate_only:
        return _estimate(cfg, goal, catalog, console)

    if client is None and cfg.llm.provider != Provider.ollama and not api_key:
        console.print(
            f"[bold red]✗[/bold red] missing API key: set [bold]{cfg.llm.api_key_env}[/bold] "
            f"in your environment / .env"
        )
        raise SystemExit(2)

    # --- live run setup --------------------------------------------------- #
    from agentforge.trace import TraceRecorder

    recorder = TraceRecorder(cfg.name, goal, secrets=[api_key] if api_key else [])
    budget = BudgetGuard(cap_usd=cfg.budget.max_usd_per_run, catalog=catalog,
                         assumed_completion_tokens=cfg.llm.max_tokens or 1024)

    # Per-run workspace (gitignored).
    workdir = Path("runs") / f"{cfg.name}-{int(time.time())}"
    workdir.mkdir(parents=True, exist_ok=True)
    from agentforge.tools.files import Workspace

    sandbox = None
    if cfg.sandbox.allow_run_python:
        from agentforge.sandbox import make_sandbox
        sandbox = make_sandbox(cfg.sandbox, str(workdir))
    ctx = ToolContext(workspace=Workspace(workdir), sandbox=sandbox, network=cfg.sandbox.network)

    import sys
    interactive = sys.stdin.isatty() and not assume_yes
    permissions = PermissionManager(
        cfg.permissions, interactive=interactive, console=console, assume_yes=assume_yes
    )

    if client is None:
        client = _build_client(
            cfg.llm.provider, cfg.llm.model, api_key=api_key,
            base_url=cfg.llm.base_url, timeout=cfg.llm.timeout_s,
        )

    # Pipeline-level guards.
    t0 = time.monotonic()
    state = {"total_iters": 0}

    def pipeline_iter_guard() -> None:
        state["total_iters"] += 1
        if state["total_iters"] > cfg.budget.max_total_iterations:
            raise PipelineStop(StopReason.MAX_TOTAL_ITERATIONS)
        if cfg.budget.wall_clock_s and (time.monotonic() - t0) > cfg.budget.wall_clock_s:
            raise PipelineStop(StopReason.WALL_CLOCK)

    # --- sequential handoff loop ------------------------------------------ #
    current = cfg.start
    incoming: Optional[str] = None
    final_output = ""
    stopped_reason = StopReason.COMPLETED
    visits: dict[str, int] = {}
    handoff_cap = max(3, 2 * len(cfg.agents))

    console.print(f"[bold cyan]▶ running pipeline[/bold cyan] '{cfg.name}' — goal: {goal}")

    while True:
        visits[current] = visits.get(current, 0) + 1
        if visits[current] > handoff_cap:
            stopped_reason = StopReason.HANDOFF_CYCLE
            break

        agent_cfg = cfg.agents[current]
        model = agent_cfg.model or cfg.llm.model
        at = recorder.add_agent(current, model)

        try:
            tools = REGISTRY.subset(agent_cfg.tools)
        except KeyError as e:
            console.print(f"[bold red]✗ {e}[/bold red]")
            raise SystemExit(2)

        console.print(f"  [bold]{current}[/bold] ({model}) — tools: {agent_cfg.tools or 'none'}")
        runner = AgentRunner(
            name=current,
            role=agent_cfg.role,
            client=client,
            model=model,
            tools=tools,
            ctx=ctx,
            permissions=permissions,
            budget=budget,
            agent_trace=at,
            console=console,
            terminal=agent_cfg.terminal,
            handoff_to=agent_cfg.handoff_to,
            max_iterations=agent_cfg.max_iterations,
            temperature=cfg.llm.temperature,
            max_tokens=cfg.llm.max_tokens,
            context_token_cap=None,
            on_iteration=pipeline_iter_guard,
        )

        try:
            result: AgentResult = runner.run(goal, incoming)
        except BudgetExceeded as e:
            console.print(f"  [yellow]⛔ budget cap hit (${e.cap}); returning partial result[/yellow]")
            stopped_reason = StopReason.BUDGET_EXCEEDED
            final_output = at.handoff_context or final_output
            break
        except PipelineStop as e:
            console.print(f"  [yellow]⛔ {e.reason}; returning partial result[/yellow]")
            stopped_reason = e.reason
            break

        # An agent that concludes with empty text (a weak model dropping its
        # task) must never clobber substantive earlier output — fall back to the
        # best partial we have so the run returns useful content, not "(none)".
        if result.kind == "final":
            final_output = result.output or final_output
            stopped_reason = at.stopped_reason or StopReason.COMPLETED
            break
        if result.kind == "stopped":
            final_output = result.output or final_output
            stopped_reason = result.stopped_reason or StopReason.COMPLETED
            break
        # handoff — carry the last non-empty context/partial forward
        incoming = result.output or incoming
        final_output = result.output or final_output
        if not result.handoff_to:
            break
        current = result.handoff_to

    trace = recorder.finalize(stopped_reason=stopped_reason, final_output=final_output)
    if trace_path is not None:
        recorder.write(trace_path)

    _print_summary(console, recorder, trace_path)
    return trace.to_dict()


def _estimate(cfg, goal: str, catalog: PricingCatalog, console: Console) -> dict:
    """Project an upper-ish bound on run cost without calling any model."""
    from agentforge.context import count_message_tokens
    from agentforge.cost import ModelPricing, cost_usd
    from agentforge.messages import Message, Usage

    console.print(f"[bold]Estimate[/bold] for pipeline '{cfg.name}' (no model calls made):")
    total = 0.0
    # Walk the sequential chain once (no cycles) for a representative estimate.
    seen = set()
    current = cfg.start
    rows = []
    while current and current not in seen:
        seen.add(current)
        a = cfg.agents[current]
        model = a.model or cfg.llm.model
        pricing = catalog.get(model) or ModelPricing(0.0, 0.0)
        sys_user = [
            Message(role="system", content=a.role),
            Message(role="user", content=goal),
        ]
        ptoks = count_message_tokens(sys_user, model)
        comp = cfg.llm.max_tokens or 512
        # assume up to max_iterations turns, prompt growing modestly each turn
        agent_cost = sum(
            cost_usd(Usage(prompt_tokens=ptoks * i, completion_tokens=comp), pricing)
            for i in range(1, a.max_iterations + 1)
        )
        total += agent_cost
        rows.append((current, model, a.max_iterations, agent_cost))
        current = a.handoff_to

    for name, model, iters, c in rows:
        console.print(f"  {name:<14} {model:<32} ≤{iters} iters  ~${c:.4f}")
    console.print(f"[bold]Projected upper bound:[/bold] ~${total:.4f}  "
                  f"(cap is ${cfg.budget.max_usd_per_run})")
    if total > cfg.budget.max_usd_per_run:
        console.print("[yellow]⚠ projection exceeds the cap — the run may abort early with a partial result.[/yellow]")
    return {"projected_usd": round(total, 6), "cap_usd": cfg.budget.max_usd_per_run}


def _print_summary(console: Console, recorder, trace_path) -> None:
    t = recorder.trace
    console.print()
    console.print(f"[bold green]✓ done[/bold green] — stopped_reason: [bold]{t.stopped_reason}[/bold]")
    console.print(
        f"  cost: [bold]${t.cost.get('total_usd', 0):.6f}[/bold] "
        f"({t.cost.get('prompt_tokens', 0)} prompt + {t.cost.get('completion_tokens', 0)} completion tokens)"
    )
    console.print(f"  duration: {t.duration_ms} ms")
    if trace_path is not None:
        console.print(f"  trace: [bold]{trace_path}[/bold]")
    console.print()
    console.rule("final output")
    console.print(t.final_output or "[dim](no output)[/dim]")
