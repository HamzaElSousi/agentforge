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
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
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
    emit=None,
) -> dict:
    """Execute a pipeline against ``goal``. Returns the trace dict.

    ``client`` and ``catalog`` are injection seams: pass a scripted client (and
    a pricing catalog) to drive the pipeline deterministically in tests or to
    embed AgentForge without its default provider wiring. When ``client`` is
    given, the API-key requirement and live pricing fetch are skipped.

    ``emit`` is an optional callback receiving structured run events (dicts) —
    used by the web UI to stream a run live. It is a no-op by default.
    """
    console = console or Console()
    emit = emit or (lambda event: None)
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

    # Pipeline-level guards. ``cancel_event`` lets one branch's stop (budget,
    # iteration cap, wall-clock) signal the others to wind down (parallel DAG).
    t0 = time.monotonic()
    state = {"total_iters": 0}
    iter_lock = threading.Lock()
    cancel_event = threading.Event()

    def pipeline_iter_guard() -> None:
        # Checked at the top of every ReAct iteration (via AgentRunner.on_iteration),
        # so it is also the cooperative-cancellation point for in-flight branches.
        if cancel_event.is_set():
            raise PipelineStop(StopReason.CANCELLED)
        with iter_lock:
            state["total_iters"] += 1
            n = state["total_iters"]
        if n > cfg.budget.max_total_iterations:
            raise PipelineStop(StopReason.MAX_TOTAL_ITERATIONS)
        if cfg.budget.wall_clock_s and (time.monotonic() - t0) > cfg.budget.wall_clock_s:
            raise PipelineStop(StopReason.WALL_CLOCK)

    # --- per-agent execution (shared by the sequential + DAG paths) ------- #
    def run_agent(name: str, incoming: Optional[str]):
        agent_cfg = cfg.agents[name]
        model = agent_cfg.model or cfg.llm.model
        at = recorder.add_agent(name, model)
        at.branch_id = name
        at.depends_on = list(agent_cfg.depends_on)
        try:
            tools = REGISTRY.subset(agent_cfg.tools)
        except KeyError as e:
            console.print(f"[bold red]✗ {e}[/bold red]")
            raise SystemExit(2)

        console.print(f"  [bold]{name}[/bold] ({model}) — tools: {agent_cfg.tools or 'none'}")
        emit({"type": "agent_start", "agent": name, "branch_id": name,
              "model": model, "tools": agent_cfg.tools})
        runner = AgentRunner(
            name=name, role=agent_cfg.role, client=client, model=model, tools=tools,
            ctx=ctx, permissions=permissions, budget=budget, agent_trace=at, console=console,
            terminal=agent_cfg.terminal, handoff_to=agent_cfg.handoff_to,
            max_iterations=agent_cfg.max_iterations, temperature=cfg.llm.temperature,
            max_tokens=cfg.llm.max_tokens, context_token_cap=None,
            on_iteration=pipeline_iter_guard, emit=emit,
        )
        result = runner.run(goal, incoming)
        emit({"type": "agent_end", "agent": name, "branch_id": name, "kind": result.kind,
              "output": (result.output or "")[:1000], "handoff_to": result.handoff_to,
              "stopped_reason": result.stopped_reason, "cost_usd": round(at.cost_usd, 6)})
        return result, at

    dag_mode = any(a.depends_on for a in cfg.agents.values())
    console.print(f"[bold cyan]▶ running pipeline[/bold cyan] '{cfg.name}'"
                  f"{' [dim](DAG)[/dim]' if dag_mode else ''} — goal: {goal}")
    emit({"type": "run_start", "pipeline": cfg.name, "goal": goal,
          "agents": list(cfg.agents), "start": cfg.start,
          "mode": "dag" if dag_mode else "sequential"})

    if dag_mode and cfg.budget.max_parallel > 1:
        stopped_reason, final_output = _run_dag_parallel(
            cfg, run_agent, console, cfg.budget.max_parallel, cancel_event
        )
    elif dag_mode:
        stopped_reason, final_output = _run_dag(cfg, run_agent, console)
    else:
        stopped_reason, final_output = _run_sequential(cfg, run_agent, console)

    trace = recorder.finalize(stopped_reason=stopped_reason, final_output=final_output)
    if trace_path is not None:
        recorder.write(trace_path)

    _print_summary(console, recorder, trace_path)
    emit({"type": "run_end", "stopped_reason": trace.stopped_reason,
          "cost": trace.cost, "final_output": trace.final_output,
          "trace": trace.to_dict()})
    return trace.to_dict()


def _run_sequential(cfg, run_agent, console: Console) -> tuple[str, str]:
    """v1 path: follow ``start`` → ``handoff_to`` until a terminal/chain-end.

    Returns ``(stopped_reason, final_output)``. Every stop is non-fatal.
    """
    current = cfg.start
    incoming: Optional[str] = None
    final_output = ""
    visits: dict[str, int] = {}
    handoff_cap = max(3, 2 * len(cfg.agents))

    while True:
        visits[current] = visits.get(current, 0) + 1
        if visits[current] > handoff_cap:
            return StopReason.HANDOFF_CYCLE, final_output
        try:
            result, at = run_agent(current, incoming)
        except BudgetExceeded as e:
            console.print(f"  [yellow]⛔ budget cap hit (${e.cap}); returning partial result[/yellow]")
            return StopReason.BUDGET_EXCEEDED, final_output
        except PipelineStop as e:
            console.print(f"  [yellow]⛔ {e.reason}; returning partial result[/yellow]")
            return e.reason, final_output

        # An empty conclusion (a weak model dropping its task) must never clobber
        # substantive earlier output — fall back to the best partial.
        if result.kind == "final":
            return (at.stopped_reason or StopReason.COMPLETED), (result.output or final_output)
        if result.kind == "stopped":
            return (result.stopped_reason or StopReason.COMPLETED), (result.output or final_output)
        incoming = result.output or incoming
        final_output = result.output or final_output
        if not result.handoff_to:
            return StopReason.COMPLETED, final_output
        current = result.handoff_to


def _topological_order(cfg) -> list[str]:
    """Kahn's algorithm over ``depends_on``. Deterministic (``start`` first, then
    config order). Raises :class:`PipelineStop` on a cycle — the deadlock guard."""
    names = list(cfg.agents)
    idx = {n: i for i, n in enumerate(names)}
    indeg = {n: len(cfg.agents[n].depends_on) for n in names}
    adj: dict[str, list[str]] = {n: [] for n in names}
    for n in names:
        for d in cfg.agents[n].depends_on:
            adj[d].append(n)

    def key(n: str):
        return (0 if n == cfg.start else 1, idx[n])

    ready = sorted([n for n in names if indeg[n] == 0], key=key)
    order: list[str] = []
    while ready:
        node = ready.pop(0)
        order.append(node)
        newly = []
        for m in adj[node]:
            indeg[m] -= 1
            if indeg[m] == 0:
                newly.append(m)
        if newly:
            ready = sorted(ready + newly, key=key)
    if len(order) != len(names):
        raise PipelineStop(StopReason.HANDOFF_CYCLE)
    return order


def _package_deps(deps: list[str], outputs: dict[str, str]) -> Optional[str]:
    """Bundle each dependency's output into the next agent's incoming context."""
    parts = [f"--- Context from '{d}' ---\n{outputs[d]}" for d in deps if outputs.get(d)]
    return "\n\n".join(parts) if parts else None


def _run_dag(cfg, run_agent, console: Console) -> tuple[str, str]:
    """V2 path (sequential execution of a DAG): run agents in topological order,
    feeding each its dependencies' outputs. The terminal (join) agent's output is
    the final result. Parallel execution of independent branches lands in V2
    Phase 2 — this proves ``depends_on``, join semantics, and the deadlock guard.
    """
    order = _topological_order(cfg)
    terminal_name = next((n for n, a in cfg.agents.items() if a.terminal), order[-1])
    outputs: dict[str, str] = {}
    final_output = ""

    for name in order:
        incoming = _package_deps(cfg.agents[name].depends_on, outputs)
        try:
            result, _ = run_agent(name, incoming)
        except BudgetExceeded as e:
            console.print(f"  [yellow]⛔ budget cap hit (${e.cap}); returning partial result[/yellow]")
            return StopReason.BUDGET_EXCEEDED, (outputs.get(terminal_name) or final_output)
        except PipelineStop as e:
            console.print(f"  [yellow]⛔ {e.reason}; returning partial result[/yellow]")
            return e.reason, (outputs.get(terminal_name) or final_output)
        outputs[name] = result.output or ""
        if result.output:
            final_output = result.output

    return StopReason.COMPLETED, (outputs.get(terminal_name) or final_output)


def _run_dag_parallel(cfg, run_agent, console: Console, max_workers: int,
                      cancel_event: threading.Event) -> tuple[str, str]:
    """V2 Phase 2: run the DAG on a thread pool so independent branches execute
    concurrently. A ready-set scheduler submits in-degree-0 agents; as each
    finishes, its dependents' in-degrees drop and any newly-ready agent is
    submitted. The join (terminal) agent runs only once all its deps complete.

    Scheduling state (``outputs``, ``indeg``, ``in_flight``) is touched only on
    this driver thread — workers just run ``run_agent`` — so the cross-thread
    shared state is limited to the already-locked budget/trace/notes/iteration
    guard. On the first real stop (budget/iteration/wall-clock) the driver sets
    ``cancel_event`` and stops submitting; in-flight branches wind down at their
    next iteration. Every stop is non-fatal → partial result + trace.
    """
    names = list(cfg.agents)
    indeg = {n: len(cfg.agents[n].depends_on) for n in names}
    adj: dict[str, list[str]] = {n: [] for n in names}
    for n in names:
        for d in cfg.agents[n].depends_on:
            adj[d].append(n)
    terminal_name = next((n for n, a in cfg.agents.items() if a.terminal), None)

    outputs: dict[str, str] = {}
    final_output = ""
    first_stop: Optional[str] = None

    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="agentforge") as ex:
        in_flight: dict = {}  # Future -> agent name

        def submit(name: str) -> None:
            incoming = _package_deps(cfg.agents[name].depends_on, outputs)
            in_flight[ex.submit(run_agent, name, incoming)] = name

        for n in [n for n in names if indeg[n] == 0]:
            submit(n)

        while in_flight:
            done, _ = wait(list(in_flight), return_when=FIRST_COMPLETED)
            newly_ready: list[str] = []
            for fut in done:
                name = in_flight.pop(fut)
                try:
                    result, _at = fut.result()
                except BudgetExceeded as e:
                    if first_stop is None:
                        first_stop = StopReason.BUDGET_EXCEEDED
                        console.print(f"  [yellow]⛔ budget cap hit (${e.cap}); cancelling remaining branches[/yellow]")
                        cancel_event.set()
                    continue
                except PipelineStop as e:
                    if e.reason != StopReason.CANCELLED and first_stop is None:
                        first_stop = e.reason
                        console.print(f"  [yellow]⛔ {e.reason}; cancelling remaining branches[/yellow]")
                        cancel_event.set()
                    continue
                outputs[name] = result.output or ""
                if result.output:
                    final_output = result.output
                for m in adj[name]:
                    indeg[m] -= 1
                    if indeg[m] == 0:
                        newly_ready.append(m)
            if first_stop is None:
                for m in newly_ready:
                    submit(m)
            # if cancelled, stop submitting; let in-flight branches drain

    reason = first_stop or StopReason.COMPLETED
    if terminal_name and outputs.get(terminal_name):
        final_output = outputs[terminal_name]
    return reason, final_output


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
