# V2 Scoping — DAG Pipelines & Parallel Fan-out / Fan-in

> Status: **design / not built.** v1 ships sequential handoffs + a terminal agent.
> This document scopes the V2 concurrency work and shows exactly which v1
> structures were built DAG-ready so V2 is an extension, not a rewrite.

## 1. Goal

Let a pipeline express **parallelism**: independent agents run concurrently, one
agent fans out to N workers, and a join agent merges their results — while every
v1 guarantee (hard budget cap, sandbox, permissions, loop safety, full trace)
still holds across branches.

Target pipeline shape:

```yaml
agents:
  planner:
    role: Break the topic into 3 subtopics and emit them as a JSON list.
    tools: [save_note]
    fan_out:                      # NEW: spawn one worker per item
      to: researcher
      over: subtopics             # a list the planner produced
  researcher:                     # runs as N concurrent instances
    role: Research one subtopic thoroughly.
    tools: [web_search, read_url, save_note]
    depends_on: [planner]         # already a field in v1 config (reserved)
  synthesizer:
    role: Merge all researcher notes into one report.
    tools: [read_note, write_file]
    depends_on: [researcher]      # join: waits for all researcher instances
    terminal: true
start: planner
```

## 2. What v1 already gives us (the DAG-ready seams)

| v1 structure | Where | Why it matters for V2 |
|---|---|---|
| `AgentConfig.depends_on: list[str]` | `config.py` | The DAG edge already exists and is validated; V2 makes the scheduler honor it. |
| Per-agent context packaging via `incoming_context` | `orchestrator.py` / `agent.py` | Each agent already gets an isolated prompt + handoff package — no shared mutable conversation, so instances are independent by construction. |
| Orchestrator-level `BudgetGuard` | `guards.py` | The cap already lives above the agent; it naturally governs N parallel branches if made thread-safe. |
| Trace is a **list** of `AgentTrace` branches | `trace.py` | The schema already holds multiple agent records; concurrent branches just append (needs a lock + an `instance`/`branch_id` field). |
| Structured `emit(event)` stream | `orchestrator.py` / `agent.py` | The web UI already consumes per-agent events; add a `branch_id` and the dashboard renders parallel lanes with no protocol change. |
| `Workspace` + `save_note`/`read_note` | `tools/files.py` | A per-run shared store already exists; V2 makes it concurrency-safe (file locks / a small KV with a mutex). |
| `RepeatTracker`, per-agent iteration caps | `guards.py` / `agent.py` | Already per-agent instance state — correct under concurrency without change. |

## 3. New work

### 3.1 Config (`config.py`)
- Add `fan_out: {to: str, over: str}` to `AgentConfig` (optional). `over` names a
  list the agent emits (via a `return_list` convention or a structured final).
- Keep `handoff_to` (sequential) and `depends_on` (DAG) as alternative wiring; a
  pipeline uses one model. Validate: no cycles in `depends_on` (topological sort
  must succeed), `fan_out.to` exists, a fan-out target must `depends_on` its source.
- A pipeline is "DAG mode" if any agent has `depends_on`/`fan_out`; else v1 sequential.

### 3.2 Scheduler (`orchestrator.py` → extract `scheduler.py`)
- Replace the single `current` pointer with a **ready-set executor**:
  1. Topologically order agents from `depends_on`.
  2. An agent is *ready* when all deps are `done`. Submit ready agents to a
     `ThreadPoolExecutor(max_workers=...)`.
  3. A `fan_out` agent, on completion, expands into N `researcher` instances
     (`researcher#0…#N-1`), each a node depending on the planner; the join agent
     (`depends_on: [researcher]`) waits for **all** instances.
- Concurrency via threads (LLM calls are I/O-bound on `httpx`) — no asyncio
  rewrite needed; the sync `LLMClient` interface stays.

### 3.3 Thread-safe budget + cancellation
- Wrap `BudgetGuard.record/check` in a `threading.Lock`. On `BudgetExceeded`,
  signal a shared `cancel_event`; running branches check it at each ReAct
  iteration (alongside `on_iteration`) and stop gracefully with a partial result.
- Same pattern for pipeline wall-clock and `max_total_iterations` (now an atomic
  counter).

### 3.4 Concurrency-safe blackboard (`tools/files.py`)
- Back `save_note`/`read_note` with a process-wide lock per run (or `fcntl`
  file locks) so parallel writers don't corrupt the notes store. Reads are
  snapshot-consistent. Keys namespaced by branch to avoid clobbering
  (`researcher#2/finding`), with a shared read view for the join agent.

### 3.5 Trace + events
- Add `instance` / `branch_id` and `parent` to `AgentTrace` and every emitted
  event. Trace stays a flat list (analysis groups by `branch_id`); the UI renders
  concurrent lanes. Cost rollup already sums the list — unchanged.

## 4. Loop / safety semantics under a DAG
- **Deadlock guard:** if the ready-set is empty but not all agents are done →
  unsatisfiable deps (cycle or missing producer) → abort with a clear trace.
- **Fan-out cap:** `max_fan_out` (default e.g. 8) bounds N so a malformed list
  can't spawn thousands of branches / blow the budget.
- **Per-branch caps** unchanged; **global** caps become atomic. Every stop stays
  non-fatal → partial result + trace.

## 5. Phased build plan
1. ✅ **DAG (no parallelism yet) — DONE:** topological scheduler (`_topological_order`,
   `_run_dag` in `orchestrator.py`) runs agents in dependency order; the join agent
   receives all `depends_on` outputs; `depends_on`/`branch_id` land in the trace;
   cycles are rejected at config load + by a runtime deadlock guard. Pure refactor —
   all v1 sequential pipelines/tests unchanged (`run_pipeline` dispatches on
   `dag_mode`). Covered by `tests/test_dag.py`; example `examples/research-dag.yaml`.
2. ✅ **Parallel execution — DONE:** `_run_dag_parallel` (`orchestrator.py`) runs the DAG
   on a `ThreadPoolExecutor` (driver-thread ready-set scheduler) when
   `budget.max_parallel > 1` (default 4); `max_parallel <= 1` keeps the deterministic
   sequential path. Thread-safety added where state is shared: `BudgetGuard` lock,
   `TraceRecorder.add_agent` lock, notes lock (`tools/files.py`), atomic iteration guard.
   Cooperative cancellation via a `cancel_event` checked in `pipeline_iter_guard` — the
   first real stop (budget/iteration/wall-clock) winds the others down gracefully.
   Covered by `tests/test_parallel.py` (incl. a `threading.Barrier` proof of real
   concurrency) + a thread-safe agent-routed `RoutingFakeLLMClient`.
3. ✅ **Fan-out — DONE:** `AgentConfig.fan_out: {to, max}` (`config.py`). A fan-out agent's
   output is parsed into a list (`_parse_list`: JSON array or bullet/numbered lines, capped at
   `max`); the scheduler spawns one instance of the `to` template per item (`to#0..to#k-1`),
   runs them on the pool, and the join (`depends_on: [to]`) waits for the whole group via
   group-accounting in `_run_dag_parallel`. The template must `depends_on` its source (validated
   at load). Covered by `tests/test_fanout.py`; example `examples/fanout-research.yaml`. (Design
   note: the list comes from the agent's *output*, not a named note — simpler than the original
   `over:` sketch and needs no extra config.)
4. **UI:** parallel lanes in the dashboard (the event schema already supports it).

## 6. Risks / open questions
- **Determinism in tests:** concurrent ordering is non-deterministic — assert on
  *sets* of events and final state, not order. The `FakeLLMClient` already makes
  each branch deterministic in isolation.
- **Budget fairness:** with N branches in flight, the cap can be hit by aggregate
  spend mid-call; the per-call pre-check + cancellation bounds overshoot to
  ~one in-flight call per branch — document this, same trade-off as v1.
- **Provider rate limits:** parallel calls hit 429s harder; `tenacity` backoff
  already handles it, but `max_workers` should default conservative (e.g. 4).
- **Sandbox concurrency:** subprocess/Docker backends must run per-branch temp
  dirs (already per-run; extend to per-branch) to avoid file collisions.

## 7. Out of scope for V2
Distributed execution across machines, a persistent run queue, and resumable
runs (listed separately in the roadmap) — V2 stays single-process, thread-pooled.
