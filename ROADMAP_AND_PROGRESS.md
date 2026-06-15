# AgentForge — Roadmap & Progress

Living build tracker. The [PRD.md](PRD.md) is the design contract; this file tracks **what's in V1 vs V2** and the **current state of the build**. The executor updates the status tables and the log as work lands.

**Status legend:** ⬜ not started · 🟦 in progress · ✅ done · ⏸️ blocked · 🔮 V2 (deferred)

---

## Current State

- **Phase:** ✅ **V1 complete + post-V1 polish.** All 9 phases built, live-verified (paid OpenRouter **and** local Ollama, all 4 examples), public, CI-green.
- **Last updated:** 2026-06-14
- **Overall:** 9 / 9 V1 phases done. Repo public at https://github.com/HamzaElSousi/agentforge — **CI green (259 tests, Python 3.11 + 3.12).** Profile README, resume bullets, LinkedIn posts updated.
- **Post-V1 additions:** live **web dashboard** (`agentforge serve`, FastAPI+SSE) with two demo GIFs; `.env` auto-load; empty-output robustness; **PyPI-ready** (build + `twine check` pass, wheel ships static assets, fresh-venv install verified).
- **V2 in progress:** Phase 1 (DAG via `depends_on`) ✅, Phase 2 (parallel thread-pool execution) ✅, and **Phase 3 (dynamic fan-out: one agent → N parallel worker instances → join)** ✅. See [V2 design doc](docs/V2_DAG_PARALLELISM.md). Next: Phase 4 (parallel lanes in the web UI).
- **Live E2E result:** the 3-agent research pipeline (researcher → writer → reviewer) ran **end-to-end against a real local model** (`gemma4:e4b` via Ollama), `stopped_reason: completed`, 7993 prompt + 3230 completion tokens, producing a real Notion-vs-Linear competitive analysis with a full `trace.json` (saved as `examples/sample-trace.json`). The run caught + fixed 3 real bugs offline tests missed: Ollama endpoint normalization, Ollama tool-call argument serialization, and web tools wrongly gated by `network:false` + the deprecated `duckduckgo_search` package (→ `ddgs`).
- **Paid-provider note:** OpenRouter path proven via keyless live catalog/estimate + an offline-mocked request/parse test; a paid OpenRouter live run is one command away once `OPENROUTER_API_KEY` is in `.env`.
- **Tests:** ~250-test pytest suite (zero real API). **Run with** `.venv/bin/python -m pytest tests/ -v`.

---

## V1 Scope (shipping)

Sequential multi-agent pipelines defined in YAML, with provider-agnostic models, tiered sandboxing, human-in-the-loop permissions, hard cost caps, and full tracing.

| # | Phase | Deliverable | Status |
|---|-------|-------------|--------|
| 0 | Repo & env setup | git repo, MIT license, `.gitignore` (incl. `LEARNING_OUTCOMES.md`), `.env.example`, Python 3.11 venv | ✅ |
| 1 | Scaffold & packaging | `pyproject.toml` (package `agentforge`, PyPI-ready), CLI skeleton (`run`/`models`/`validate`/`estimate`), config models | ✅ |
| 2 | Config & validation | `pipeline.yaml` → Pydantic (`llm`, `budget`, `sandbox`, `permissions`, `agents`); clear load-time errors | ✅ |
| 3 | LLM provider layer | `LLMClient` base + OpenRouter, Anthropic, OpenAI, Ollama; token/cost accounting; `agentforge models` live probe | ✅ |
| 4 | Tool registry & built-ins | `@tool` decorator + schema; `web_search`, `read_url`, file/note tools (SSRF + workspace jail) | ✅ |
| 5 | Sandbox layer | `Sandbox` base + subprocess (default), Docker (opt), E2B (opt); `run_python` (opt-in) | ✅ |
| 6 | Permissions & HITL | risk classification, approval prompts (approve/deny/edit/always), CI-safe non-interactive policy, trace audit | ✅ |
| 7 | ReAct agent loop | native tool-calling + text-ReAct fallback; context truncation/trimming; per-agent loop caps | ✅ |
| 8 | Orchestrator | handoffs, budget enforcement, loop-safety guards, `trace.json` writer | ✅ |
| 9 | CLI, examples, docs, live E2E | 4 example pipelines ✅, README + Mermaid ✅, live 3-agent run ✅ (Ollama), public GitHub repo ✅, profile update ✅, CI green (258 tests) ✅ | ✅ |

**V1 done = all PRD success criteria met:** 3-agent pipeline runs end-to-end on a real model; trace shows tool calls + tokens + cost; works across ≥2 providers + Ollama; custom tool in <10 lines; budget cap aborts cleanly; sandbox blocks traversal/SSRF/timeouts; permission gate pauses correctly and never hangs in CI; loop caps exit gracefully.

---

## V2 Scope (deferred — design for it now, don't build it yet)

The V1 config and data structures are built DAG-ready so these don't require a rewrite. **Detailed design: [docs/V2_DAG_PARALLELISM.md](docs/V2_DAG_PARALLELISM.md)** — maps each V2 piece to the v1 seam it extends, with a 4-phase build plan.

| Item | What | Status |
|------|------|--------|
| DAG pipelines | Optional `depends_on` graph with topological execution + join (fan-in) | ✅ **Phase 1 done** (sequential exec; see `examples/research-dag.yaml`) |
| Parallel execution | Independent DAG branches run concurrently on a thread pool (`budget.max_parallel`, default 4); join waits for all | ✅ **Phase 2 done** (cooperative cancel on stop) |
| Parallel fan-out / fan-in | One agent spawns N worker instances over a parsed list (`fan_out: {to, max}`); a join merges | ✅ **Phase 3 done** (`examples/fanout-research.yaml`) |
| Concurrency-safe blackboard | `save_note`/`read_note` serialized under a lock for parallel agents | ✅ (notes lock) |
| Cross-branch budgeting | Hard cap governs parallel branches (thread-safe `BudgetGuard` + cancel_event) | ✅ |
| Resumable runs | Restart a failed run from its trace | 🔮 |
| Streaming/observability UI | Richer live view of concurrent branches | 🔮 |

---

## Decision Log

- **2026-06-14** — Providers: ship OpenRouter + Anthropic + OpenAI + Ollama in V1 (user chose multiple direct providers).
- **2026-06-14** — Sandbox: tiered abstraction (subprocess default + optional Docker + optional E2B).
- **2026-06-14** — Spend safety: hard per-run USD cap that aborts + live token/cost accounting.
- **2026-06-14** — Execution: sequential handoffs in V1; structures DAG-ready for V2 parallelism.
- **2026-06-14** — Added human-in-the-loop permission layer (approve/deny/edit; CI-safe).
- **2026-06-14** — Default model `deepseek/deepseek-v4-flash`; confirm slugs/pricing via live probe at build time (free Gemma 3 retired).

---

## Build Log

_(append-only; executor adds dated entries as phases complete)_

- _2026-06-14 — Planning complete; PRD finalized; build not yet started._
- **2026-06-15 — V2 Phase 3: dynamic fan-out.** A `fan_out: {to, max}` agent produces a list (parsed from its output via `_parse_list`: JSON array or bullet lines, capped at `max`); the scheduler spawns one instance of the `to` template per item (`to#0..to#k-1`) onto the same thread pool, and the join (`depends_on: [to]`) waits for the whole group (group-accounting in `_run_dag_parallel`). Generalized `run_agent` to run a template config under an instance name; the template must `depends_on` its source (validated at load). New `tests/test_fanout.py` + `examples/fanout-research.yaml`.
- **2026-06-15 — V2 Phase 2: parallel DAG execution.** Independent branches now run concurrently on a `ThreadPoolExecutor` (driver-thread ready-set scheduler, `_run_dag_parallel`) when `budget.max_parallel > 1` (default 4); `max_parallel: 1` keeps the deterministic sequential DAG path. Made the shared state thread-safe (`BudgetGuard` lock, `TraceRecorder.add_agent` lock, notes lock, atomic iteration guard) and added cooperative cancellation via a `cancel_event` (first real stop winds the rest down gracefully → partial result + trace). New `tests/test_parallel.py` with a `threading.Barrier` proof of real concurrency + a thread-safe agent-routed `RoutingFakeLLMClient`; `agent.py` unchanged (cancellation rides the existing `on_iteration` hook).
- **2026-06-14 — Post-V1 polish.** Added a live **web dashboard** (`agentforge serve`): FastAPI + SSE streams structured run events to a single-page UI (pick pipeline → set goal → Run → watch agents/tools/cost live → download trace); plumbed a no-op `emit` callback through the orchestrator + agent loop. Recorded a **demo GIF** of a real OpenRouter 3-agent run (captured the live cost meter + a workspace-jail block). Verified **PyPI readiness**: `python -m build` + `twine check` pass, wheel ships `web/static/index.html`, fresh-venv install runs the entry point; leaned the sdist. Wrote the **[V2 DAG/parallelism scoping doc](docs/V2_DAG_PARALLELISM.md)**. CI green at 259 tests.
- **2026-06-14 — Phase 9 + live E2E.** Wrote the full README (Mermaid diagrams adapted from the PRD, BYO-key callout, config reference, badges). Added an offline Ollama example (`research-pipeline.ollama.yaml`). Ran the 3-agent pipeline **live end-to-end** against `gemma4:e4b` (Ollama) — completed, real tool calls + token accounting + `trace.json` (saved `examples/sample-trace.json`). The live run surfaced and fixed three bugs the offline suite couldn't: (1) Ollama `base_url` not normalized to `/api/chat` → 405; (2) Ollama tool-call args serialized as a string instead of an object → 400 on the 2nd turn; (3) `web_search`/`read_url` wrongly blocked by `sandbox.network:false` (PRD says exempt) and the retired `duckduckgo_search` returning nothing → switched to `ddgs`. Drafted resume bullets + 2 LinkedIn posts (parent folder).
- **2026-06-14 — Phases 0–8 built (one milestone).** git repo + MIT + `.env.example` + PyPI-ready `pyproject.toml` (hatchling) + CI workflow. Locked the shared type contract (`messages.py`, `llm/base.py`, `tools/registry.py`, `sandbox/base.py`, `config.py`) then parallelized with subagents: LLM clients (OpenRouter/Anthropic/OpenAI/Ollama) + `cost.py`; built-in tools (SSRF-guarded web, workspace-jailed files, opt-in run_python) + tiered sandbox (subprocess/docker/e2b); permissions (approve/deny/edit/always, CI never-hang). Wrote the integration core myself: `context.py` (tiktoken truncation/trimming), `guards.py` (budget + repeated-action), `agent.py` (ReAct loop, native + text fallback), `orchestrator.py` (sequential handoffs, budget abort, loop guards), `trace.py` (trace.json + secret redaction). Model slugs verified against the live OpenRouter catalog: default `deepseek/deepseek-v4-flash` holds; retired `z-ai/glm-4.7-flash` swapped for `qwen/qwen3.6-flash`. CLI `validate` + keyless live `estimate` working; orchestration/budget/loop/redaction proven via scripted-LLM smoke runs. ~250-test pytest suite written (zero real API). **Remaining:** README + diagrams, live E2E, push, profile update.
