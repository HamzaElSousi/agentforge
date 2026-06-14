# AgentForge — Roadmap & Progress

Living build tracker. The [PRD.md](PRD.md) is the design contract; this file tracks **what's in V1 vs V2** and the **current state of the build**. The executor updates the status tables and the log as work lands.

**Status legend:** ⬜ not started · 🟦 in progress · ✅ done · ⏸️ blocked · 🔮 V2 (deferred)

---

## Current State

- **Phase:** Phases 0–8 complete; Phase 9 (docs + live E2E) in progress.
- **Last updated:** 2026-06-14
- **Overall:** 8 / 9 V1 phases complete (core framework done; docs + live run remain).
- **Next action:** Phase 9 — README with Mermaid diagrams, then live 3-agent E2E ($0.25 cap), push, profile update.
- **Verified so far:** package installs editable; all modules import; `agentforge validate` passes on all 3 examples; `agentforge estimate` hits the live OpenRouter catalog keylessly (~$0.005 projected); orchestration (3-agent handoff + tool exec + trace), budget abort, repeated-action/iteration caps, and secret redaction all confirmed via scripted-LLM smoke runs. Full pytest suite (~250 tests) written; **run it with** `.venv/bin/python -m pytest tests/ -v`.

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
| 9 | CLI, examples, docs, live E2E | 3 example pipelines ✅, README ⬜, public GitHub repo ⬜, live 3-agent research run ⬜ | 🟦 |

**V1 done = all PRD success criteria met:** 3-agent pipeline runs end-to-end on a real model; trace shows tool calls + tokens + cost; works across ≥2 providers + Ollama; custom tool in <10 lines; budget cap aborts cleanly; sandbox blocks traversal/SSRF/timeouts; permission gate pauses correctly and never hangs in CI; loop caps exit gracefully.

---

## V2 Scope (deferred — design for it now, don't build it yet)

The V1 config and data structures are built DAG-ready so these don't require a rewrite.

| Item | What | Status |
|------|------|--------|
| DAG pipelines | Replace single `handoff_to`/`start` with optional `depends_on` graph | 🔮 |
| Parallel fan-out / fan-in | One agent spawns N parallel workers; a join agent merges results | 🔮 |
| Concurrency-safe blackboard | Generalize `save_note`/`read_note` into a shared store for parallel agents | 🔮 |
| Cross-branch budgeting | Per-run cap governs parallel branches (orchestrator-level, already designed for it) | 🔮 |
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
- **2026-06-14 — Phases 0–8 built (one milestone).** git repo + MIT + `.env.example` + PyPI-ready `pyproject.toml` (hatchling) + CI workflow. Locked the shared type contract (`messages.py`, `llm/base.py`, `tools/registry.py`, `sandbox/base.py`, `config.py`) then parallelized with subagents: LLM clients (OpenRouter/Anthropic/OpenAI/Ollama) + `cost.py`; built-in tools (SSRF-guarded web, workspace-jailed files, opt-in run_python) + tiered sandbox (subprocess/docker/e2b); permissions (approve/deny/edit/always, CI never-hang). Wrote the integration core myself: `context.py` (tiktoken truncation/trimming), `guards.py` (budget + repeated-action), `agent.py` (ReAct loop, native + text fallback), `orchestrator.py` (sequential handoffs, budget abort, loop guards), `trace.py` (trace.json + secret redaction). Model slugs verified against the live OpenRouter catalog: default `deepseek/deepseek-v4-flash` holds; retired `z-ai/glm-4.7-flash` swapped for `qwen/qwen3.6-flash`. CLI `validate` + keyless live `estimate` working; orchestration/budget/loop/redaction proven via scripted-LLM smoke runs. ~250-test pytest suite written (zero real API). **Remaining:** README + diagrams, live E2E, push, profile update.
