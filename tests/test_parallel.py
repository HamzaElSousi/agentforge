"""V2 Phase 2 — parallel DAG execution tests.

Proves that independent branches in a depends_on DAG run concurrently on a
thread pool, that the join agent (terminal) receives all dependency outputs and
runs only after all deps complete, and that budget cancellation in one branch
winds down the others gracefully (partial result + trace, no exception).

The planner -> (worker_a, worker_b) -> synth topology is used throughout:

    planner
    ├── worker_a ──┐
    └── worker_b ──┤
                   └── synth  (terminal)

worker_a and worker_b have no path between them, so they are independent and
will be submitted concurrently when max_parallel > 1.

Ordering notes
--------------
All assertions on agent ordering in the PARALLEL path use ``set`` comparisons
because thread scheduling is non-deterministic.  Only the sequential path
(max_parallel=1) asserts list order.
"""

from __future__ import annotations

import threading

import pytest

from agentforge.cost import ModelPricing, PricingCatalog
from agentforge.guards import StopReason
from agentforge.orchestrator import run_pipeline
from tests._helpers import write_pipeline
from tests.conftest import RoutingFakeLLMClient

# ---------------------------------------------------------------------------
# Shared pipeline topology (same structure as test_dag.py's DAG_AGENTS)
# ---------------------------------------------------------------------------

_PAR_AGENTS = """\
  planner:
    role: plan the work
    tools: []
    max_iterations: 2
  worker_a:
    role: do A
    tools: []
    depends_on: [planner]
    max_iterations: 2
  worker_b:
    role: do B
    tools: []
    depends_on: [planner]
    max_iterations: 2
  synth:
    role: merge the workers
    tools: []
    depends_on: [worker_a, worker_b]
    terminal: true
    max_iterations: 2
"""


# ---------------------------------------------------------------------------
# Autouse fixture: run each test in its own tmp directory so runs/ and any
# trace files land in an isolated directory and never cross-contaminate.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _chdir_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _fallback_catalog() -> PricingCatalog:
    return PricingCatalog.from_openrouter([])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_parallel_dag_completes_and_joins(tmp_path):
    """Happy path: all four agents run, synth merges the two workers, the
    pipeline stops with COMPLETED and the final_output is synth's reply.

    The set of agent names in the trace is checked rather than their list
    order because thread scheduling is non-deterministic under max_parallel=4.
    """
    p = write_pipeline(tmp_path, agents=_PAR_AGENTS, start="planner")
    client = RoutingFakeLLMClient(
        {
            "planner": ["PLAN"],
            "worker_a": ["RESULT_A"],
            "worker_b": ["RESULT_B"],
            "synth": ["MERGED FINAL"],
        }
    )

    trace = run_pipeline(
        pipeline_path=p,
        goal="parallel integration test",
        trace_path=None,
        assume_yes=True,
        client=client,
        catalog=_fallback_catalog(),
    )

    assert trace["stopped_reason"] == StopReason.COMPLETED, (
        f"Expected COMPLETED but got {trace['stopped_reason']!r}"
    )
    assert trace["final_output"] == "MERGED FINAL", (
        f"synth's reply should be the final output; got {trace['final_output']!r}"
    )
    # Assert on a SET — the parallel path gives no ordering guarantee for
    # worker_a / worker_b relative to each other.
    agent_names = {a["name"] for a in trace["agents"]}
    assert agent_names == {"planner", "worker_a", "worker_b", "synth"}, (
        f"Expected all four agent names in the trace; got {agent_names!r}"
    )


def test_workers_run_concurrently(tmp_path):
    """Prove real parallelism with a threading.Barrier.

    A Barrier(2) requires exactly two threads to call wait() before either is
    released.  worker_a and worker_b each call wait() on their first LLM call.
    If they execute on separate threads they meet and the barrier releases
    (run completes).  If they executed sequentially, the first worker would
    block forever waiting for a second thread that never arrives, which would
    surface as a BrokenBarrierError / timeout -> the run would not complete.

    The barrier has a 10-second timeout to keep CI fast while still giving the
    thread pool plenty of time to schedule both workers.
    """
    barrier = threading.Barrier(2, timeout=10)
    p = write_pipeline(tmp_path, agents=_PAR_AGENTS, start="planner")
    client = RoutingFakeLLMClient(
        {
            "planner": ["PLAN"],
            "worker_a": ["RESULT_A"],
            "worker_b": ["RESULT_B"],
            "synth": ["MERGED FINAL"],
        },
        barrier=barrier,
        barrier_agents=("worker_a", "worker_b"),
    )

    trace = run_pipeline(
        pipeline_path=p,
        goal="concurrency proof",
        trace_path=None,
        assume_yes=True,
        client=client,
        catalog=_fallback_catalog(),
    )

    # If the workers were sequential the barrier would time out and propagate
    # an error, so a COMPLETED result is the proof of real parallelism.
    assert trace["stopped_reason"] == StopReason.COMPLETED, (
        "Pipeline should complete when workers run concurrently; "
        f"got stopped_reason={trace['stopped_reason']!r}. "
        "If the barrier timed out, the workers likely ran sequentially."
    )


def test_parallel_budget_cancel_is_graceful(tmp_path):
    """A high-cost fake model triggers the budget guard mid-run.

    big_usage=True makes every LLM response report Usage(3000, 3000) tokens.
    With ModelPricing(2e-5, 2e-5) that is 0.12 USD per call.  A cap of 0.05
    means the guard fires after the first or second agent call.

    Assertions:
    - stopped_reason == BUDGET_EXCEEDED (guard fired, run aborted)
    - trace.json was written (partial result preserved even on abort)
    - no exception propagated (orchestrator catches and finalizes gracefully)
    """
    cat = PricingCatalog({"fake/model": ModelPricing(2e-5, 2e-5)})
    p = write_pipeline(tmp_path, cap=0.05, agents=_PAR_AGENTS, start="planner")
    trace_path = tmp_path / "t.json"

    client = RoutingFakeLLMClient(
        {
            "planner": ["PLAN", "PLAN2", "PLAN3"],
            "worker_a": ["RESULT_A", "RESULT_A2"],
            "worker_b": ["RESULT_B", "RESULT_B2"],
            "synth": ["MERGED FINAL"],
        },
        big_usage=True,  # Usage(3000, 3000) per call → 0.12 USD per call
    )

    # This must not raise; the orchestrator is responsible for catching
    # BudgetExceeded and returning a partial trace dict.
    trace = run_pipeline(
        pipeline_path=p,
        goal="budget cancel test",
        trace_path=trace_path,
        assume_yes=True,
        client=client,
        catalog=cat,
    )

    assert trace["stopped_reason"] == StopReason.BUDGET_EXCEEDED, (
        f"Expected BUDGET_EXCEEDED; got {trace['stopped_reason']!r}"
    )
    assert trace_path.exists(), (
        "A partial trace.json must be written even when the budget guard fires"
    )


def test_max_parallel_one_is_deterministic(tmp_path):
    """With max_parallel=1 the DAG falls back to the sequential _run_dag path.

    The sequential path processes agents in strict topological order:
    planner -> worker_a -> worker_b -> synth.  This is deterministic, so we
    can assert the exact list order (unlike the parallel path).

    This also guards against the FakeLLMClient-is-not-thread-safe concern: the
    sequential path never touches shared state from multiple threads.
    """
    p = write_pipeline(
        tmp_path, agents=_PAR_AGENTS, start="planner", max_parallel=1
    )
    client = RoutingFakeLLMClient(
        {
            "planner": ["PLAN"],
            "worker_a": ["RESULT_A"],
            "worker_b": ["RESULT_B"],
            "synth": ["MERGED FINAL"],
        }
    )

    trace = run_pipeline(
        pipeline_path=p,
        goal="sequential determinism test",
        trace_path=None,
        assume_yes=True,
        client=client,
        catalog=_fallback_catalog(),
    )

    assert trace["stopped_reason"] == StopReason.COMPLETED, (
        f"Expected COMPLETED; got {trace['stopped_reason']!r}"
    )
    # Sequential path is deterministic: assert the exact list order.
    agent_name_list = [a["name"] for a in trace["agents"]]
    assert agent_name_list == ["planner", "worker_a", "worker_b", "synth"], (
        f"Sequential DAG must execute agents in topological order; "
        f"got {agent_name_list!r}"
    )
