"""V2 Phase 1 — DAG pipelines via ``depends_on`` (sequential execution).

Proves topological ordering, join (fan-in) context, the terminal-as-final rule,
the deadlock/cycle guard, and that the budget abort stays graceful in DAG mode.
"""

from __future__ import annotations

import pytest

from agentforge.config import ConfigError, load_pipeline
from agentforge.cost import ModelPricing, PricingCatalog
from agentforge.guards import StopReason
from agentforge.orchestrator import _topological_order, run_pipeline
from tests._helpers import write_pipeline
from tests.conftest import FakeLLMClient, text_response, tool_response

# planner -> (worker_a, worker_b) -> synth (join, terminal)
DAG_AGENTS = """\
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


@pytest.fixture(autouse=True)
def _chdir_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def _cat():
    return PricingCatalog.from_openrouter([])


def test_topological_order_start_first(tmp_path):
    p = write_pipeline(tmp_path, agents=DAG_AGENTS, start="planner")
    cfg = load_pipeline(p)
    order = _topological_order(cfg)
    assert order[0] == "planner"
    assert order[-1] == "synth"
    # dependencies always precede dependents
    assert order.index("planner") < order.index("worker_a")
    assert order.index("worker_a") < order.index("synth")
    assert order.index("worker_b") < order.index("synth")


def test_dag_completes_and_terminal_is_final(tmp_path):
    p = write_pipeline(tmp_path, agents=DAG_AGENTS, start="planner")
    client = FakeLLMClient([
        text_response("PLAN"), text_response("RESULT_A"),
        text_response("RESULT_B"), text_response("MERGED FINAL"),
    ])
    trace = run_pipeline(pipeline_path=p, goal="g", trace_path=None,
                         assume_yes=True, client=client, catalog=_cat())
    assert trace["stopped_reason"] == StopReason.COMPLETED
    assert trace["final_output"] == "MERGED FINAL"
    assert [a["name"] for a in trace["agents"]] == ["planner", "worker_a", "worker_b", "synth"]


def test_join_agent_receives_all_dependency_outputs(tmp_path):
    p = write_pipeline(tmp_path, agents=DAG_AGENTS, start="planner")
    client = FakeLLMClient([
        text_response("PLAN"), text_response("RESULT_A"),
        text_response("RESULT_B"), text_response("MERGED"),
    ])
    run_pipeline(pipeline_path=p, goal="g", trace_path=None,
                 assume_yes=True, client=client, catalog=_cat())
    synth_prompt = "\n".join(m.content for m in client.calls[3])
    assert "RESULT_A" in synth_prompt and "RESULT_B" in synth_prompt


def test_dag_records_depends_on_and_branch_id(tmp_path):
    p = write_pipeline(tmp_path, agents=DAG_AGENTS, start="planner")
    client = FakeLLMClient([text_response(x) for x in ("PLAN", "A", "B", "M")])
    trace = run_pipeline(pipeline_path=p, goal="g", trace_path=None,
                         assume_yes=True, client=client, catalog=_cat())
    synth = trace["agents"][-1]
    assert synth["depends_on"] == ["worker_a", "worker_b"]
    assert synth["branch_id"] == "synth"


def test_cycle_is_rejected_at_load(tmp_path):
    cyclic = """\
  a:
    role: a
    tools: []
    depends_on: [b]
    terminal: true
  b:
    role: b
    tools: []
    depends_on: [a]
"""
    p = write_pipeline(tmp_path, agents=cyclic, start="a")
    with pytest.raises(ConfigError, match="cycle"):
        load_pipeline(p)


def test_dag_budget_abort_is_graceful(tmp_path):
    cat = PricingCatalog({"fake/model": ModelPricing(2e-5, 2e-5)})
    p = write_pipeline(tmp_path, cap=0.05, agents=DAG_AGENTS, start="planner")
    # every agent keeps acting with big-token calls -> cap trips mid-DAG
    client = FakeLLMClient([
        tool_response("noop", {"i": i}, prompt_tokens=3000, completion_tokens=3000)
        for i in range(40)
    ])
    # 'noop' isn't a real tool, so it returns an error observation and the agent
    # keeps looping until the budget guard aborts — which must be graceful.
    trace = run_pipeline(pipeline_path=p, goal="g", trace_path=tmp_path / "t.json",
                         assume_yes=True, client=client, catalog=cat)
    assert trace["stopped_reason"] == StopReason.BUDGET_EXCEEDED
    assert (tmp_path / "t.json").exists()
