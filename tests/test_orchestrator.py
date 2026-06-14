"""Orchestration behavior: handoffs, budget abort, and loop-safety guards.

All deterministic via the scripted FakeLLMClient — zero real API calls.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from agentforge.cost import ModelPricing, PricingCatalog
from agentforge.guards import StopReason
from agentforge.orchestrator import run_pipeline
from tests._helpers import write_pipeline
from tests.conftest import FakeLLMClient, text_response, tool_response

THREE_AGENTS = """\
  researcher:
    role: search and collect facts
    tools: [web_search, save_note]
    handoff_to: writer
    max_iterations: 5
  writer:
    role: write the analysis
    tools: [save_note]
    handoff_to: reviewer
    max_iterations: 5
  reviewer:
    role: review and finalize
    tools: []
    terminal: true
    max_iterations: 5
"""


@pytest.fixture(autouse=True)
def _chdir_tmp(tmp_path, monkeypatch):
    """Run each test in an isolated CWD so runs/ + trace land in tmp."""
    monkeypatch.chdir(tmp_path)


def _fallback_catalog():
    return PricingCatalog.from_openrouter([])


def test_three_agent_handoff_completes(tmp_path):
    p = write_pipeline(tmp_path, agents=THREE_AGENTS, start="researcher")
    client = FakeLLMClient([
        tool_response("web_search", {"query": "facts"}),
        text_response("Collected 5 facts."),     # researcher -> writer
        text_response("Wrote the analysis."),     # writer -> reviewer
        text_response("FINAL approved version."), # reviewer terminal
    ])
    trace = run_pipeline(
        pipeline_path=p, goal="g", trace_path=tmp_path / "trace.json",
        assume_yes=True, client=client, catalog=_fallback_catalog(),
    )
    assert trace["stopped_reason"] == StopReason.COMPLETED
    assert [a["name"] for a in trace["agents"]] == ["researcher", "writer", "reviewer"]
    assert trace["agents"][0]["tool_calls"][0]["tool"] == "web_search"
    assert "FINAL" in trace["final_output"]
    assert (tmp_path / "trace.json").exists()


def test_handoff_context_passed_to_next_agent(tmp_path):
    p = write_pipeline(tmp_path, agents=THREE_AGENTS, start="researcher")
    client = FakeLLMClient([
        text_response("HANDOFF-PAYLOAD-123"),  # researcher concludes immediately
        text_response("writer done"),
        text_response("reviewer final"),
    ])
    run_pipeline(pipeline_path=p, goal="g", trace_path=None,
                 assume_yes=True, client=client, catalog=_fallback_catalog())
    # The second agent's first call must contain the researcher's handoff text.
    writer_first_call = client.calls[1]
    joined = "\n".join(m.content for m in writer_first_call)
    assert "HANDOFF-PAYLOAD-123" in joined


def test_budget_aborts_with_partial_and_trace(tmp_path):
    # Price the fake model so a few calls accumulate, then the cap trips.
    cat = PricingCatalog({"fake/model": ModelPricing(2e-5, 2e-5)})
    p = write_pipeline(tmp_path, cap=0.20, agents=THREE_AGENTS, start="researcher")
    # Vary args each turn so repeated-action guard doesn't fire first; report
    # large usage so cost accumulates quickly.
    script = [
        tool_response("web_search", {"query": f"q{i}"},
                      prompt_tokens=2000, completion_tokens=2000)
        for i in range(20)
    ]
    client = FakeLLMClient(script)
    trace = run_pipeline(pipeline_path=p, goal="g", trace_path=tmp_path / "trace.json",
                         assume_yes=True, client=client, catalog=cat)
    assert trace["stopped_reason"] == StopReason.BUDGET_EXCEEDED
    assert trace["cost"]["total_usd"] > 0
    # The conservative pre-call check stops new calls once the cap is in reach;
    # a single in-flight call may overshoot by at most ~one call's cost (you
    # can't know a call's true cost until it returns). It must not run away.
    assert trace["cost"]["total_usd"] <= 0.20 * 2
    assert (tmp_path / "trace.json").exists()  # partial trace still written


def test_repeated_action_stops_gracefully(tmp_path):
    p = write_pipeline(tmp_path, agents=THREE_AGENTS, start="researcher")
    # Same tool + same args every time -> repeated-action guard (threshold 3).
    client = FakeLLMClient([tool_response("web_search", {"query": "same"})] * 10)
    trace = run_pipeline(pipeline_path=p, goal="g", trace_path=None,
                         assume_yes=True, client=client, catalog=_fallback_catalog())
    assert trace["stopped_reason"] == StopReason.REPEATED_ACTION


def test_max_total_iterations_caps_pipeline(tmp_path):
    # Tiny pipeline iteration cap; agents keep acting with varied args.
    p = write_pipeline(tmp_path, total_iters=2, agents=THREE_AGENTS, start="researcher")
    client = FakeLLMClient([
        tool_response("web_search", {"query": f"q{i}"}) for i in range(20)
    ])
    trace = run_pipeline(pipeline_path=p, goal="g", trace_path=None,
                         assume_yes=True, client=client, catalog=_fallback_catalog())
    assert trace["stopped_reason"] == StopReason.MAX_TOTAL_ITERATIONS


def test_max_iterations_per_agent_then_handoff(tmp_path):
    # Researcher never concludes (always acts) -> hits its 5-iter cap.
    agents = THREE_AGENTS
    p = write_pipeline(tmp_path, agents=agents, start="researcher")
    client = FakeLLMClient([
        tool_response("web_search", {"query": f"q{i}"}) for i in range(20)
    ])
    trace = run_pipeline(pipeline_path=p, goal="g", trace_path=None,
                         assume_yes=True, client=client, catalog=_fallback_catalog())
    # First agent stops at its cap; pipeline returns a partial result, no crash.
    assert trace["stopped_reason"] in (StopReason.MAX_ITERATIONS, StopReason.MAX_TOTAL_ITERATIONS)
    assert trace["agents"][0]["iterations"] <= 5


def test_secrets_redacted_in_trace(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-SECRET-TOKEN-XYZ")
    p = write_pipeline(tmp_path, agents=THREE_AGENTS, start="researcher")
    # Echo the secret back in the model output to prove redaction on write.
    client = FakeLLMClient([
        text_response("leaking sk-SECRET-TOKEN-XYZ here"),
        text_response("writer"),
        text_response("reviewer final"),
    ])
    run_pipeline(pipeline_path=p, goal="g", trace_path=tmp_path / "trace.json",
                 assume_yes=True, client=client, catalog=_fallback_catalog())
    data = (tmp_path / "trace.json").read_text()
    assert "sk-SECRET-TOKEN-XYZ" not in data
    assert "***" in data
