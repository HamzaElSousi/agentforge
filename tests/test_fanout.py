"""V2 Phase 3 — fan-out DAG execution tests.

Proves that a fan-out agent's list output spawns one instance of the template
per item (named ``<template>#0``, ``<template>#1``, ...), that instance counts
are capped by ``fan_out.max``, that the join (synthesizer) runs only after all
instances finish, and that the config validator rejects a fan-out target that
does not ``depends_on`` its source.

Fan-out topology used throughout:

    planner  (fan_out -> researcher, max: N)
        |
    researcher  (template — never run bare; spawned as researcher#0..#k-1)
        |
    synthesizer  (terminal join)

All assertions on the SET of instance names use ``set`` comparisons because
fan-out instances run concurrently and their scheduling order is
non-deterministic.
"""

from __future__ import annotations

import pytest

from agentforge.config import ConfigError, load_pipeline
from agentforge.cost import PricingCatalog
from agentforge.guards import StopReason
from agentforge.orchestrator import _parse_list, run_pipeline
from tests._helpers import write_pipeline
from tests.conftest import RoutingFakeLLMClient


# ---------------------------------------------------------------------------
# Autouse fixture: isolate each test in its own tmp directory so the runs/
# workspace and any trace files never cross-contaminate between tests.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _chdir_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _catalog() -> PricingCatalog:
    """Zero-pricing catalog — no real models, no real cost accounting."""
    return PricingCatalog.from_openrouter([])


def _fanout_pipeline(
    tmp_path,
    *,
    fan_out_max: int = 5,
    name: str = "fo-test",
) -> object:
    """Write a planner -> researcher (template) -> synthesizer pipeline.

    ``fan_out_max`` controls how many researcher instances the runtime may spawn.
    The YAML is constructed via write_pipeline so it uses the same minimal
    template as every other test in the suite.
    """
    agents_yaml = f"""\
  planner:
    role: split the goal into subtopics and output a JSON array
    tools: []
    max_iterations: 2
    fan_out:
      to: researcher
      max: {fan_out_max}
  researcher:
    role: research the assigned item
    tools: []
    depends_on: [planner]
    max_iterations: 2
  synthesizer:
    role: merge all researcher findings into a final report
    tools: []
    depends_on: [researcher]
    terminal: true
    max_iterations: 2
"""
    return write_pipeline(tmp_path, agents=agents_yaml, start="planner", name=name)


# ---------------------------------------------------------------------------
# Test a — spawns exactly one instance per item
# ---------------------------------------------------------------------------


def test_fan_out_spawns_one_instance_per_item(tmp_path):
    """Planner returns a 3-item JSON array; exactly 3 researcher instances appear
    in the trace, named researcher#0, researcher#1, researcher#2.

    The bare template name 'researcher' must NOT appear in the trace — the
    template is never run directly, only its instances are.
    """
    p = _fanout_pipeline(tmp_path)

    client = RoutingFakeLLMClient(
        {
            "planner": ['["topic_a", "topic_b", "topic_c"]'],
            "researcher": ["findings_a", "findings_b", "findings_c"],
            "synthesizer": ["FINAL REPORT"],
        }
    )

    trace = run_pipeline(
        pipeline_path=p,
        goal="test fan-out spawns one instance per item",
        trace_path=None,
        assume_yes=True,
        client=client,
        catalog=_catalog(),
    )

    assert trace["stopped_reason"] == StopReason.COMPLETED, (
        f"Pipeline should stop with COMPLETED; got {trace['stopped_reason']!r}"
    )

    all_names = {a["name"] for a in trace["agents"]}

    # The three instances must all be present.
    expected_instances = {"researcher#0", "researcher#1", "researcher#2"}
    assert expected_instances == all_names & expected_instances, (
        f"Expected instance names {expected_instances!r} in trace; "
        f"got agent names {all_names!r}"
    )

    # The bare template name must NOT appear — it is never run on its own.
    assert "researcher" not in all_names, (
        "The bare template 'researcher' must not appear in the trace; "
        f"only its instances should be present. Got names: {all_names!r}"
    )

    # The non-template agents must be present.
    assert "planner" in all_names, "planner must appear in the trace"
    assert "synthesizer" in all_names, "synthesizer must appear in the trace"


# ---------------------------------------------------------------------------
# Test b — max cap is respected
# ---------------------------------------------------------------------------


def test_fan_out_respects_max_cap(tmp_path):
    """Planner returns 10 items but fan_out.max=3; exactly 3 instances are spawned.

    The runtime must cap the fan-out at the configured maximum regardless of
    how many items the agent's output contains.
    """
    ten_items = '["i0","i1","i2","i3","i4","i5","i6","i7","i8","i9"]'

    # max=3 is baked into the pipeline by _fanout_pipeline(fan_out_max=3).
    p = _fanout_pipeline(tmp_path, fan_out_max=3, name="cap-test")

    client = RoutingFakeLLMClient(
        {
            "planner": [ten_items],
            # Supply enough replies for the capped 3 instances.
            "researcher": ["r0", "r1", "r2"],
            "synthesizer": ["MERGED"],
        }
    )

    trace = run_pipeline(
        pipeline_path=p,
        goal="test fan-out cap",
        trace_path=None,
        assume_yes=True,
        client=client,
        catalog=_catalog(),
    )

    assert trace["stopped_reason"] == StopReason.COMPLETED, (
        f"Expected COMPLETED; got {trace['stopped_reason']!r}"
    )

    instances = {a["name"] for a in trace["agents"] if "#" in a["name"]}
    assert len(instances) == 3, (
        f"fan_out.max=3 must cap spawn count at 3 instances; "
        f"got {len(instances)} instances: {instances!r}"
    )


# ---------------------------------------------------------------------------
# Test c — join receives all instance outputs (structural check)
# ---------------------------------------------------------------------------


def test_join_receives_all_instance_outputs(tmp_path):
    """The synthesizer (join) runs only after ALL researcher instances complete.

    Because instances share the template's reply list and execute concurrently,
    we can't assert exact per-item content without a custom fake.  Instead we
    verify the structural guarantee:
    - every spawned instance appears in the trace,
    - the synthesizer (terminal) is present and the pipeline completed,
    - the final_output is synthesizer's reply (not an instance's).

    This is the key join-semantics invariant: the synthesizer collects all
    instance results before it runs.
    """
    items = ["alpha", "beta", "gamma"]
    planner_response = '["alpha", "beta", "gamma"]'

    p = _fanout_pipeline(tmp_path, fan_out_max=5, name="join-test")

    client = RoutingFakeLLMClient(
        {
            "planner": [planner_response],
            # One reply per instance (3 instances share this list).
            "researcher": ["summary_alpha", "summary_beta", "summary_gamma"],
            "synthesizer": ["JOINED REPORT"],
        }
    )

    trace = run_pipeline(
        pipeline_path=p,
        goal="test join receives all outputs",
        trace_path=None,
        assume_yes=True,
        client=client,
        catalog=_catalog(),
    )

    assert trace["stopped_reason"] == StopReason.COMPLETED, (
        f"Pipeline should complete; got {trace['stopped_reason']!r}"
    )

    all_names = {a["name"] for a in trace["agents"]}
    instances = {n for n in all_names if "#" in n}

    # One instance per item.
    assert len(instances) == len(items), (
        f"Expected {len(items)} researcher instances; got {len(instances)}: {instances!r}"
    )

    # The synthesizer is the terminal/last recorded agent and produced the final output.
    assert "synthesizer" in all_names, (
        "synthesizer (join) must appear in the trace after all instances"
    )
    assert trace["final_output"] == "JOINED REPORT", (
        f"Final output should be synthesizer's reply; got {trace['final_output']!r}"
    )

    # Verify the synthesizer is the last agent recorded (it waits for all instances).
    last_agent_name = trace["agents"][-1]["name"]
    assert last_agent_name == "synthesizer", (
        f"synthesizer must be the last agent in the trace (ran after all instances); "
        f"got {last_agent_name!r}"
    )


# ---------------------------------------------------------------------------
# Test d — _parse_list: JSON array, bullet/numbered list, cap, empty
# ---------------------------------------------------------------------------


def test_parse_list_json_and_bullets():
    """_parse_list correctly handles JSON arrays, bullet/numbered lists, cap, and empty input."""
    # JSON array is preferred when a valid [...] span is present.
    result = _parse_list('["alpha", "beta", "gamma"]', max_items=10)
    assert result == ["alpha", "beta", "gamma"], (
        f"JSON array should parse to its elements; got {result!r}"
    )

    # JSON array embedded in surrounding prose.
    result = _parse_list('Here are the subtopics: ["x", "y", "z"] — done.', max_items=10)
    assert result == ["x", "y", "z"], (
        f"JSON array embedded in prose should be extracted; got {result!r}"
    )

    # Bullet list with "- " prefix.
    bullet_text = "- topic one\n- topic two\n- topic three"
    result = _parse_list(bullet_text, max_items=10)
    assert result == ["topic one", "topic two", "topic three"], (
        f"Bullet list should strip leading '- '; got {result!r}"
    )

    # Bullet list with "* " prefix.
    star_text = "* first\n* second\n* third"
    result = _parse_list(star_text, max_items=10)
    assert result == ["first", "second", "third"], (
        f"Bullet list should strip leading '* '; got {result!r}"
    )

    # Numbered list with "1. " prefix.
    numbered_text = "1. uno\n2. dos\n3. tres"
    result = _parse_list(numbered_text, max_items=10)
    assert result == ["uno", "dos", "tres"], (
        f"Numbered list should strip leading digits and dot; got {result!r}"
    )

    # Cap is respected: 5 items with max_items=3 returns only 3.
    long_json = '["a", "b", "c", "d", "e"]'
    result = _parse_list(long_json, max_items=3)
    assert result == ["a", "b", "c"], (
        f"max_items=3 should cap output at 3 items; got {result!r}"
    )

    # Cap also applies to line-based parsing.
    long_bullets = "- p\n- q\n- r\n- s\n- t"
    result = _parse_list(long_bullets, max_items=2)
    assert result == ["p", "q"], (
        f"max_items=2 should cap bullet list at 2 items; got {result!r}"
    )

    # Empty / whitespace-only input returns an empty list.
    assert _parse_list("", max_items=10) == [], (
        "Empty string should return []"
    )
    assert _parse_list("   \n  ", max_items=10) == [], (
        "Whitespace-only string should return []"
    )


# ---------------------------------------------------------------------------
# Test e — fan-out target must depend_on its source (validated at load time)
# ---------------------------------------------------------------------------


def test_fan_out_target_must_depend_on_source(tmp_path):
    """load_pipeline raises ConfigError when fan_out.to does not depends_on the source.

    The config validator enforces: if planner has fan_out.to = researcher then
    researcher must declare depends_on: [planner].  Omitting this dependency is
    a structural error caught before any money is spent.
    """
    # researcher deliberately omits depends_on: [planner]
    bad_agents = """\
  planner:
    role: plan
    tools: []
    max_iterations: 2
    fan_out:
      to: researcher
      max: 3
  researcher:
    role: research
    tools: []
    max_iterations: 2
  synthesizer:
    role: merge
    tools: []
    depends_on: [researcher]
    terminal: true
    max_iterations: 2
"""
    p = write_pipeline(tmp_path, agents=bad_agents, start="planner", name="bad-fanout")

    with pytest.raises(ConfigError, match="depends_on"):
        load_pipeline(p)
