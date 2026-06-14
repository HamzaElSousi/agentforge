"""Tests for agentforge/trace.py — TraceRecorder, Trace, AgentTrace, secret redaction.

Verified source behaviors:
- add_agent() appends an AgentTrace and returns it for mutation.
- finalize() sets duration_ms, cost rollup, stopped_reason, final_output.
- cost dict keys: total_usd, prompt_tokens, completion_tokens.
- as_json() / write() both apply _redact() before serializing.
- _redact() replaces all occurrences of each secret with '***' recursively.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from agentforge.trace import (
    AgentTrace,
    ToolCallRecord,
    Trace,
    TraceRecorder,
    _redact,
)


# ---------------------------------------------------------------------------
# _redact helper
# ---------------------------------------------------------------------------


class TestRedact:
    def test_redacts_secret_in_plain_string(self):
        result = _redact("my key is sk-SECRET here", ["sk-SECRET"])
        assert "sk-SECRET" not in result
        assert "***" in result

    def test_redacts_secret_in_nested_dict(self):
        obj = {"key": "value with sk-SECRET inside"}
        result = _redact(obj, ["sk-SECRET"])
        assert "sk-SECRET" not in result["key"]
        assert "***" in result["key"]

    def test_redacts_secret_in_list(self):
        obj = ["safe", "has sk-SECRET here", "also safe"]
        result = _redact(obj, ["sk-SECRET"])
        assert "sk-SECRET" not in result[1]
        assert "***" in result[1]

    def test_no_secrets_list_leaves_string_unchanged(self):
        result = _redact("plain string", [])
        assert result == "plain string"

    def test_empty_secret_is_ignored(self):
        """Empty string secrets must not alter output (avoid replacing every char)."""
        result = _redact("hello world", [""])
        assert result == "hello world"

    def test_multiple_secrets_all_redacted(self):
        result = _redact("key1=AAAA key2=BBBB", ["AAAA", "BBBB"])
        assert "AAAA" not in result
        assert "BBBB" not in result

    def test_non_string_values_pass_through(self):
        obj = {"count": 42, "flag": True, "nothing": None}
        result = _redact(obj, ["sk-SECRET"])
        assert result == {"count": 42, "flag": True, "nothing": None}


# ---------------------------------------------------------------------------
# TraceRecorder — add_agent
# ---------------------------------------------------------------------------


class TestAddAgent:
    def test_add_agent_returns_agent_trace(self):
        rec = TraceRecorder("pipe", "goal")
        at = rec.add_agent("agent1", "model/v1")
        assert isinstance(at, AgentTrace)

    def test_add_agent_appended_to_trace_agents(self):
        rec = TraceRecorder("pipe", "goal")
        rec.add_agent("a1", "model")
        rec.add_agent("a2", "model")
        assert len(rec.trace.agents) == 2

    def test_add_agent_name_and_model_set(self):
        rec = TraceRecorder("pipe", "goal")
        at = rec.add_agent("my_agent", "deepseek/v4")
        assert at.name == "my_agent"
        assert at.model == "deepseek/v4"

    def test_agent_trace_mutable_by_caller(self):
        rec = TraceRecorder("pipe", "goal")
        at = rec.add_agent("a", "m")
        at.iterations = 5
        at.cost_usd = 0.01
        # Changes should be visible in rec.trace
        assert rec.trace.agents[0].iterations == 5


# ---------------------------------------------------------------------------
# TraceRecorder — finalize
# ---------------------------------------------------------------------------


class TestFinalize:
    def _recorder_with_two_agents(self) -> TraceRecorder:
        rec = TraceRecorder("test-pipeline", "Do something")
        at1 = rec.add_agent("agent1", "model-a")
        at1.iterations = 3
        at1.cost_usd = 0.001
        at1.prompt_tokens = 500
        at1.completion_tokens = 200

        at2 = rec.add_agent("agent2", "model-b")
        at2.iterations = 2
        at2.cost_usd = 0.002
        at2.prompt_tokens = 300
        at2.completion_tokens = 100
        return rec

    def test_finalize_returns_trace(self):
        rec = TraceRecorder("pipe", "goal")
        trace = rec.finalize(stopped_reason="completed", final_output="done")
        assert isinstance(trace, Trace)

    def test_finalize_sets_stopped_reason(self):
        rec = self._recorder_with_two_agents()
        trace = rec.finalize(stopped_reason="budget_exceeded", final_output="partial")
        assert trace.stopped_reason == "budget_exceeded"

    def test_finalize_sets_final_output(self):
        rec = self._recorder_with_two_agents()
        trace = rec.finalize(stopped_reason="completed", final_output="The answer is 42")
        assert trace.final_output == "The answer is 42"

    def test_finalize_sets_duration_ms_non_negative(self):
        rec = self._recorder_with_two_agents()
        trace = rec.finalize(stopped_reason="completed", final_output="x")
        assert trace.duration_ms >= 0, "duration_ms must be non-negative"

    def test_finalize_cost_total_usd_sums_agents(self):
        rec = self._recorder_with_two_agents()
        trace = rec.finalize(stopped_reason="completed", final_output="x")
        assert trace.cost["total_usd"] == pytest.approx(0.003)

    def test_finalize_cost_prompt_tokens_sums_agents(self):
        rec = self._recorder_with_two_agents()
        trace = rec.finalize(stopped_reason="completed", final_output="x")
        assert trace.cost["prompt_tokens"] == 800  # 500 + 300

    def test_finalize_cost_completion_tokens_sums_agents(self):
        rec = self._recorder_with_two_agents()
        trace = rec.finalize(stopped_reason="completed", final_output="x")
        assert trace.cost["completion_tokens"] == 300  # 200 + 100

    def test_finalize_cost_dict_has_required_keys(self):
        rec = TraceRecorder("pipe", "goal")
        rec.add_agent("a", "m")
        trace = rec.finalize(stopped_reason="completed", final_output="x")
        assert "total_usd" in trace.cost
        assert "prompt_tokens" in trace.cost
        assert "completion_tokens" in trace.cost


# ---------------------------------------------------------------------------
# TraceRecorder — to_dict shape / PRD keys
# ---------------------------------------------------------------------------


class TestToDict:
    def test_to_dict_contains_pipeline_key(self):
        rec = TraceRecorder("my-pipeline", "goal")
        trace = rec.finalize(stopped_reason="completed", final_output="x")
        d = trace.to_dict()
        assert d["pipeline"] == "my-pipeline"

    def test_to_dict_contains_goal_key(self):
        rec = TraceRecorder("pipe", "do something useful")
        trace = rec.finalize(stopped_reason="completed", final_output="x")
        d = trace.to_dict()
        assert d["goal"] == "do something useful"

    def test_to_dict_contains_duration_ms(self):
        rec = TraceRecorder("pipe", "goal")
        trace = rec.finalize(stopped_reason="completed", final_output="x")
        d = trace.to_dict()
        assert "duration_ms" in d
        assert isinstance(d["duration_ms"], int)

    def test_to_dict_contains_cost_dict(self):
        rec = TraceRecorder("pipe", "goal")
        trace = rec.finalize(stopped_reason="completed", final_output="x")
        d = trace.to_dict()
        assert "cost" in d
        assert isinstance(d["cost"], dict)

    def test_to_dict_contains_stopped_reason(self):
        rec = TraceRecorder("pipe", "goal")
        trace = rec.finalize(stopped_reason="loop_cap", final_output="x")
        d = trace.to_dict()
        assert d["stopped_reason"] == "loop_cap"

    def test_to_dict_contains_agents_list(self):
        rec = TraceRecorder("pipe", "goal")
        rec.add_agent("a", "m")
        trace = rec.finalize(stopped_reason="completed", final_output="x")
        d = trace.to_dict()
        assert "agents" in d
        assert isinstance(d["agents"], list)
        assert len(d["agents"]) == 1

    def test_to_dict_contains_final_output(self):
        rec = TraceRecorder("pipe", "goal")
        trace = rec.finalize(stopped_reason="completed", final_output="final text")
        d = trace.to_dict()
        assert d["final_output"] == "final text"

    def test_to_dict_agent_entry_has_expected_fields(self):
        rec = TraceRecorder("pipe", "goal")
        at = rec.add_agent("worker", "model/x")
        at.iterations = 4
        at.cost_usd = 0.01
        trace = rec.finalize(stopped_reason="completed", final_output="x")
        agent_dict = trace.to_dict()["agents"][0]
        assert agent_dict["name"] == "worker"
        assert agent_dict["model"] == "model/x"
        assert agent_dict["iterations"] == 4
        assert "tool_calls" in agent_dict


# ---------------------------------------------------------------------------
# ToolCallRecord in trace
# ---------------------------------------------------------------------------


class TestToolCallRecord:
    def test_tool_call_record_appended_to_agent(self):
        rec = TraceRecorder("pipe", "goal")
        at = rec.add_agent("a", "m")
        tcr = ToolCallRecord(
            tool="write_file",
            args={"path": "out.txt", "content": "data"},
            result="Written 4 chars to 'out.txt'.",
        )
        at.tool_calls.append(tcr)
        assert len(at.tool_calls) == 1
        assert at.tool_calls[0].tool == "write_file"

    def test_tool_call_serialized_in_to_dict(self):
        rec = TraceRecorder("pipe", "goal")
        at = rec.add_agent("a", "m")
        at.tool_calls.append(ToolCallRecord(
            tool="read_url",
            args={"url": "https://example.com"},
            result="content",
        ))
        trace = rec.finalize(stopped_reason="completed", final_output="x")
        agent_dict = trace.to_dict()["agents"][0]
        assert len(agent_dict["tool_calls"]) == 1
        assert agent_dict["tool_calls"][0]["tool"] == "read_url"


# ---------------------------------------------------------------------------
# Secret redaction in as_json and write
# ---------------------------------------------------------------------------


class TestSecretRedaction:
    SECRET = "sk-SECRET-abc123"

    def _recorder_with_secret(self) -> TraceRecorder:
        rec = TraceRecorder("pipe", "goal", secrets=[self.SECRET])
        at = rec.add_agent("a", "m")
        at.tool_calls.append(ToolCallRecord(
            tool="web_search",
            args={"query": f"query with {self.SECRET} embedded"},
            result=f"Result containing {self.SECRET}",
        ))
        rec.finalize(
            stopped_reason="completed",
            final_output=f"Final output with {self.SECRET} exposed",
        )
        return rec

    def test_as_json_does_not_contain_secret(self):
        rec = self._recorder_with_secret()
        j = rec.as_json()
        assert self.SECRET not in j, "as_json() must not contain the raw secret"

    def test_as_json_contains_redaction_marker(self):
        rec = self._recorder_with_secret()
        j = rec.as_json()
        assert "***" in j, "as_json() must contain '***' where secrets were"

    def test_write_does_not_contain_secret(self, tmp_path: Path):
        rec = self._recorder_with_secret()
        p = rec.write(tmp_path / "trace.json")
        content = p.read_text(encoding="utf-8")
        assert self.SECRET not in content, "trace.json must not contain the raw secret"

    def test_write_contains_redaction_marker(self, tmp_path: Path):
        rec = self._recorder_with_secret()
        p = rec.write(tmp_path / "trace.json")
        content = p.read_text(encoding="utf-8")
        assert "***" in content

    def test_write_produces_valid_json(self, tmp_path: Path):
        rec = self._recorder_with_secret()
        p = rec.write(tmp_path / "trace.json")
        # Should not raise
        data = json.loads(p.read_text(encoding="utf-8"))
        assert isinstance(data, dict)

    def test_redaction_in_final_output(self):
        rec = TraceRecorder("pipe", "goal", secrets=[self.SECRET])
        rec.add_agent("a", "m")
        rec.finalize(
            stopped_reason="completed",
            final_output=f"Output: {self.SECRET}",
        )
        j = rec.as_json()
        assert self.SECRET not in j

    def test_redaction_in_tool_args(self):
        rec = TraceRecorder("pipe", "goal", secrets=[self.SECRET])
        at = rec.add_agent("a", "m")
        at.tool_calls.append(ToolCallRecord(
            tool="fetch",
            args={"api_key": self.SECRET},
            result="ok",
        ))
        rec.finalize(stopped_reason="completed", final_output="x")
        j = rec.as_json()
        assert self.SECRET not in j

    def test_pipeline_with_no_secrets_unaffected(self):
        """With no secrets, as_json() must not introduce spurious '***'."""
        rec = TraceRecorder("pipe", "goal")  # no secrets
        at = rec.add_agent("a", "m")
        rec.finalize(stopped_reason="completed", final_output="clean output")
        j = rec.as_json()
        assert "***" not in j, "No secrets registered — no redaction markers expected"

    def test_empty_secret_string_not_registered(self):
        """Empty string secrets must be filtered out and not corrupt output."""
        rec = TraceRecorder("pipe", "goal", secrets=["", None, ""])  # type: ignore[list-item]
        rec.add_agent("a", "m")
        rec.finalize(stopped_reason="completed", final_output="normal output")
        j = rec.as_json()
        # Should parse cleanly and not mangle strings
        data = json.loads(j)
        assert data["final_output"] == "normal output"
