"""The ReAct loop: text-protocol parsing, native + fallback transports,
tool execution, denial handling, and output truncation."""

from __future__ import annotations

import pytest

from agentforge.agent import AgentRunner, parse_text_react
from agentforge.config import PermissionsConfig
from agentforge.cost import PricingCatalog
from agentforge.guards import BudgetGuard
from agentforge.permissions import PermissionManager
from agentforge.tools.registry import REGISTRY, Tool, ToolContext, tool
from agentforge.trace import AgentTrace
from tests.conftest import FakeLLMClient, text_response, tool_response


# --- text-ReAct parser ------------------------------------------------------ #


def test_parse_final_answer():
    p = parse_text_react("Thought: done.\nFinal Answer: the result is 42")
    assert p.final == "the result is 42"
    assert p.action is None


def test_parse_action_with_json_input():
    p = parse_text_react('Thought: search.\nAction: web_search\nAction Input: {"query": "cats"}')
    assert p.action == "web_search"
    assert p.action_input == {"query": "cats"}


def test_parse_action_with_fenced_json():
    text = 'Action: read_url\nAction Input: ```json\n{"url": "https://x.com"}\n```'
    p = parse_text_react(text)
    assert p.action == "read_url"
    assert p.action_input == {"url": "https://x.com"}


def test_parse_plain_prose_is_conclusion():
    p = parse_text_react("Here is my final summary with no protocol markers.")
    assert p.action is None
    assert "final summary" in p.final


def test_parse_final_wins_over_later_action():
    p = parse_text_react("Final Answer: stop now\nAction: web_search")
    assert p.final == "stop now\nAction: web_search" or p.final.startswith("stop now")


# --- AgentRunner ------------------------------------------------------------ #


def _runner(client, tools, *, perms_cfg=None, tmp_path=None, terminal=True, max_iterations=5):
    from agentforge.tools.files import Workspace

    at = AgentTrace(name="a", model="fake/model")
    ws = Workspace(tmp_path) if tmp_path else None
    ctx = ToolContext(workspace=ws, sandbox=None, network=True)
    perms = PermissionManager(perms_cfg or PermissionsConfig(mode="auto"),
                              interactive=False, assume_yes=True)
    budget = BudgetGuard(cap_usd=10.0, catalog=PricingCatalog.from_openrouter([]))
    return AgentRunner(
        name="a", role="do the thing", client=client, model="fake/model",
        tools=tools, ctx=ctx, permissions=perms, budget=budget, agent_trace=at,
        terminal=terminal, max_iterations=max_iterations,
    ), at


def test_native_tool_call_then_conclude(tmp_path):
    @tool(risk="read_only")
    def echo(ctx: ToolContext, msg: str) -> str:
        """Echo a message."""
        return f"echoed:{msg}"

    client = FakeLLMClient([
        tool_response("echo", {"msg": "hi"}),
        text_response("done with the work"),
    ])
    runner, at = _runner(client, [REGISTRY.get("echo")], tmp_path=tmp_path)
    result = runner.run("goal")
    assert result.kind == "final"
    assert "done with the work" in result.output
    assert at.tool_calls[0].tool == "echo"
    assert "echoed:hi" in at.tool_calls[0].result


def test_text_fallback_drives_tool(tmp_path):
    @tool(risk="read_only")
    def ping(ctx: ToolContext) -> str:
        """Return pong."""
        return "pong"

    # Non-native client: returns text protocol, then a final answer.
    client = FakeLLMClient(
        [text_response("Action: ping\nAction Input: {}"), text_response("Final Answer: complete")],
        supports_native_tools=False,
    )
    runner, at = _runner(client, [REGISTRY.get("ping")], tmp_path=tmp_path)
    result = runner.run("goal")
    assert result.kind == "final"
    assert at.tool_calls[0].tool == "ping"
    assert at.tool_calls[0].result == "pong"


def test_denied_tool_returns_observation_not_crash(tmp_path):
    @tool(risk="side_effecting")
    def danger(ctx: ToolContext) -> str:
        """Should never run."""
        raise AssertionError("must not execute when denied")

    cfg = PermissionsConfig(mode="prompt", deny=["danger"])
    client = FakeLLMClient([
        tool_response("danger", {}),
        text_response("adapted after denial"),
    ])
    runner, at = _runner(client, [REGISTRY.get("danger")], perms_cfg=cfg,
                         tmp_path=tmp_path)
    # assume_yes would approve; turn it off so deny policy applies.
    runner.permissions = PermissionManager(cfg, interactive=False, assume_yes=False)
    result = runner.run("goal")
    assert result.kind == "final"
    assert at.tool_calls[0].outcome == "denied"
    assert "denied" in at.tool_calls[0].result.lower()


def test_unknown_tool_is_reported(tmp_path):
    client = FakeLLMClient([
        tool_response("nonexistent", {}),
        text_response("Final Answer: handled"),
    ])
    runner, at = _runner(client, [], tmp_path=tmp_path)
    result = runner.run("goal")
    assert result.kind == "final"


def test_tool_output_is_truncated(tmp_path):
    @tool(risk="read_only")
    def big(ctx: ToolContext) -> str:
        """Return a huge string."""
        return "x" * 50000

    client = FakeLLMClient([tool_response("big", {}), text_response("done")])
    runner, at = _runner(client, [REGISTRY.get("big")], tmp_path=tmp_path)
    runner.run("goal")
    assert "omitted" in at.tool_calls[0].result
    assert len(at.tool_calls[0].result) < 50000
