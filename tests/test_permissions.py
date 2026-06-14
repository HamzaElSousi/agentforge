"""Permission policy + human-in-the-loop, including the CI never-hang guarantee."""

from __future__ import annotations

import pytest

from agentforge.config import PermissionsConfig
from agentforge.messages import ToolCall
from agentforge.permissions import Decision, PermissionManager
from agentforge.tools.registry import Tool


def _tool(name: str, risk: str) -> Tool:
    return Tool(name=name, description="d", func=lambda **k: "", parameters={"type": "object"}, risk=risk)


READ = _tool("read_url", "read_only")
WRITE = _tool("write_file", "side_effecting")
DANGER = _tool("run_python", "dangerous")
CALL = ToolCall(id="c1", name="x", arguments={"a": 1})


# --- classification truth table -------------------------------------------- #


def test_deny_always_wins_over_auto_approve():
    cfg = PermissionsConfig(mode="auto", auto_approve=["write_file"], deny=["write_file"])
    pm = PermissionManager(cfg, interactive=False)
    assert pm.classify(WRITE) == "deny"


def test_prompt_mode_read_only_auto_side_effecting_gate():
    cfg = PermissionsConfig(mode="prompt")
    pm = PermissionManager(cfg, interactive=False)
    assert pm.classify(READ) == "auto"
    assert pm.classify(WRITE) == "gate"
    assert pm.classify(DANGER) == "gate"


def test_auto_mode_defaults_auto_but_require_approval_gates():
    cfg = PermissionsConfig(mode="auto", require_approval=["write_file"])
    pm = PermissionManager(cfg, interactive=False)
    assert pm.classify(READ) == "auto"
    assert pm.classify(WRITE) == "gate"


def test_strict_mode_gates_everything_unless_auto_approved():
    cfg = PermissionsConfig(mode="strict", auto_approve=["read_url"])
    pm = PermissionManager(cfg, interactive=False)
    assert pm.classify(READ) == "auto"   # explicit auto_approve overrides strict
    assert pm.classify(WRITE) == "gate"


# --- non-interactive never hangs (CI safety) ------------------------------- #


def test_non_interactive_deny_returns_immediately_without_input():
    cfg = PermissionsConfig(mode="prompt", non_interactive="deny")
    pm = PermissionManager(cfg, interactive=False)
    # Sabotage _ask so any attempt to prompt would raise — proving it's untouched.
    pm._ask = lambda *a, **k: pytest.fail("prompted in non-interactive mode!")
    d = pm.check(WRITE, CALL)
    assert d.outcome == Decision.denied
    assert d.auto is True


def test_non_interactive_allow_auto_approved_denies_gated_tool():
    cfg = PermissionsConfig(mode="prompt", non_interactive="allow_auto_approved")
    pm = PermissionManager(cfg, interactive=False)
    pm._ask = lambda *a, **k: pytest.fail("prompted in non-interactive mode!")
    # read-only is auto (runs); gated side-effecting is denied (never prompts).
    assert pm.check(READ, CALL).outcome == Decision.approved
    assert pm.check(WRITE, CALL).outcome == Decision.denied


def test_assume_yes_approves_gated_without_prompt():
    cfg = PermissionsConfig(mode="strict")
    pm = PermissionManager(cfg, interactive=False, assume_yes=True)
    pm._ask = lambda *a, **k: pytest.fail("prompted with assume_yes!")
    assert pm.check(WRITE, CALL).outcome == Decision.approved


# --- interactive approve / deny / edit / always ---------------------------- #


def test_interactive_approve_once():
    cfg = PermissionsConfig(mode="prompt")
    pm = PermissionManager(cfg, interactive=True)
    pm._ask = lambda *a, **k: "a"
    assert pm.check(WRITE, CALL).outcome == Decision.approved


def test_interactive_deny():
    cfg = PermissionsConfig(mode="prompt")
    pm = PermissionManager(cfg, interactive=True)
    pm._ask = lambda *a, **k: "d"
    assert pm.check(WRITE, CALL).outcome == Decision.denied


def test_interactive_edit_args():
    cfg = PermissionsConfig(mode="prompt")
    pm = PermissionManager(cfg, interactive=True)
    answers = iter(["e", '{"path": "safe.txt"}'])  # choose edit, then supply JSON
    pm._ask = lambda *a, **k: next(answers)
    d = pm.check(WRITE, CALL)
    assert d.outcome == Decision.edited
    assert d.edited_args == {"path": "safe.txt"}


def test_interactive_always_allow_persists_for_session():
    cfg = PermissionsConfig(mode="prompt")
    pm = PermissionManager(cfg, interactive=True)
    answers = iter(["A"])  # always-allow on first prompt
    pm._ask = lambda *a, **k: next(answers)
    first = pm.check(WRITE, CALL)
    assert first.outcome == Decision.approved
    # Second call must NOT prompt (sabotage _ask) — tool is now always-allowed.
    pm._ask = lambda *a, **k: pytest.fail("prompted after always-allow!")
    assert pm.check(WRITE, CALL).outcome == Decision.approved
