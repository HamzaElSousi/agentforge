"""Tests for agentforge/config.py — pipeline YAML loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from agentforge.config import (
    BudgetConfig,
    ConfigError,
    PermissionMode,
    PermissionsConfig,
    PipelineConfig,
    load_pipeline,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def write_yaml(tmp_path: Path, data: dict) -> Path:
    """Write a dict as YAML and return the path."""
    p = tmp_path / "pipeline.yaml"
    p.write_text(yaml.dump(data), encoding="utf-8")
    return p


def minimal_pipeline(**overrides) -> dict:
    """Return a minimal valid pipeline dict, with optional overrides merged."""
    base = {
        "name": "test-pipeline",
        "start": "worker",
        "agents": {
            "worker": {
                "role": "Does things",
                "terminal": True,
            }
        },
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_valid_single_agent_pipeline_loads(tmp_path):
    """A minimal valid pipeline with one terminal agent loads without error."""
    p = write_yaml(tmp_path, minimal_pipeline())
    cfg = load_pipeline(p)
    assert cfg.name == "test-pipeline"
    assert cfg.start == "worker"
    assert "worker" in cfg.agents
    assert cfg.agents["worker"].terminal is True


def test_valid_two_agent_handoff_pipeline_loads(tmp_path):
    """Two agents where the first hands off to the terminal agent loads cleanly."""
    data = {
        "name": "two-step",
        "start": "step1",
        "agents": {
            "step1": {"role": "First step", "handoff_to": "step2"},
            "step2": {"role": "Second step", "terminal": True},
        },
    }
    p = write_yaml(tmp_path, data)
    cfg = load_pipeline(p)
    assert cfg.agents["step1"].handoff_to == "step2"
    assert cfg.agents["step2"].terminal is True


def test_default_budget_cap_is_0_25(tmp_path):
    """When no budget section is provided, max_usd_per_run defaults to 0.25."""
    p = write_yaml(tmp_path, minimal_pipeline())
    cfg = load_pipeline(p)
    assert cfg.budget.max_usd_per_run == pytest.approx(0.25)


def test_default_permission_mode_is_prompt(tmp_path):
    """When no permissions section is provided, mode defaults to 'prompt'."""
    p = write_yaml(tmp_path, minimal_pipeline())
    cfg = load_pipeline(p)
    assert cfg.permissions.mode == PermissionMode.prompt


def test_budget_defaults_max_total_iterations_30(tmp_path):
    """Default total iteration cap is 30."""
    p = write_yaml(tmp_path, minimal_pipeline())
    cfg = load_pipeline(p)
    assert cfg.budget.max_total_iterations == 30


def test_explicit_budget_and_permissions_honored(tmp_path):
    """Explicitly set budget cap and permission mode are preserved."""
    data = minimal_pipeline()
    data["budget"] = {"max_usd_per_run": 1.5}
    data["permissions"] = {"mode": "auto"}
    p = write_yaml(tmp_path, data)
    cfg = load_pipeline(p)
    assert cfg.budget.max_usd_per_run == pytest.approx(1.5)
    assert cfg.permissions.mode == PermissionMode.auto


def test_agent_max_iterations_default_is_10(tmp_path):
    """Agent max_iterations defaults to 10 when not specified."""
    p = write_yaml(tmp_path, minimal_pipeline())
    cfg = load_pipeline(p)
    assert cfg.agents["worker"].max_iterations == 10


# ---------------------------------------------------------------------------
# Error: missing / unknown start agent
# ---------------------------------------------------------------------------


def test_missing_start_agent_raises_config_error(tmp_path):
    """ConfigError with a non-empty message when 'start' names an undefined agent."""
    data = {
        "name": "bad",
        "start": "does_not_exist",
        "agents": {
            "worker": {"role": "Worker", "terminal": True},
        },
    }
    p = write_yaml(tmp_path, data)
    with pytest.raises(ConfigError) as exc_info:
        load_pipeline(p)
    msg = str(exc_info.value)
    assert msg, "ConfigError message must not be empty"
    assert "does_not_exist" in msg


# ---------------------------------------------------------------------------
# Error: handoff_to unknown agent
# ---------------------------------------------------------------------------


def test_handoff_to_unknown_agent_raises_config_error(tmp_path):
    """ConfigError when an agent's handoff_to names an agent that doesn't exist."""
    data = {
        "name": "bad-handoff",
        "start": "a",
        "agents": {
            "a": {"role": "Agent A", "handoff_to": "ghost"},
        },
    }
    p = write_yaml(tmp_path, data)
    with pytest.raises(ConfigError) as exc_info:
        load_pipeline(p)
    msg = str(exc_info.value)
    assert msg
    assert "ghost" in msg


# ---------------------------------------------------------------------------
# Error: agent both terminal and handoff_to
# ---------------------------------------------------------------------------


def test_terminal_and_handoff_to_raises_config_error(tmp_path):
    """ConfigError when an agent is both terminal:true and has a handoff_to."""
    data = {
        "name": "bad-both",
        "start": "a",
        "agents": {
            "a": {"role": "Agent A", "terminal": True, "handoff_to": "b"},
            "b": {"role": "Agent B", "terminal": True},
        },
    }
    p = write_yaml(tmp_path, data)
    with pytest.raises(ConfigError) as exc_info:
        load_pipeline(p)
    msg = str(exc_info.value)
    assert msg
    assert "terminal" in msg.lower() or "handoff" in msg.lower()


# ---------------------------------------------------------------------------
# Error: all-agents-handoff cycle with no terminal
# ---------------------------------------------------------------------------


def test_all_agents_handoff_no_terminal_raises_config_error(tmp_path):
    """ConfigError when every agent has handoff_to but no agent is terminal."""
    data = {
        "name": "cycle",
        "start": "a",
        "agents": {
            "a": {"role": "Agent A", "handoff_to": "b"},
            "b": {"role": "Agent B", "handoff_to": "a"},
        },
    }
    p = write_yaml(tmp_path, data)
    with pytest.raises(ConfigError) as exc_info:
        load_pipeline(p)
    msg = str(exc_info.value)
    assert msg
    assert "terminal" in msg.lower()


# ---------------------------------------------------------------------------
# Error: unknown top-level key (extra=forbid)
# ---------------------------------------------------------------------------


def test_unknown_top_level_key_raises_config_error(tmp_path):
    """ConfigError when YAML contains a key not in the PipelineConfig schema."""
    data = minimal_pipeline()
    data["totally_unknown_field"] = "oops"
    p = write_yaml(tmp_path, data)
    with pytest.raises(ConfigError) as exc_info:
        load_pipeline(p)
    msg = str(exc_info.value)
    assert msg


# ---------------------------------------------------------------------------
# Error: empty agents dict
# ---------------------------------------------------------------------------


def test_empty_agents_dict_raises_config_error(tmp_path):
    """ConfigError when the agents map is empty."""
    data = {
        "name": "empty",
        "start": "nobody",
        "agents": {},
    }
    p = write_yaml(tmp_path, data)
    with pytest.raises(ConfigError) as exc_info:
        load_pipeline(p)
    msg = str(exc_info.value)
    assert msg
    assert "agent" in msg.lower()


# ---------------------------------------------------------------------------
# Error: file not found
# ---------------------------------------------------------------------------


def test_missing_file_raises_config_error(tmp_path):
    """ConfigError when the pipeline file path does not exist."""
    with pytest.raises(ConfigError) as exc_info:
        load_pipeline(tmp_path / "no_such_file.yaml")
    assert "not found" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# Error: invalid YAML syntax
# ---------------------------------------------------------------------------


def test_invalid_yaml_raises_config_error(tmp_path):
    """ConfigError when the file contains malformed YAML."""
    p = tmp_path / "bad.yaml"
    p.write_text("name: [unclosed bracket\n  - badly\n", encoding="utf-8")
    with pytest.raises(ConfigError) as exc_info:
        load_pipeline(p)
    assert "yaml" in str(exc_info.value).lower() or "invalid" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# Error: unknown agent-level key
# ---------------------------------------------------------------------------


def test_unknown_agent_key_raises_config_error(tmp_path):
    """ConfigError when an agent dict has an unrecognised key (extra=forbid)."""
    data = minimal_pipeline()
    data["agents"]["worker"]["nonexistent_key"] = "value"
    p = write_yaml(tmp_path, data)
    with pytest.raises(ConfigError) as exc_info:
        load_pipeline(p)
    assert str(exc_info.value)
