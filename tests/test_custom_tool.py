"""Tests for agentforge/tools/registry.py — @tool decorator and schema generation.

This file proves the PRD '<10 lines' criterion: a typed Python function
decorated with @tool is automatically registered with a correct JSON schema,
with ctx excluded from the model-visible parameters.

Each test uses a fresh ToolRegistry to avoid polluting the global REGISTRY.
"""

from __future__ import annotations

from typing import Optional

import pytest

from agentforge.messages import ToolSpec
from agentforge.tools.registry import (
    REGISTRY,
    Tool,
    ToolContext,
    ToolRegistry,
    _build_schema,
    tool,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def fresh_registry() -> ToolRegistry:
    """A fresh per-test ToolRegistry that does not affect REGISTRY."""
    return ToolRegistry()


# ---------------------------------------------------------------------------
# PRD '<10 lines' criterion
# ---------------------------------------------------------------------------


def test_tool_registered_in_under_10_lines(fresh_registry: ToolRegistry):
    """Defining and registering a tool takes fewer than 10 lines of code."""

    @tool(registry=fresh_registry)
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    # The tool is now registered — that's it. No boilerplate, no framework edits.
    assert "add" in fresh_registry


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TestToolRegistration:
    def test_tool_registered_by_function_name(self, fresh_registry: ToolRegistry):
        @tool(registry=fresh_registry)
        def my_func(x: int) -> int:
            """My function."""
            return x

        assert "my_func" in fresh_registry

    def test_tool_registered_with_custom_name(self, fresh_registry: ToolRegistry):
        @tool(name="custom_name", registry=fresh_registry)
        def internal_name(x: int) -> int:
            """Custom name."""
            return x

        assert "custom_name" in fresh_registry
        assert "internal_name" not in fresh_registry

    def test_get_returns_tool_object(self, fresh_registry: ToolRegistry):
        @tool(registry=fresh_registry)
        def add(a: int, b: int) -> int:
            """Add."""
            return a + b

        t = fresh_registry.get("add")
        assert t is not None
        assert isinstance(t, Tool)

    def test_get_unknown_tool_returns_none(self, fresh_registry: ToolRegistry):
        result = fresh_registry.get("nonexistent_tool")
        assert result is None

    def test_tool_names_listed(self, fresh_registry: ToolRegistry):
        @tool(registry=fresh_registry)
        def tool_a(x: int) -> int:
            """A."""
            return x

        @tool(registry=fresh_registry)
        def tool_b(x: int) -> int:
            """B."""
            return x

        names = fresh_registry.names()
        assert "tool_a" in names
        assert "tool_b" in names

    def test_subset_returns_tools_in_order(self, fresh_registry: ToolRegistry):
        @tool(registry=fresh_registry)
        def f1(x: int) -> int:
            """F1."""
            return x

        @tool(registry=fresh_registry)
        def f2(x: int) -> int:
            """F2."""
            return x

        subset = fresh_registry.subset(["f2", "f1"])
        assert [t.name for t in subset] == ["f2", "f1"]

    def test_subset_unknown_tool_raises_key_error(self, fresh_registry: ToolRegistry):
        with pytest.raises(KeyError, match="ghost"):
            fresh_registry.subset(["ghost"])


# ---------------------------------------------------------------------------
# JSON schema generation
# ---------------------------------------------------------------------------


class TestSchemaGeneration:
    def test_integer_parameters_have_integer_type(self, fresh_registry: ToolRegistry):
        @tool(registry=fresh_registry)
        def add(a: int, b: int) -> int:
            """Add two integers."""
            return a + b

        t = fresh_registry.get("add")
        props = t.parameters["properties"]
        assert props["a"] == {"type": "integer"}
        assert props["b"] == {"type": "integer"}

    def test_both_params_required_when_no_defaults(self, fresh_registry: ToolRegistry):
        @tool(registry=fresh_registry)
        def add(a: int, b: int) -> int:
            """Add."""
            return a + b

        t = fresh_registry.get("add")
        assert "required" in t.parameters
        assert set(t.parameters["required"]) == {"a", "b"}

    def test_ctx_not_in_schema_properties(self, fresh_registry: ToolRegistry):
        """ToolContext is injected at runtime and must be excluded from the schema."""

        @tool(registry=fresh_registry)
        def ctx_tool(ctx: ToolContext, path: str) -> str:
            """A tool with ctx."""
            return path

        t = fresh_registry.get("ctx_tool")
        props = t.parameters["properties"]
        assert "ctx" not in props, "ctx must not appear in model-visible schema"
        assert "path" in props

    def test_ctx_not_in_required_list(self, fresh_registry: ToolRegistry):
        @tool(registry=fresh_registry)
        def ctx_tool(ctx: ToolContext, x: int) -> int:
            """Tool with ctx."""
            return x

        t = fresh_registry.get("ctx_tool")
        required = t.parameters.get("required", [])
        assert "ctx" not in required

    def test_string_parameter_type(self, fresh_registry: ToolRegistry):
        @tool(registry=fresh_registry)
        def greet(name: str) -> str:
            """Greet someone."""
            return f"Hello, {name}"

        t = fresh_registry.get("greet")
        assert t.parameters["properties"]["name"] == {"type": "string"}

    def test_float_parameter_type(self, fresh_registry: ToolRegistry):
        @tool(registry=fresh_registry)
        def scale(factor: float) -> float:
            """Scale."""
            return factor

        t = fresh_registry.get("scale")
        assert t.parameters["properties"]["factor"] == {"type": "number"}

    def test_bool_parameter_type(self, fresh_registry: ToolRegistry):
        @tool(registry=fresh_registry)
        def toggle(flag: bool) -> bool:
            """Toggle."""
            return not flag

        t = fresh_registry.get("toggle")
        assert t.parameters["properties"]["flag"] == {"type": "boolean"}

    def test_optional_parameter_not_in_required(self, fresh_registry: ToolRegistry):
        @tool(registry=fresh_registry)
        def with_default(x: int, y: int = 0) -> int:
            """With default."""
            return x + y

        t = fresh_registry.get("with_default")
        required = t.parameters.get("required", [])
        assert "x" in required
        assert "y" not in required

    def test_list_parameter_type(self, fresh_registry: ToolRegistry):
        @tool(registry=fresh_registry)
        def sum_list(values: list[int]) -> int:
            """Sum."""
            return sum(values)

        t = fresh_registry.get("sum_list")
        assert t.parameters["properties"]["values"]["type"] == "array"
        assert t.parameters["properties"]["values"]["items"] == {"type": "integer"}

    def test_parameters_schema_is_object_type(self, fresh_registry: ToolRegistry):
        @tool(registry=fresh_registry)
        def add(a: int, b: int) -> int:
            """Add."""
            return a + b

        t = fresh_registry.get("add")
        assert t.parameters["type"] == "object"


# ---------------------------------------------------------------------------
# Tool metadata
# ---------------------------------------------------------------------------


class TestToolMetadata:
    def test_description_from_docstring(self, fresh_registry: ToolRegistry):
        @tool(registry=fresh_registry)
        def documented(x: int) -> int:
            """This is the docstring description."""
            return x

        t = fresh_registry.get("documented")
        assert "docstring description" in t.description

    def test_custom_description_overrides_docstring(self, fresh_registry: ToolRegistry):
        @tool(description="Custom description.", registry=fresh_registry)
        def func(x: int) -> int:
            """Original docstring."""
            return x

        t = fresh_registry.get("func")
        assert t.description == "Custom description."

    def test_risk_level_stored(self, fresh_registry: ToolRegistry):
        @tool(risk="dangerous", registry=fresh_registry)
        def risky(x: int) -> int:
            """Risky."""
            return x

        t = fresh_registry.get("risky")
        assert t.risk == "dangerous"

    def test_default_risk_is_side_effecting(self, fresh_registry: ToolRegistry):
        @tool(registry=fresh_registry)
        def default_risk(x: int) -> int:
            """Default."""
            return x

        t = fresh_registry.get("default_risk")
        assert t.risk == "side_effecting"

    def test_needs_network_false_by_default(self, fresh_registry: ToolRegistry):
        @tool(registry=fresh_registry)
        def no_net(x: int) -> int:
            """No network."""
            return x

        t = fresh_registry.get("no_net")
        assert t.needs_network is False

    def test_needs_network_true_when_set(self, fresh_registry: ToolRegistry):
        @tool(needs_network=True, registry=fresh_registry)
        def net_tool(url: str) -> str:
            """Needs network."""
            return url

        t = fresh_registry.get("net_tool")
        assert t.needs_network is True

    def test_needs_context_true_when_ctx_param_present(self, fresh_registry: ToolRegistry):
        @tool(registry=fresh_registry)
        def ctx_aware(ctx: ToolContext, x: int) -> int:
            """Context aware."""
            return x

        t = fresh_registry.get("ctx_aware")
        assert t.needs_context is True

    def test_needs_context_false_when_no_ctx_param(self, fresh_registry: ToolRegistry):
        @tool(registry=fresh_registry)
        def no_ctx(x: int) -> int:
            """No context."""
            return x

        t = fresh_registry.get("no_ctx")
        assert t.needs_context is False


# ---------------------------------------------------------------------------
# Tool.spec() returns a ToolSpec
# ---------------------------------------------------------------------------


class TestToolSpec:
    def test_spec_returns_tool_spec_instance(self, fresh_registry: ToolRegistry):
        @tool(registry=fresh_registry)
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        t = fresh_registry.get("add")
        spec = t.spec()
        assert isinstance(spec, ToolSpec)

    def test_spec_name_matches_tool_name(self, fresh_registry: ToolRegistry):
        @tool(registry=fresh_registry)
        def add(a: int, b: int) -> int:
            """Add."""
            return a + b

        spec = fresh_registry.get("add").spec()
        assert spec.name == "add"

    def test_spec_description_matches_docstring(self, fresh_registry: ToolRegistry):
        @tool(registry=fresh_registry)
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        spec = fresh_registry.get("add").spec()
        assert "Add two numbers" in spec.description

    def test_spec_parameters_are_dict(self, fresh_registry: ToolRegistry):
        @tool(registry=fresh_registry)
        def add(a: int, b: int) -> int:
            """Add."""
            return a + b

        spec = fresh_registry.get("add").spec()
        assert isinstance(spec.parameters, dict)

    def test_spec_parameters_has_properties(self, fresh_registry: ToolRegistry):
        @tool(registry=fresh_registry)
        def add(a: int, b: int) -> int:
            """Add."""
            return a + b

        spec = fresh_registry.get("add").spec()
        assert "properties" in spec.parameters
        assert "a" in spec.parameters["properties"]
        assert "b" in spec.parameters["properties"]

    def test_spec_to_openai_format(self, fresh_registry: ToolRegistry):
        @tool(registry=fresh_registry)
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        spec = fresh_registry.get("add").spec()
        openai_fmt = spec.to_openai_format()
        assert openai_fmt["type"] == "function"
        assert "function" in openai_fmt
        assert openai_fmt["function"]["name"] == "add"

    def test_spec_to_anthropic_format(self, fresh_registry: ToolRegistry):
        @tool(registry=fresh_registry)
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        spec = fresh_registry.get("add").spec()
        anthropic_fmt = spec.to_anthropic_format()
        assert anthropic_fmt["name"] == "add"
        assert "input_schema" in anthropic_fmt


# ---------------------------------------------------------------------------
# Tool callable
# ---------------------------------------------------------------------------


class TestToolCallable:
    def test_tool_is_callable_and_returns_correct_result(self, fresh_registry: ToolRegistry):
        @tool(registry=fresh_registry)
        def add(a: int, b: int) -> int:
            """Add."""
            return a + b

        t = fresh_registry.get("add")
        assert t(2, 3) == 5

    def test_tool_with_ctx_callable(self, fresh_registry: ToolRegistry):
        @tool(registry=fresh_registry)
        def echo(ctx: ToolContext, value: str) -> str:
            """Echo."""
            return value

        t = fresh_registry.get("echo")
        ctx = ToolContext()
        assert t(ctx, "hello") == "hello"


# ---------------------------------------------------------------------------
# _build_schema standalone
# ---------------------------------------------------------------------------


class TestBuildSchema:
    def test_build_schema_excludes_ctx_parameter(self):
        def my_tool(ctx: ToolContext, path: str, content: str) -> str:
            return ""

        schema, needs_context = _build_schema(my_tool)
        assert needs_context is True
        assert "ctx" not in schema["properties"]
        assert "path" in schema["properties"]
        assert "content" in schema["properties"]

    def test_build_schema_no_ctx(self):
        def plain(x: int, y: str) -> str:
            return ""

        schema, needs_context = _build_schema(plain)
        assert needs_context is False
        assert "x" in schema["properties"]
        assert "y" in schema["properties"]
