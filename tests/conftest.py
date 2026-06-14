"""Shared test fixtures and the scripted fake LLM client.

Agents are non-deterministic, so CI never calls a real model. ``FakeLLMClient``
returns a canned sequence of :class:`LLMResponse` objects (or callables that
build one from the incoming messages), letting tests assert exact orchestration
behavior — handoffs, budget aborts, permission gating, loop caps, truncation —
with zero network and zero cost.
"""

from __future__ import annotations

from typing import Callable, Optional, Union

import pytest

from agentforge.llm.base import LLMClient
from agentforge.messages import LLMResponse, Message, ToolCall, ToolSpec, Usage

ScriptItem = Union[LLMResponse, Callable[[list[Message]], LLMResponse]]


class FakeLLMClient(LLMClient):
    """A scripted, deterministic LLM client for tests."""

    provider = "fake"

    def __init__(
        self,
        script: Optional[list[ScriptItem]] = None,
        *,
        supports_native_tools: bool = True,
        model: str = "fake/model",
        default: Optional[LLMResponse] = None,
    ) -> None:
        super().__init__(model)
        self.supports_native_tools = supports_native_tools
        self._script: list[ScriptItem] = list(script or [])
        self._default = default
        self.calls: list[list[Message]] = []
        self.tools_seen: list[Optional[list[ToolSpec]]] = []
        self.models_seen: list[Optional[str]] = []

    def complete(self, messages, *, tools=None, model=None, temperature=0.7, max_tokens=None):
        # Store copies for assertions.
        self.calls.append(list(messages))
        self.tools_seen.append(tools)
        self.models_seen.append(model)
        if self._script:
            item = self._script.pop(0)
            return item(messages) if callable(item) else item
        if self._default is not None:
            return self._default
        # Exhausted script with no default -> conclude with a final answer so
        # loops always terminate.
        return text_response("FINAL: script exhausted")


# --- response builders ------------------------------------------------------ #


def tool_response(
    name: str,
    arguments: dict,
    *,
    call_id: str = "call-1",
    text: str = "",
    prompt_tokens: int = 50,
    completion_tokens: int = 20,
) -> LLMResponse:
    return LLMResponse(
        text=text,
        tool_calls=[ToolCall(id=call_id, name=name, arguments=arguments)],
        usage=Usage(prompt_tokens, completion_tokens),
        finish_reason="tool_calls",
        model="fake/model",
    )


def text_response(
    text: str, *, prompt_tokens: int = 50, completion_tokens: int = 20
) -> LLMResponse:
    return LLMResponse(
        text=text,
        tool_calls=[],
        usage=Usage(prompt_tokens, completion_tokens),
        finish_reason="stop",
        model="fake/model",
    )


# --- fixtures --------------------------------------------------------------- #


@pytest.fixture
def workspace(tmp_path):
    """A clean per-test workspace directory."""
    d = tmp_path / "workspace"
    d.mkdir()
    return d
