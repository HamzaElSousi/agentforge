"""Shared conversational primitives used across the whole framework.

These are deliberately provider-neutral plain dataclasses (no Pydantic, no
provider SDK types) so the LLM layer, agent loop, tools, and trace writer all
speak the same language without import cycles. Each concrete ``LLMClient``
translates *to and from* these types at its boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class ToolCall:
    """A single tool invocation requested by the model.

    ``arguments`` is always a parsed dict — clients are responsible for
    decoding provider-specific JSON-string argument payloads before
    constructing a ``ToolCall``.
    """

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class Message:
    """One turn in the conversation history.

    - ``assistant`` messages may carry ``tool_calls``.
    - ``tool`` messages carry the result of a single call and reference it via
      ``tool_call_id`` plus the tool ``name``.
    """

    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: Optional[str] = None
    name: Optional[str] = None

    def __post_init__(self) -> None:
        if self.content is None:
            self.content = ""


@dataclass
class Usage:
    """Token accounting for a single LLM call (or a rolled-up sum)."""

    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
        )


@dataclass
class ToolSpec:
    """Provider-neutral description of a tool the model may call.

    ``parameters`` is a JSON-Schema object describing the arguments. Concrete
    clients format this into their provider's tool/function schema.
    """

    name: str
    description: str
    parameters: dict[str, Any]

    def to_openai_format(self) -> dict[str, Any]:
        """OpenAI / OpenRouter ``tools`` array entry."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def to_anthropic_format(self) -> dict[str, Any]:
        """Anthropic ``tools`` array entry."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }


@dataclass
class LLMResponse:
    """Normalized result of one LLM call.

    Exactly one of ``text`` / ``tool_calls`` is the "primary" payload, but both
    may be present (some models narrate then call a tool). The agent loop
    decides what to act on.
    """

    text: Optional[str] = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    finish_reason: Optional[str] = None
    model: Optional[str] = None
    raw: Any = None

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)
