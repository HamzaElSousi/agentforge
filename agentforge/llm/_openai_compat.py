"""Shared helpers for providers that speak the OpenAI chat-completion wire format.

Both :mod:`agentforge.llm.openrouter` and :mod:`agentforge.llm.openai` import
from here so the translation and parse logic lives in exactly one place.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from agentforge.messages import LLMResponse, Message, ToolCall, ToolSpec, Usage

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request translation: agentforge.messages → OpenAI wire format
# ---------------------------------------------------------------------------


def messages_to_openai(messages: list[Message]) -> list[dict[str, Any]]:
    """Translate a list of neutral :class:`~agentforge.messages.Message` objects
    into the OpenAI ``messages`` array format.

    Handles all four roles:

    - ``system``    → ``{"role": "system", "content": "..."}``
    - ``user``      → ``{"role": "user",   "content": "..."}``
    - ``assistant`` with tool calls → ``{"role": "assistant", "tool_calls": [...]}``
    - ``assistant`` text only       → ``{"role": "assistant", "content": "..."}``
    - ``tool``      → ``{"role": "tool", "tool_call_id": ..., "content": ..., "name": ...}``
    """
    out: list[dict[str, Any]] = []

    for msg in messages:
        if msg.role == "system":
            out.append({"role": "system", "content": msg.content})

        elif msg.role == "user":
            out.append({"role": "user", "content": msg.content})

        elif msg.role == "assistant":
            if msg.tool_calls:
                wire_tool_calls = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            # OpenAI expects a JSON string here
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in msg.tool_calls
                ]
                entry: dict[str, Any] = {
                    "role": "assistant",
                    "tool_calls": wire_tool_calls,
                }
                # Some models also emit narration alongside tool calls
                if msg.content:
                    entry["content"] = msg.content
                out.append(entry)
            else:
                out.append({"role": "assistant", "content": msg.content})

        elif msg.role == "tool":
            entry = {
                "role": "tool",
                "tool_call_id": msg.tool_call_id or "",
                "content": msg.content,
            }
            if msg.name:
                entry["name"] = msg.name
            out.append(entry)

    return out


def tools_to_openai(tools: list[ToolSpec]) -> list[dict[str, Any]]:
    """Convert a list of :class:`~agentforge.messages.ToolSpec` objects to the
    OpenAI ``tools`` array (each entry is ``{"type": "function", "function": {...}}``).
    """
    return [spec.to_openai_format() for spec in tools]


# ---------------------------------------------------------------------------
# Response parsing: OpenAI wire format → agentforge.messages
# ---------------------------------------------------------------------------


def _decode_arguments(raw: str, call_id: str) -> dict[str, Any]:
    """JSON-decode the ``function.arguments`` string from an OpenAI tool call.

    Falls back to ``{"_raw": raw}`` on malformed JSON so the agent loop can
    still inspect the raw string and decide what to do.
    """
    try:
        result = json.loads(raw)
        if not isinstance(result, dict):
            # Some models return a non-object JSON value (e.g. a bare string)
            return {"_raw": raw}
        return result
    except (json.JSONDecodeError, ValueError):
        logger.warning("tool call %s: could not decode arguments JSON; keeping raw", call_id)
        return {"_raw": raw}


def parse_openai_response(
    data: dict[str, Any],
    *,
    model_override: Optional[str] = None,
) -> LLMResponse:
    """Parse an OpenAI-style chat completion response dict into an
    :class:`~agentforge.messages.LLMResponse`.

    Parameters
    ----------
    data:
        The full JSON response body as a dict.
    model_override:
        When set, overrides the ``model`` field in the response (useful when
        the provider echoes a different slug than what was requested).
    """
    choice = data.get("choices", [{}])[0]
    message = choice.get("message", {})

    # --- text content ---
    text: Optional[str] = message.get("content") or None

    # --- tool calls ---
    tool_calls: list[ToolCall] = []
    raw_tool_calls = message.get("tool_calls") or []
    for raw_tc in raw_tool_calls:
        fn = raw_tc.get("function", {})
        call_id = raw_tc.get("id", "")
        name = fn.get("name", "")
        raw_args = fn.get("arguments", "")
        if isinstance(raw_args, dict):
            # Defensive: some providers pre-parse the dict
            arguments = raw_args
        else:
            arguments = _decode_arguments(raw_args, call_id)
        tool_calls.append(ToolCall(id=call_id, name=name, arguments=arguments))

    # --- usage ---
    raw_usage = data.get("usage", {})
    usage = Usage(
        prompt_tokens=raw_usage.get("prompt_tokens", 0),
        completion_tokens=raw_usage.get("completion_tokens", 0),
    )

    return LLMResponse(
        text=text,
        tool_calls=tool_calls,
        usage=usage,
        finish_reason=choice.get("finish_reason"),
        model=model_override or data.get("model"),
        raw=data,
    )
