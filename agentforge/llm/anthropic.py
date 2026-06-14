"""Anthropic Messages API client.

Uses ``httpx`` (sync) directly against ``/v1/messages``. The Anthropic wire
format differs significantly from OpenAI: system prompts are a top-level
field, tool results are embedded as user-turn content blocks, and usage
counts are ``input_tokens``/``output_tokens``.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

import httpx
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from agentforge.llm.base import LLMClient, LLMError
from agentforge.messages import LLMResponse, Message, ToolCall, ToolSpec, Usage

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://api.anthropic.com/v1/messages"
_ANTHROPIC_VERSION = "2023-06-01"
_DEFAULT_MAX_TOKENS = 4096

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in _RETRYABLE_STATUS
    if isinstance(exc, httpx.TransportError):
        return True
    return False


class AnthropicClient(LLMClient):
    """LLM client that talks directly to Anthropic's Messages API.

    Parameters
    ----------
    model:
        Anthropic model slug (e.g. ``claude-opus-4-5``, ``claude-sonnet-4-5``).
    api_key:
        Anthropic API key (``ANTHROPIC_API_KEY``). Never logged.
    base_url:
        Override the endpoint URL.
    timeout:
        Per-request timeout in seconds (default: 120).
    """

    provider: str = "anthropic"
    supports_native_tools: bool = True

    def __init__(
        self,
        model: str,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 120.0,
    ) -> None:
        super().__init__(model, api_key=api_key, base_url=base_url, timeout=timeout)
        self._endpoint = base_url or _DEFAULT_BASE_URL

    def complete(
        self,
        messages: list[Message],
        *,
        tools: Optional[list[ToolSpec]] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """Run one chat completion against the Anthropic Messages API and return
        a normalized :class:`~agentforge.messages.LLMResponse`.

        ``max_tokens`` defaults to 4096 when not provided (Anthropic requires it).
        """
        resolved_model = model or self.model
        resolved_max_tokens = max_tokens if max_tokens is not None else _DEFAULT_MAX_TOKENS

        system_text, wire_messages = _translate_messages(messages)

        payload: dict[str, Any] = {
            "model": resolved_model,
            "max_tokens": resolved_max_tokens,
            "temperature": temperature,
            "messages": wire_messages,
        }
        if system_text:
            payload["system"] = system_text
        if tools:
            payload["tools"] = [spec.to_anthropic_format() for spec in tools]

        data = self._call_with_retry(payload)
        return _parse_response(data, model_override=resolved_model)

    def _call_with_retry(self, payload: dict) -> dict:
        """Execute the HTTP POST with tenacity retry for transient errors."""

        @retry(
            retry=retry_if_exception(_is_retryable),
            stop=stop_after_attempt(4),
            wait=wait_exponential(multiplier=1, max=20),
            reraise=True,
        )
        def _do_call() -> dict:
            headers: dict[str, str] = {
                "Content-Type": "application/json",
                "anthropic-version": _ANTHROPIC_VERSION,
            }
            if self._api_key:
                headers["x-api-key"] = self._api_key
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    resp = client.post(self._endpoint, json=payload, headers=headers)
                    resp.raise_for_status()
                    return resp.json()
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code not in _RETRYABLE_STATUS:
                    try:
                        body = exc.response.text[:2000]
                    except Exception:
                        body = "<unreadable response>"
                    raise LLMError(
                        f"Anthropic error {exc.response.status_code}: {body}"
                    ) from exc
                raise

        return _do_call()

    def close(self) -> None:  # pragma: no cover
        """No persistent resources to release for this client."""


# ---------------------------------------------------------------------------
# Private translation helpers
# ---------------------------------------------------------------------------


def _translate_messages(
    messages: list[Message],
) -> tuple[str, list[dict[str, Any]]]:
    """Translate neutral messages into (system_text, anthropic_messages).

    Anthropic separates the system prompt from the conversation. Consecutive
    system messages are concatenated with a blank line.

    Tool results from ``tool`` role messages are folded into a user-turn
    content block list (Anthropic's requirement: results must be ``user`` role
    with ``tool_result`` content type).
    """
    system_parts: list[str] = []
    wire: list[dict[str, Any]] = []

    i = 0
    while i < len(messages):
        msg = messages[i]

        if msg.role == "system":
            system_parts.append(msg.content)
            i += 1
            continue

        if msg.role == "user":
            wire.append({"role": "user", "content": msg.content})
            i += 1
            continue

        if msg.role == "assistant":
            content_blocks: list[dict[str, Any]] = []

            # Narration text (may accompany tool_use blocks)
            if msg.content:
                content_blocks.append({"type": "text", "text": msg.content})

            # Tool call blocks
            for tc in msg.tool_calls:
                content_blocks.append(
                    {
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": tc.arguments,
                    }
                )

            wire.append(
                {
                    "role": "assistant",
                    "content": content_blocks if content_blocks else msg.content,
                }
            )
            i += 1
            continue

        if msg.role == "tool":
            # Collect consecutive tool results and merge into one user turn
            tool_result_blocks: list[dict[str, Any]] = []
            while i < len(messages) and messages[i].role == "tool":
                t = messages[i]
                tool_result_blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": t.tool_call_id or "",
                        "content": t.content,
                    }
                )
                i += 1
            wire.append({"role": "user", "content": tool_result_blocks})
            continue

        # Unknown role — pass through as user content so we don't drop data
        wire.append({"role": "user", "content": msg.content})
        i += 1

    return "\n\n".join(system_parts), wire


def _parse_response(
    data: dict[str, Any],
    *,
    model_override: Optional[str] = None,
) -> LLMResponse:
    """Parse an Anthropic Messages API response into an
    :class:`~agentforge.messages.LLMResponse`.
    """
    content_blocks: list[dict[str, Any]] = data.get("content", [])

    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []

    for block in content_blocks:
        btype = block.get("type")
        if btype == "text":
            text_parts.append(block.get("text", ""))
        elif btype == "tool_use":
            # Anthropic gives us a pre-parsed dict in ``input``
            arguments = block.get("input", {})
            if not isinstance(arguments, dict):
                # Defensive: shouldn't happen but guard anyway
                arguments = {"_raw": str(arguments)}
            tool_calls.append(
                ToolCall(
                    id=block.get("id", ""),
                    name=block.get("name", ""),
                    arguments=arguments,
                )
            )

    text: Optional[str] = "\n".join(text_parts) if text_parts else None

    raw_usage = data.get("usage", {})
    usage = Usage(
        prompt_tokens=raw_usage.get("input_tokens", 0),
        completion_tokens=raw_usage.get("output_tokens", 0),
    )

    return LLMResponse(
        text=text,
        tool_calls=tool_calls,
        usage=usage,
        finish_reason=data.get("stop_reason"),
        model=model_override or data.get("model"),
        raw=data,
    )
