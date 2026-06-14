"""Ollama local LLM client (``/api/chat`` endpoint).

Speaks the Ollama chat API, which uses an OpenAI-ish message format with an
optional ``tools`` field. Tool support varies by model — this client sends
tools when provided and parses ``message.tool_calls`` when present, but
tolerates responses with no tool_calls so the agent's text-ReAct fallback
can take over transparently.

No API key is required. The retry policy only covers ``TransportError``
(lost connection to the local Ollama process) — there is no remote rate
limiting to worry about.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from agentforge.llm._openai_compat import messages_to_openai, tools_to_openai
from agentforge.llm.base import LLMClient, LLMError
from agentforge.messages import LLMResponse, Message, ToolCall, ToolSpec, Usage

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "http://localhost:11434/api/chat"


class OllamaClient(LLMClient):
    """LLM client that talks to a local Ollama instance.

    Parameters
    ----------
    model:
        Ollama model tag (e.g. ``llama3``, ``mistral``, ``qwen2.5:7b``).
    api_key:
        Unused — Ollama runs locally with no authentication.
    base_url:
        Override the Ollama endpoint (default: ``http://localhost:11434/api/chat``).
    timeout:
        Per-request timeout in seconds (default: 120). Local inference can be
        slow on CPU; tune per model.
    """

    provider: str = "ollama"
    # Ollama tool support depends on the loaded model. We optimistically set
    # True and parse tool_calls when present. The agent loop handles the case
    # where the model emits plain text instead of a structured tool call (the
    # text-ReAct parser kicks in automatically).
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
        """Run one chat completion against Ollama and return a normalized
        :class:`~agentforge.messages.LLMResponse`.
        """
        resolved_model = model or self.model

        payload: dict[str, Any] = {
            "model": resolved_model,
            "messages": messages_to_openai(messages),
            "stream": False,
            "options": {"temperature": temperature},
        }
        if max_tokens is not None:
            payload["options"]["num_predict"] = max_tokens
        if tools:
            payload["tools"] = tools_to_openai(tools)

        data = self._call_with_retry(payload)
        return _parse_response(data, model_override=resolved_model)

    def _call_with_retry(self, payload: dict) -> dict:
        """Execute the HTTP POST with tenacity retry on transport errors only."""

        @retry(
            retry=retry_if_exception_type(httpx.TransportError),
            stop=stop_after_attempt(4),
            wait=wait_exponential(multiplier=1, max=20),
            reraise=True,
        )
        def _do_call() -> dict:
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    resp = client.post(
                        self._endpoint,
                        json=payload,
                        headers={"Content-Type": "application/json"},
                    )
                    resp.raise_for_status()
                    return resp.json()
            except httpx.HTTPStatusError as exc:
                try:
                    body = exc.response.text[:2000]
                except Exception:
                    body = "<unreadable response>"
                raise LLMError(
                    f"Ollama error {exc.response.status_code}: {body}"
                ) from exc

        return _do_call()

    def close(self) -> None:  # pragma: no cover
        """No persistent resources to release for this client."""


# ---------------------------------------------------------------------------
# Private response parser
# ---------------------------------------------------------------------------


def _decode_ollama_arguments(raw: Any, call_id: str) -> dict[str, Any]:
    """Coerce Ollama tool-call arguments into a dict.

    Ollama may return the arguments as a pre-parsed dict, a JSON string,
    or occasionally a non-standard structure. Fall back gracefully.
    """
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            result = json.loads(raw)
            if isinstance(result, dict):
                return result
        except (json.JSONDecodeError, ValueError):
            pass
        logger.warning("Ollama tool call %s: could not decode arguments; keeping raw", call_id)
        return {"_raw": raw}
    return {"_raw": str(raw)}


def _parse_response(
    data: dict[str, Any],
    *,
    model_override: Optional[str] = None,
) -> LLMResponse:
    """Parse an Ollama ``/api/chat`` response into an
    :class:`~agentforge.messages.LLMResponse`.

    Tolerates responses with no ``tool_calls`` field — the agent loop's
    text-ReAct fallback handles those transparently.
    """
    message: dict[str, Any] = data.get("message", {})

    text: Optional[str] = message.get("content") or None

    tool_calls: list[ToolCall] = []
    raw_tool_calls = message.get("tool_calls") or []
    for raw_tc in raw_tool_calls:
        # Ollama uses {"function": {"name": ..., "arguments": {...}}}
        fn = raw_tc.get("function", {})
        # Ollama does not consistently return a tool-call id
        call_id = raw_tc.get("id", f"call_{len(tool_calls)}")
        name = fn.get("name", "")
        arguments = _decode_ollama_arguments(fn.get("arguments", {}), call_id)
        tool_calls.append(ToolCall(id=call_id, name=name, arguments=arguments))

    # Ollama usage fields
    usage = Usage(
        prompt_tokens=data.get("prompt_eval_count", 0),
        completion_tokens=data.get("eval_count", 0),
    )

    return LLMResponse(
        text=text,
        tool_calls=tool_calls,
        usage=usage,
        finish_reason=data.get("done_reason") or ("stop" if data.get("done") else None),
        model=model_override or data.get("model"),
        raw=data,
    )
