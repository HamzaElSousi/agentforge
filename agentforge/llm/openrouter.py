"""OpenRouter LLM client (OpenAI-compatible chat completions endpoint).

Uses ``httpx`` (sync) with ``tenacity`` retry/backoff for transient errors.
Translates the neutral :mod:`agentforge.messages` types to/from the OpenAI
wire format via the shared :mod:`agentforge.llm._openai_compat` helpers.
"""

from __future__ import annotations

import logging
from typing import Optional

import httpx
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from agentforge.llm._openai_compat import (
    messages_to_openai,
    parse_openai_response,
    tools_to_openai,
)
from agentforge.llm.base import LLMClient, LLMError
from agentforge.messages import LLMResponse, Message, ToolSpec

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

# Statuses that are worth retrying
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def _is_retryable(exc: BaseException) -> bool:
    """Return True for transient HTTP errors and transport failures."""
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in _RETRYABLE_STATUS
    if isinstance(exc, httpx.TransportError):
        return True
    return False


class OpenRouterClient(LLMClient):
    """LLM client that talks to OpenRouter's OpenAI-compatible endpoint.

    Parameters
    ----------
    model:
        Default OpenRouter model slug (e.g. ``deepseek/deepseek-v4-flash``).
    api_key:
        OpenRouter API key (``OPENROUTER_API_KEY``). Never logged.
    base_url:
        Override the endpoint URL (useful for testing or self-hosted proxies).
    timeout:
        Per-request timeout in seconds (default: 120).
    """

    provider: str = "openrouter"
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
        """Run one chat completion against OpenRouter and return a normalized
        :class:`~agentforge.messages.LLMResponse`.
        """
        resolved_model = model or self.model
        payload = _build_payload(
            messages=messages,
            tools=tools,
            model=resolved_model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        data = self._call_with_retry(payload)
        return parse_openai_response(data, model_override=resolved_model)

    def _call_with_retry(self, payload: dict) -> dict:
        """Execute the HTTP POST with tenacity retry for transient errors."""

        @retry(
            retry=retry_if_exception(_is_retryable),
            stop=stop_after_attempt(4),
            wait=wait_exponential(multiplier=1, max=20),
            reraise=True,
        )
        def _do_call() -> dict:
            headers = _build_headers(self._api_key)
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    resp = client.post(self._endpoint, json=payload, headers=headers)
                    resp.raise_for_status()
                    return resp.json()
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code not in _RETRYABLE_STATUS:
                    # Non-retryable 4xx — raise a clean LLMError with the body
                    body = _safe_response_text(exc.response)
                    raise LLMError(
                        f"OpenRouter error {exc.response.status_code}: {body}"
                    ) from exc
                raise  # retryable — let tenacity handle it

        return _do_call()

    def close(self) -> None:  # pragma: no cover
        """No persistent resources to release for this client."""


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _build_headers(api_key: Optional[str]) -> dict[str, str]:
    headers: dict[str, str] = {
        "Content-Type": "application/json",
        "HTTP-Referer": "AgentForge",
        "X-Title": "AgentForge",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _build_payload(
    *,
    messages: list[Message],
    tools: Optional[list[ToolSpec]],
    model: str,
    temperature: float,
    max_tokens: Optional[int],
) -> dict:
    payload: dict = {
        "model": model,
        "messages": messages_to_openai(messages),
        "temperature": temperature,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if tools:
        payload["tools"] = tools_to_openai(tools)
        payload["tool_choice"] = "auto"
    return payload


def _safe_response_text(response: httpx.Response) -> str:
    """Extract the response body as text without raising."""
    try:
        return response.text[:2000]
    except Exception:
        return "<unreadable response>"
