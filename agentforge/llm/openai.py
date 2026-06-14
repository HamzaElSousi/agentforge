"""OpenAI direct LLM client (chat completions endpoint).

Shares the OpenAI wire-format helpers with the OpenRouter client via the
:mod:`agentforge.llm._openai_compat` module. The only differences are the
default endpoint URL and the ``Authorization`` header value.
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

_DEFAULT_BASE_URL = "https://api.openai.com/v1/chat/completions"

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in _RETRYABLE_STATUS
    if isinstance(exc, httpx.TransportError):
        return True
    return False


class OpenAIClient(LLMClient):
    """LLM client that talks directly to OpenAI's chat completions endpoint.

    Parameters
    ----------
    model:
        OpenAI model slug (e.g. ``gpt-4o``, ``gpt-4o-mini``).
    api_key:
        OpenAI API key (``OPENAI_API_KEY``). Never logged.
    base_url:
        Override the endpoint URL (useful for Azure OpenAI or compatible
        self-hosted proxies).
    timeout:
        Per-request timeout in seconds (default: 120).
    """

    provider: str = "openai"
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
        """Run one chat completion against OpenAI and return a normalized
        :class:`~agentforge.messages.LLMResponse`.
        """
        resolved_model = model or self.model
        payload: dict = {
            "model": resolved_model,
            "messages": messages_to_openai(messages),
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if tools:
            payload["tools"] = tools_to_openai(tools)
            payload["tool_choice"] = "auto"

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
            headers: dict[str, str] = {"Content-Type": "application/json"}
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"
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
                        f"OpenAI error {exc.response.status_code}: {body}"
                    ) from exc
                raise

        return _do_call()

    def close(self) -> None:  # pragma: no cover
        """No persistent resources to release for this client."""
