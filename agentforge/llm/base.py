"""Abstract ``LLMClient`` — the one interface every provider implements.

The agent loop only ever talks to this interface, so swapping OpenRouter for
Anthropic, OpenAI, or local Ollama is a config change, not a code change
(Strategy pattern). Concrete clients translate the neutral
:mod:`agentforge.messages` types to/from their provider wire format.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from agentforge.messages import LLMResponse, Message, ToolSpec


class LLMError(RuntimeError):
    """Raised for non-retryable provider errors (bad request, auth, etc.)."""


class LLMClient(ABC):
    """Minimal interface a provider must satisfy.

    Parameters
    ----------
    model:
        Default model slug for this client. A per-call ``model`` argument to
        :meth:`complete` overrides it (used for per-agent model overrides).
    api_key:
        Provider credential. Never logged or written to traces.
    base_url:
        Override the provider endpoint (useful for self-hosted/proxy).
    timeout:
        Per-request timeout in seconds.
    """

    #: Whether this provider supports structured native tool/function calling.
    #: When False, the agent loop uses the text-ReAct fallback parser.
    supports_native_tools: bool = True

    #: Stable provider identifier (``openrouter`` / ``anthropic`` / ...).
    provider: str = "base"

    def __init__(
        self,
        model: str,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self._api_key = api_key
        self.base_url = base_url
        self.timeout = timeout

    @abstractmethod
    def complete(
        self,
        messages: list[Message],
        *,
        tools: Optional[list[ToolSpec]] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """Run one chat completion and return a normalized :class:`LLMResponse`.

        Implementations MUST:
        - return parsed ``ToolCall`` objects (decode JSON-string args), and
        - populate ``usage`` with prompt/completion tokens when the provider
          reports them (fall back to a token estimate otherwise).
        """
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover - optional resource cleanup
        """Release any underlying HTTP resources. Optional."""
