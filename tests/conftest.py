"""Shared test fixtures and the scripted fake LLM client.

Agents are non-deterministic, so CI never calls a real model. ``FakeLLMClient``
returns a canned sequence of :class:`LLMResponse` objects (or callables that
build one from the incoming messages), letting tests assert exact orchestration
behavior — handoffs, budget aborts, permission gating, loop caps, truncation —
with zero network and zero cost.

``RoutingFakeLLMClient`` extends this for parallel DAG tests: it routes replies
by agent name (parsed from the system prompt), is fully thread-safe, and
optionally accepts a ``threading.Barrier`` to prove two agents run concurrently.
"""

from __future__ import annotations

import re
import threading
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


# --- RoutingFakeLLMClient --------------------------------------------------- #

# Pattern matches the system prompt preamble written by agent.py's SYSTEM_TEMPLATE:
#   "You are the **{name}** agent"
_AGENT_NAME_RE = re.compile(r"the \*\*(\w+)\*\* agent")


class RoutingFakeLLMClient(LLMClient):
    """A thread-safe, agent-routed fake LLM client for parallel DAG tests.

    ``responses`` maps agent name -> list of reply strings. Each call for a
    given agent pops the next reply from its list (falls back to "done" when
    exhausted). Routing is based on the agent name embedded in the system prompt
    by :data:`agentforge.agent.SYSTEM_TEMPLATE`.

    An optional ``threading.Barrier`` can be supplied together with a tuple of
    ``barrier_agents``. The first call from any agent in ``barrier_agents`` will
    block on ``barrier.wait()`` *outside* the internal lock, so two agents can
    meet at the barrier simultaneously — the canonical proof that they are
    running concurrently.

    Call counts per agent are recorded on ``self.calls_by_agent`` (dict[str,
    int]) for post-run assertions.

    Parameters
    ----------
    responses:
        ``{agent_name: [reply, reply, ...]}`` — consumed FIFO per agent.
    supports_native_tools:
        Forwarded to the base class attribute (default ``True``).
    model:
        Model slug reported in each :class:`~agentforge.messages.LLMResponse`.
    barrier:
        Optional :class:`threading.Barrier`. When set, each agent listed in
        ``barrier_agents`` calls ``barrier.wait()`` on its *first* invocation.
    barrier_agents:
        Names of agents that should rendezvous at the barrier.
    big_usage:
        When ``True`` every response reports ``Usage(3000, 3000)`` instead of
        the default ``Usage(20, 10)``. Useful for budget-cancel tests.
    """

    provider = "fake"

    def __init__(
        self,
        responses: dict[str, list[str]],
        *,
        supports_native_tools: bool = True,
        model: str = "fake/model",
        barrier: Optional[threading.Barrier] = None,
        barrier_agents: tuple[str, ...] = (),
        big_usage: bool = False,
    ) -> None:
        super().__init__(model)
        self.supports_native_tools = supports_native_tools
        # Deep-copy the reply lists so the caller's dict is not mutated.
        self._responses: dict[str, list[str]] = {k: list(v) for k, v in responses.items()}
        self._barrier = barrier
        self._barrier_agents: frozenset[str] = frozenset(barrier_agents)
        self._big_usage = big_usage

        # Mutable state guarded by _lock.
        self._lock = threading.Lock()
        self.calls_by_agent: dict[str, int] = {}  # public: per-agent call count

    def complete(
        self,
        messages: list[Message],
        *,
        tools=None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        # --- parse agent name from system prompt -------------------------- #
        system_text = messages[0].content if messages else ""
        m = _AGENT_NAME_RE.search(system_text)
        agent_name = m.group(1) if m else "__unknown__"

        # --- thread-safe state update + reply selection ------------------- #
        hit_barrier = False
        with self._lock:
            self.calls_by_agent[agent_name] = self.calls_by_agent.get(agent_name, 0) + 1
            call_number = self.calls_by_agent[agent_name]

            queue = self._responses.get(agent_name, [])
            if queue:
                reply = queue.pop(0)
            else:
                reply = "done"

            # Decide whether this call should wait at the barrier.  We check
            # outside the lock branch below, but we need to know now (while
            # holding the lock) whether it's the first call for this agent.
            if (
                self._barrier is not None
                and agent_name in self._barrier_agents
                and call_number == 1
            ):
                hit_barrier = True

        # --- barrier rendezvous (outside the lock so both threads can meet) #
        if hit_barrier:
            self._barrier.wait()

        # --- build response ----------------------------------------------- #
        usage = Usage(3000, 3000) if self._big_usage else Usage(20, 10)
        return LLMResponse(
            text=reply,
            tool_calls=[],
            usage=usage,
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
