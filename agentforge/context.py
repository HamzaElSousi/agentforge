"""Context management: keep the conversation inside the model's window and the
budget under control.

Two mechanisms, both token-aware via ``tiktoken``:

- **Truncation** — a single huge tool output (a web page, a file dump) is capped
  to head + tail with a ``[... N chars omitted ...]`` marker before it ever
  enters history.
- **Trimming** — when the running history approaches a token ceiling, the
  oldest non-essential turns are dropped while the system prompt, the original
  goal, and the most recent turns are preserved.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

from agentforge.messages import Message

_DEFAULT_MAX_CHARS = 6000


@lru_cache(maxsize=8)
def _encoder(model: str):
    """Return a tiktoken encoder for ``model``; fall back to ``cl100k_base``.

    Most non-OpenAI slugs (DeepSeek, Qwen, GLM…) are unknown to tiktoken, so we
    fall back to a general-purpose encoder — good enough for budgeting/trimming
    where we only need a consistent, slightly-conservative estimate.
    """
    try:
        import tiktoken

        try:
            return tiktoken.encoding_for_model(model)
        except KeyError:
            return tiktoken.get_encoding("cl100k_base")
    except Exception:  # tiktoken missing or no network for its data file
        return None


def count_tokens(text: str, model: str = "gpt-4") -> int:
    """Estimate the token count of ``text``. Heuristic fallback ≈ 4 chars/token."""
    if not text:
        return 0
    enc = _encoder(model)
    if enc is None:
        return max(1, len(text) // 4)
    return len(enc.encode(text))


def count_message_tokens(messages: list[Message], model: str = "gpt-4") -> int:
    """Approximate total tokens for a message list, including a small per-message
    and per-tool-call overhead to stay on the conservative side."""
    total = 0
    for m in messages:
        total += count_tokens(m.content or "", model) + 4
        for tc in m.tool_calls:
            total += count_tokens(tc.name, model)
            total += count_tokens(str(tc.arguments), model) + 4
    return total


def truncate_output(text: str, limit: int = _DEFAULT_MAX_CHARS) -> str:
    """Cap ``text`` to ``limit`` chars as head + tail with an omission marker."""
    if text is None:
        return ""
    if len(text) <= limit:
        return text
    head = limit * 2 // 3
    tail = limit - head
    omitted = len(text) - head - tail
    return f"{text[:head]}\n\n[... {omitted} chars omitted ...]\n\n{text[-tail:]}"


def trim_history(
    messages: list[Message],
    *,
    max_tokens: int,
    model: str = "gpt-4",
    keep_recent: int = 6,
) -> tuple[list[Message], int]:
    """Drop oldest middle turns until the history fits under ``max_tokens``.

    Always preserved:
    - all leading ``system`` messages (role + goal + injection-containment),
    - the first ``user`` message (the goal), and
    - the last ``keep_recent`` messages (the active reasoning).

    Returns ``(trimmed_messages, dropped_count)``. A dropped span is replaced by
    a single system breadcrumb so the model knows elision happened. Tool result
    messages are never separated from the assistant turn that requested them at
    the recent boundary (we keep whole pairs by keeping the tail intact).
    """
    if count_message_tokens(messages, model) <= max_tokens:
        return messages, 0

    # Partition: leading system block, then the body, with a protected tail.
    n = len(messages)
    head_end = 0
    while head_end < n and messages[head_end].role == "system":
        head_end += 1
    # Keep the first user (goal) message inside the protected head if present.
    if head_end < n and messages[head_end].role == "user":
        head_end += 1

    protected_head = messages[:head_end]
    tail_start = max(head_end, n - keep_recent)
    protected_tail = messages[tail_start:]
    middle = messages[head_end:tail_start]

    dropped = 0
    # Drop from the oldest end of the middle until we fit (or middle is empty).
    while middle and count_message_tokens(
        protected_head + middle + protected_tail, model
    ) > max_tokens:
        middle.pop(0)
        dropped += 1

    result = list(protected_head)
    if dropped:
        result.append(
            Message(role="system", content=f"[{dropped} earlier turn(s) elided to fit context]")
        )
    result.extend(middle)
    result.extend(protected_tail)
    return result, dropped
