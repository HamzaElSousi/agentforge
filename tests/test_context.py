"""Tests for agentforge/context.py — token counting, output truncation, history trimming.

truncate_output uses a head+tail strategy (not a simple prefix cut):
  head = limit * 2 // 3
  tail = limit - head
  omitted = len(text) - head - tail

So the result length is: head + len("\\n\\n[... N chars omitted ...]\\n\\n") + tail.
"""

from __future__ import annotations

import pytest

from agentforge.context import (
    count_message_tokens,
    count_tokens,
    truncate_output,
    trim_history,
)
from agentforge.messages import Message, ToolCall


# ---------------------------------------------------------------------------
# truncate_output
# ---------------------------------------------------------------------------


class TestTruncateOutput:
    def test_short_text_returned_unchanged(self):
        s = "hello world"
        assert truncate_output(s) == s

    def test_exact_limit_not_truncated(self):
        s = "x" * 6000
        assert truncate_output(s) == s

    def test_long_text_is_shortened(self):
        s = "A" * 100_000
        result = truncate_output(s, limit=6000)
        # Result should NOT equal the original (it was truncated)
        assert result != s

    def test_long_text_contains_omitted_marker(self):
        s = "B" * 10_000
        result = truncate_output(s, limit=6000)
        assert "omitted" in result, "Truncated output must contain 'omitted' marker"

    def test_omitted_marker_contains_correct_count(self):
        # limit=100; head=66, tail=34; omitted = 200 - 100 = 100
        s = "x" * 200
        result = truncate_output(s, limit=100)
        assert "100" in result, "Marker should report 100 chars omitted"

    def test_result_contains_head_chars(self):
        """First chunk of original text must appear at the beginning of result."""
        s = "HEAD" + "M" * 10_000 + "TAIL"
        result = truncate_output(s, limit=100)
        assert result.startswith("HEAD"), "Result must start with the head of original text"

    def test_result_contains_tail_chars(self):
        """Last chunk of original text must appear at the end of result."""
        s = "HEAD" + "M" * 10_000 + "TAIL"
        result = truncate_output(s, limit=100)
        assert result.endswith("TAIL"), "Result must end with the tail of original text"

    def test_none_input_returns_empty_string(self):
        """truncate_output(None) must return '' per the source code guard."""
        assert truncate_output(None) == ""  # type: ignore[arg-type]

    def test_empty_string_returned_unchanged(self):
        assert truncate_output("") == ""

    def test_custom_limit_honored(self):
        s = "x" * 500
        result = truncate_output(s, limit=50)
        assert "omitted" in result


# ---------------------------------------------------------------------------
# count_tokens
# ---------------------------------------------------------------------------


class TestCountTokens:
    def test_empty_string_returns_zero(self):
        assert count_tokens("") == 0

    def test_non_empty_string_returns_positive(self):
        result = count_tokens("hello world")
        assert result > 0, "Non-empty text must have at least one token"

    def test_longer_text_has_more_tokens(self):
        short = count_tokens("hi")
        long = count_tokens("hi " * 100)
        assert long > short

    def test_single_word_positive(self):
        assert count_tokens("word") > 0

    def test_paragraph_positive(self):
        text = "The quick brown fox jumps over the lazy dog."
        assert count_tokens(text) > 0

    def test_whitespace_only_may_return_positive_or_zero(self):
        # Just verify no crash and returns int
        result = count_tokens("   ")
        assert isinstance(result, int)
        assert result >= 0


# ---------------------------------------------------------------------------
# count_message_tokens
# ---------------------------------------------------------------------------


class TestCountMessageTokens:
    def test_empty_list_returns_zero(self):
        assert count_message_tokens([]) == 0

    def test_single_system_message_positive(self):
        msgs = [Message(role="system", content="You are helpful.")]
        assert count_message_tokens(msgs) > 0

    def test_more_messages_more_tokens(self):
        short = [Message(role="user", content="Hi")]
        long = [Message(role="user", content="Hi " * 200)]
        assert count_message_tokens(long) > count_message_tokens(short)

    def test_message_with_tool_calls_counted(self):
        tc = ToolCall(id="call-1", name="my_tool", arguments={"key": "value"})
        msg = Message(role="assistant", content="", tool_calls=[tc])
        result = count_message_tokens([msg])
        assert result > 0, "Messages with tool calls must count tokens"

    def test_returns_integer(self):
        msgs = [Message(role="user", content="test")]
        assert isinstance(count_message_tokens(msgs), int)


# ---------------------------------------------------------------------------
# trim_history
# ---------------------------------------------------------------------------


class TestTrimHistory:
    def _make_history(self, n_middle_pairs: int = 20) -> list[Message]:
        """Build a long conversation that exceeds small token budgets."""
        msgs = [
            Message(role="system", content="You are a helpful assistant."),
            Message(role="user", content="Please write a comprehensive analysis of climate change effects."),
        ]
        for i in range(n_middle_pairs):
            msgs.append(Message(role="assistant", content="Analysis part " + ("A" * 200)))
            msgs.append(Message(role="user", content="Continue. " + ("B" * 200)))
        return msgs

    def test_short_history_returned_unchanged(self):
        msgs = [
            Message(role="system", content="sys"),
            Message(role="user", content="hi"),
        ]
        result, dropped = trim_history(msgs, max_tokens=10_000, model="gpt-4")
        assert dropped == 0
        assert result == msgs

    def test_long_history_triggers_trimming(self):
        msgs = self._make_history(n_middle_pairs=20)
        total_before = count_message_tokens(msgs)
        _, dropped = trim_history(msgs, max_tokens=200, model="gpt-4", keep_recent=4)
        assert dropped > 0, (
            f"Expected drops for {total_before} tokens against max_tokens=200"
        )

    def test_trimmed_result_fits_token_budget_or_drops_occurred(self):
        """After trim, either the result fits OR messages were dropped."""
        msgs = self._make_history(n_middle_pairs=20)
        result, dropped = trim_history(msgs, max_tokens=300, model="gpt-4", keep_recent=4)
        fits = count_message_tokens(result) <= 300
        # Either it fits OR something was dropped (or both)
        assert fits or dropped > 0, (
            "Trim must either produce a fitting history or report drops"
        )

    def test_system_message_always_preserved(self):
        msgs = self._make_history(n_middle_pairs=20)
        result, _ = trim_history(msgs, max_tokens=200, model="gpt-4", keep_recent=4)
        assert result[0].role == "system", "System message must always be first in trimmed result"
        assert "helpful assistant" in result[0].content

    def test_first_user_goal_preserved(self):
        msgs = self._make_history(n_middle_pairs=20)
        result, _ = trim_history(msgs, max_tokens=200, model="gpt-4", keep_recent=4)
        user_contents = [m.content for m in result if m.role == "user"]
        goal_text = "comprehensive analysis of climate change"
        assert any(goal_text in c for c in user_contents), (
            "The first user goal message must be preserved after trimming"
        )

    def test_recent_messages_preserved(self):
        """The last keep_recent messages must be in the trimmed result."""
        msgs = self._make_history(n_middle_pairs=20)
        last_content = msgs[-1].content
        result, _ = trim_history(msgs, max_tokens=200, model="gpt-4", keep_recent=4)
        result_contents = [m.content for m in result]
        assert last_content in result_contents, (
            "The most recent message must be preserved after trimming"
        )

    def test_dropped_count_is_non_negative_int(self):
        msgs = self._make_history(n_middle_pairs=5)
        _, dropped = trim_history(msgs, max_tokens=50, model="gpt-4")
        assert isinstance(dropped, int)
        assert dropped >= 0

    def test_elision_breadcrumb_inserted_when_drops_occur(self):
        """When turns are elided, a system breadcrumb should appear in the result."""
        msgs = self._make_history(n_middle_pairs=20)
        result, dropped = trim_history(msgs, max_tokens=200, model="gpt-4", keep_recent=4)
        if dropped > 0:
            system_contents = [m.content for m in result if m.role == "system"]
            assert any("elided" in c or "omitted" in c or str(dropped) in c
                       for c in system_contents), (
                "An elision breadcrumb system message must be inserted when turns are dropped"
            )

    def test_no_trimming_needed_returns_same_messages(self):
        msgs = [
            Message(role="system", content="sys"),
            Message(role="user", content="hi"),
            Message(role="assistant", content="hello"),
        ]
        result, dropped = trim_history(msgs, max_tokens=100_000, model="gpt-4")
        assert dropped == 0
        assert result == msgs
