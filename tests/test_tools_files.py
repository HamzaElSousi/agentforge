"""Tests for agentforge/tools/files.py — Workspace jail and file/note tools.

Key source behavior verified before writing these tests:
- write_file / read_file / save_note / read_note return ERROR STRINGS (not raise)
  when a traversal is attempted.
- Workspace.resolve() DOES raise ValueError directly.
- save_note key '../evil' resolves to workspace root (stays inside) — NOT blocked.
- save_note key '../../etc/passwd' escapes workspace and IS blocked (returns error string).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agentforge.tools.files import Workspace, read_file, read_note, save_note, write_file
from agentforge.tools.registry import ToolContext


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def ws(tmp_path: Path) -> Workspace:
    """Fresh Workspace rooted at tmp_path/workspace."""
    root = tmp_path / "workspace"
    return Workspace(root)


@pytest.fixture()
def ctx(ws: Workspace) -> ToolContext:
    """ToolContext with the per-test workspace."""
    return ToolContext(workspace=ws)


# ---------------------------------------------------------------------------
# Workspace.resolve() — direct path jail
# ---------------------------------------------------------------------------


class TestWorkspaceResolve:
    def test_simple_relative_path_resolves_inside_root(self, ws: Workspace):
        resolved = ws.resolve("output.txt")
        assert resolved.is_relative_to(ws.root), (
            "A simple relative path should resolve inside the workspace root"
        )

    def test_subdirectory_path_resolves_inside_root(self, ws: Workspace):
        resolved = ws.resolve("subdir/notes.md")
        assert resolved.is_relative_to(ws.root)

    def test_absolute_path_raises_value_error(self, ws: Workspace):
        with pytest.raises(ValueError, match="(?i)absolute"):
            ws.resolve("/etc/passwd")

    def test_single_dotdot_traversal_raises_value_error(self, ws: Workspace):
        with pytest.raises(ValueError):
            ws.resolve("../escape.txt")

    def test_nested_dotdot_traversal_raises_value_error(self, ws: Workspace):
        with pytest.raises(ValueError):
            ws.resolve("../../etc/shadow")

    def test_dotdot_in_middle_of_path_raises_value_error(self, ws: Workspace):
        with pytest.raises(ValueError):
            ws.resolve("subdir/../../escape.txt")


# ---------------------------------------------------------------------------
# write_file / read_file round-trip
# ---------------------------------------------------------------------------


class TestWriteReadFile:
    def test_write_then_read_round_trip(self, ctx: ToolContext, ws: Workspace):
        result = write_file(ctx, "hello.txt", "Hello, world!")
        assert "Written" in result, f"Expected confirmation message, got: {result!r}"

        content = read_file(ctx, "hello.txt")
        assert content == "Hello, world!"

    def test_write_creates_file_under_workspace_root(self, ctx: ToolContext, ws: Workspace):
        write_file(ctx, "output.txt", "data")
        expected = ws.root / "output.txt"
        assert expected.exists(), "File must be created inside the workspace root"

    def test_write_does_not_create_file_outside_root_on_traversal(
        self, ctx: ToolContext, ws: Workspace, tmp_path: Path
    ):
        """A traversal attempt must not create any file above the workspace root."""
        write_file(ctx, "../escape.txt", "should not exist")
        # The file should NOT exist at the parent level
        assert not (tmp_path / "escape.txt").exists(), (
            "Traversal must not create files outside the workspace"
        )

    def test_write_creates_parent_directories(self, ctx: ToolContext, ws: Workspace):
        write_file(ctx, "deep/nested/dir/file.txt", "content")
        assert (ws.root / "deep" / "nested" / "dir" / "file.txt").exists()

    def test_write_overwrites_existing_file(self, ctx: ToolContext):
        write_file(ctx, "file.txt", "first")
        write_file(ctx, "file.txt", "second")
        assert read_file(ctx, "file.txt") == "second"

    def test_read_missing_file_returns_error_string(self, ctx: ToolContext):
        result = read_file(ctx, "does_not_exist.txt")
        assert result.startswith("[read_file error]"), (
            "Missing file must return an error string, not raise"
        )
        assert "not found" in result.lower() or "does_not_exist" in result

    def test_write_dotdot_traversal_returns_error_string(self, ctx: ToolContext):
        result = write_file(ctx, "../escape.txt", "bad")
        assert result.startswith("[write_file error]"), (
            "Path traversal must return an error string, not raise"
        )

    def test_write_absolute_path_returns_error_string(self, ctx: ToolContext):
        result = write_file(ctx, "/etc/passwd", "bad")
        assert result.startswith("[write_file error]"), (
            "Absolute path must return an error string, not raise"
        )

    def test_read_dotdot_traversal_returns_error_string(self, ctx: ToolContext):
        result = read_file(ctx, "../secret.txt")
        assert result.startswith("[read_file error]"), (
            "Read traversal must return an error string, not raise"
        )

    def test_read_absolute_path_returns_error_string(self, ctx: ToolContext):
        result = read_file(ctx, "/etc/passwd")
        assert result.startswith("[read_file error]"), (
            "Read with absolute path must return an error string, not raise"
        )

    def test_write_file_unicode_content(self, ctx: ToolContext):
        content = "こんにちは — Ünïcödé — emoji 🚀"
        write_file(ctx, "unicode.txt", content)
        assert read_file(ctx, "unicode.txt") == content

    def test_write_returns_confirmation_with_char_count(self, ctx: ToolContext):
        result = write_file(ctx, "count.txt", "12345")
        assert "5" in result, "Confirmation should mention the character count"

    def test_no_workspace_returns_unavailable_string(self):
        ctx_no_ws = ToolContext(workspace=None)
        result = write_file(ctx_no_ws, "file.txt", "data")
        assert "unavailable" in result.lower() or "no workspace" in result.lower()

        result2 = read_file(ctx_no_ws, "file.txt")
        assert "unavailable" in result2.lower() or "no workspace" in result2.lower()


# ---------------------------------------------------------------------------
# Nested traversal does NOT escape (files stay inside workspace)
# ---------------------------------------------------------------------------


def test_write_nested_escape_does_not_create_file_outside_root(
    ctx: ToolContext, ws: Workspace, tmp_path: Path
):
    """../../ traversal attempt via write_file must not place a file outside workspace."""
    write_file(ctx, "../../outside.txt", "should not exist")
    assert not (tmp_path.parent / "outside.txt").exists(), (
        "Deep traversal must be blocked and no file created outside workspace"
    )


# ---------------------------------------------------------------------------
# save_note / read_note round-trip
# ---------------------------------------------------------------------------


class TestNotes:
    def test_save_and_read_note_round_trip(self, ctx: ToolContext):
        save_note(ctx, "summary", "This is my summary.")
        result = read_note(ctx, "summary")
        assert result == "This is my summary."

    def test_save_note_creates_file_in_dot_notes_dir(self, ctx: ToolContext, ws: Workspace):
        save_note(ctx, "mykey", "value")
        note_file = ws.root / ".notes" / "mykey.txt"
        assert note_file.exists(), "Note file must live under .notes/ in the workspace"

    def test_save_note_returns_confirmation(self, ctx: ToolContext):
        result = save_note(ctx, "k", "abc")
        assert "k" in result or "saved" in result.lower()

    def test_read_missing_note_returns_error_string(self, ctx: ToolContext):
        result = read_note(ctx, "no_such_key")
        assert result.startswith("[read_note error]"), (
            "Missing note must return an error string, not raise"
        )

    def test_save_note_overwrite(self, ctx: ToolContext):
        save_note(ctx, "k", "first")
        save_note(ctx, "k", "second")
        assert read_note(ctx, "k") == "second"

    def test_save_note_deep_escape_key_returns_error_string(self, ctx: ToolContext):
        """A key like '../../etc/passwd' must not write outside the workspace."""
        result = save_note(ctx, "../../etc/passwd", "bad")
        assert result.startswith("[save_note error]"), (
            "Deep traversal via note key must return an error string"
        )

    def test_no_workspace_save_note_returns_unavailable(self):
        ctx_no_ws = ToolContext(workspace=None)
        result = save_note(ctx_no_ws, "k", "v")
        assert "unavailable" in result.lower() or "no workspace" in result.lower()

    def test_no_workspace_read_note_returns_unavailable(self):
        ctx_no_ws = ToolContext(workspace=None)
        result = read_note(ctx_no_ws, "k")
        assert "unavailable" in result.lower() or "no workspace" in result.lower()
