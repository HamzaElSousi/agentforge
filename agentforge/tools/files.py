"""Workspace-jailed file and note tools.

All tools operate exclusively within the per-run workspace directory managed
by :class:`Workspace`. Any attempt to escape the jail (``../`` traversal,
absolute paths, symlinks that point outside) is rejected with a ``ValueError``
before any I/O occurs.

Notes are stored under ``<workspace>/.notes/<key>.txt`` so they are
persistent within a run but cleanly separated from user-created files.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from agentforge.tools.registry import ToolContext, tool


# ---------------------------------------------------------------------------
# Workspace jail
# ---------------------------------------------------------------------------


class Workspace:
    """A root-anchored directory that enforces path-traversal containment.

    All file access is funnelled through :meth:`resolve`, which normalises the
    path, checks symlinks, and raises :class:`ValueError` for any access that
    would escape the root.

    Parameters
    ----------
    root:
        Absolute path to the workspace root. Created if it does not exist.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Core jail logic
    # ------------------------------------------------------------------

    def resolve(self, relpath: str) -> Path:
        """Resolve *relpath* relative to the workspace root, rejecting traversal.

        Rules (all enforced before any I/O):
        1. Absolute paths are rejected — the model must not choose where files
           live on the host filesystem.
        2. After joining to the root, ``Path.resolve()`` canonicalises ``..``
           and resolves real symlinks; the result must still be inside root.
        3. Any component of the *raw* path that is a symlink and resolves to a
           location outside the root is also rejected.

        Parameters
        ----------
        relpath:
            A relative file path supplied by the LLM (e.g. ``"output.txt"`` or
            ``"subdir/notes.md"``).

        Returns
        -------
        Path
            Absolute, canonicalised path guaranteed to be inside the workspace.

        Raises
        ------
        ValueError
            If the path is absolute, traverses outside the root, or contains
            escaping symlinks. The error message deliberately avoids leaking
            the host root path beyond its basename.
        """
        # Rule 1: reject absolute paths.
        if Path(relpath).is_absolute():
            raise ValueError(
                f"Absolute paths are not allowed inside the sandbox workspace. "
                f"Use a relative path (got: {relpath!r})."
            )

        joined = (self.root / relpath).resolve()

        # Rule 2: resolved path must be inside root.
        try:
            joined.relative_to(self.root)
        except ValueError:
            raise ValueError(
                f"Path {relpath!r} resolves outside the workspace. "
                "Path traversal is not permitted."
            )

        # Rule 3: walk each component of the *unresolved* joined path and check
        # symlinks progressively — catches symlinks that are inside root but
        # point somewhere outside.
        raw_joined = self.root / relpath
        accumulated = self.root
        for part in Path(relpath).parts:
            accumulated = accumulated / part
            if accumulated.exists() and accumulated.is_symlink():
                target = accumulated.resolve()
                try:
                    target.relative_to(self.root)
                except ValueError:
                    raise ValueError(
                        f"Symlink in path {relpath!r} points outside the workspace. "
                        "Escaping symlinks are not permitted."
                    )

        return joined

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def notes_dir(self) -> Path:
        """Return (and create) the ``.notes`` directory inside the workspace."""
        d = self.root / ".notes"
        d.mkdir(exist_ok=True)
        return d


# ---------------------------------------------------------------------------
# File tools
# ---------------------------------------------------------------------------


@tool(risk="side_effecting")
def write_file(ctx: ToolContext, path: str, content: str) -> str:
    """Write *content* to a file at *path* inside the workspace.

    Creates parent directories as needed. Overwrites existing files.

    Parameters
    ----------
    path:
        Relative path within the workspace (e.g. ``"results/output.txt"``).
    content:
        The text content to write.

    Returns
    -------
    str
        A short confirmation message, or an error description if the operation
        is rejected (path traversal, permission error, etc.).
    """
    ws: Optional[Workspace] = ctx.workspace
    if ws is None:
        return "write_file is unavailable: no workspace configured."
    try:
        target = ws.resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"Written {len(content)} chars to '{path}'."
    except ValueError as exc:
        return f"[write_file error] {exc}"
    except OSError as exc:
        return f"[write_file error] Could not write '{path}': {exc.strerror}."


@tool(risk="read_only")
def read_file(ctx: ToolContext, path: str) -> str:
    """Read and return the contents of a file at *path* inside the workspace.

    Parameters
    ----------
    path:
        Relative path within the workspace (e.g. ``"results/output.txt"``).

    Returns
    -------
    str
        The file contents, or an error description if the file is missing,
        the path is rejected, or a read error occurs.
    """
    ws: Optional[Workspace] = ctx.workspace
    if ws is None:
        return "read_file is unavailable: no workspace configured."
    try:
        target = ws.resolve(path)
        if not target.exists():
            return f"[read_file error] File not found: '{path}'."
        if not target.is_file():
            return f"[read_file error] '{path}' is not a regular file."
        return target.read_text(encoding="utf-8", errors="replace")
    except ValueError as exc:
        return f"[read_file error] {exc}"
    except OSError as exc:
        return f"[read_file error] Could not read '{path}': {exc.strerror}."


# ---------------------------------------------------------------------------
# Note tools (persistent scratchpad within a run)
# ---------------------------------------------------------------------------


@tool(risk="side_effecting")
def save_note(ctx: ToolContext, key: str, content: str) -> str:
    """Save a named note to the workspace scratchpad.

    Notes are stored under ``<workspace>/.notes/<key>.txt``. The *key* must
    be a simple alphanumeric identifier (no path separators) and is itself
    validated against traversal.

    Parameters
    ----------
    key:
        Note identifier (e.g. ``"summary"`` or ``"step_1_output"``).
    content:
        The text content to store.

    Returns
    -------
    str
        Confirmation message, or an error description on failure.
    """
    ws: Optional[Workspace] = ctx.workspace
    if ws is None:
        return "save_note is unavailable: no workspace configured."
    try:
        # Validate key via the jail (treat as a relative path inside .notes).
        note_relpath = f".notes/{key}.txt"
        target = ws.resolve(note_relpath)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"Note '{key}' saved ({len(content)} chars)."
    except ValueError as exc:
        return f"[save_note error] {exc}"
    except OSError as exc:
        return f"[save_note error] Could not save note '{key}': {exc.strerror}."


@tool(risk="read_only")
def read_note(ctx: ToolContext, key: str) -> str:
    """Read a previously saved scratchpad note by *key*.

    Parameters
    ----------
    key:
        Note identifier matching a prior :func:`save_note` call.

    Returns
    -------
    str
        The note contents, or an error description if the note does not exist
        or the key is invalid.
    """
    ws: Optional[Workspace] = ctx.workspace
    if ws is None:
        return "read_note is unavailable: no workspace configured."
    try:
        note_relpath = f".notes/{key}.txt"
        target = ws.resolve(note_relpath)
        if not target.exists():
            return f"[read_note error] No note found for key '{key}'."
        return target.read_text(encoding="utf-8", errors="replace")
    except ValueError as exc:
        return f"[read_note error] {exc}"
    except OSError as exc:
        return f"[read_note error] Could not read note '{key}': {exc.strerror}."
