"""Human-in-the-loop permission layer for AgentForge.

Every tool call is classified by a risk policy derived from the pipeline
``PermissionsConfig`` and the tool's own ``risk`` field.  The result is one
of three outcomes recorded in ``PermissionDecision.outcome``:

- **approved** — call may proceed (possibly with edited args).
- **denied**   — call is blocked; the agent receives a denial message.
- **edited**   — user modified the arguments before approving.

Non-hang guarantee (PRD success criterion, and unit-tested)
-----------------------------------------------------------
When ``interactive=False`` the :meth:`PermissionManager.check` method
**MUST return immediately** without ever calling ``input()``, ``sys.stdin``
reads, or any blocking I/O.  The guard is enforced by the single branch
``if not self.interactive`` inside ``check`` which resolves the decision
entirely from policy data and returns before :meth:`_prompt` is ever
reached.  CI pipelines and automated runs are therefore safe — the process
will never hang waiting for a human that is not there.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.prompt import Prompt

from agentforge.config import NonInteractivePolicy, PermissionMode, PermissionsConfig
from agentforge.messages import ToolCall
from agentforge.tools.registry import Tool


# ---------------------------------------------------------------------------
# Decision enum
# ---------------------------------------------------------------------------


class Decision(str, Enum):
    """Outcome of a permission check — stored verbatim in the trace."""

    approved = "approved"
    denied = "denied"
    edited = "edited"


# ---------------------------------------------------------------------------
# PermissionDecision dataclass
# ---------------------------------------------------------------------------


@dataclass
class PermissionDecision:
    """The record appended to the trace for every permission check.

    Attributes
    ----------
    tool:
        Name of the tool that was checked.
    args:
        The arguments as originally presented by the model (before any edit).
    outcome:
        One of ``Decision.approved``, ``Decision.denied``, ``Decision.edited``.
    reason:
        Human-readable explanation — populated for denials and gated decisions.
        If the model supplied reasoning via the ``reasoning`` parameter, that
        text is forwarded here.
    edited_args:
        Present (non-None) only when ``outcome == Decision.edited``; holds the
        user-modified argument dict that will be passed to the tool instead.
    auto:
        ``True`` when the decision was resolved entirely by policy without
        asking a human (auto-approve, deny-by-policy, CI non-interactive).
        ``False`` only when a real human was prompted and responded.
    """

    tool: str
    args: dict[str, Any]
    outcome: Decision
    reason: str = ""
    edited_args: Optional[dict[str, Any]] = None
    auto: bool = False


# ---------------------------------------------------------------------------
# PermissionManager
# ---------------------------------------------------------------------------


class PermissionManager:
    """Classifies tool calls and gates them according to the pipeline policy.

    Parameters
    ----------
    cfg:
        The ``PermissionsConfig`` loaded from the pipeline YAML.
    interactive:
        Whether a real human TTY is available.  Callers should pass
        ``sys.stdin.isatty() and not assume_yes``.
    console:
        Optional ``rich.console.Console`` to write prompts to.  A fresh
        ``Console()`` is created if *None* is given.
    assume_yes:
        If ``True``, every gated call is auto-approved without prompting.
        Equivalent to the ``--yes`` / auto CI flag.

    Non-hang guarantee
    ------------------
    When ``interactive`` is ``False``, :meth:`check` resolves every decision
    synchronously from policy data and returns without invoking any I/O.  The
    internal :meth:`_prompt` helper is **never reached** in that branch.
    """

    def __init__(
        self,
        cfg: PermissionsConfig,
        *,
        interactive: bool,
        console: Optional[Console] = None,
        assume_yes: bool = False,
    ) -> None:
        self._cfg = cfg
        self.interactive = interactive
        self.assume_yes = assume_yes
        self._console = console or Console()

        # Precompute sets for O(1) membership tests.
        self._deny_set: frozenset[str] = frozenset(cfg.deny)
        self._auto_approve_set: frozenset[str] = frozenset(cfg.auto_approve)
        self._require_approval_set: frozenset[str] = frozenset(cfg.require_approval)

        # Session-scoped "always allow" set — starts empty each run.
        self._always_allow: set[str] = set()

        # Injectable ask callable — replaced by tests to avoid real TTY.
        # Signature: (prompt_text: str) -> str
        self._ask: Callable[[str], str] = Prompt.ask

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(self, tool: Tool) -> str:
        """Classify *tool* as ``"deny"``, ``"auto"``, or ``"gate"``.

        Classification truth table
        --------------------------

        ``deny`` always wins regardless of mode or explicit lists.
        After deny, explicit lists are checked, then mode defaults.

        +--------+-------------------+----------------------------+----------+
        | Mode   | Explicit list     | risk                       | Result   |
        +========+===================+============================+==========+
        | any    | deny list         | (any)                      | deny     |
        +--------+-------------------+----------------------------+----------+
        | any    | session always-   | (any)                      | auto     |
        |        | allow set         |                            |          |
        +--------+-------------------+----------------------------+----------+
        | any    | auto_approve list | (any)                      | auto     |
        +--------+-------------------+----------------------------+----------+
        | any    | require_approval  | (any)                      | gate     |
        |        | list              |                            |          |
        +--------+-------------------+----------------------------+----------+
        | auto   | (neither list)    | (any)                      | auto     |
        +--------+-------------------+----------------------------+----------+
        | prompt | (neither list)    | read_only                  | auto     |
        | prompt | (neither list)    | side_effecting / dangerous | gate     |
        +--------+-------------------+----------------------------+----------+
        | strict | (neither list)    | (any)                      | gate     |
        +--------+-------------------+----------------------------+----------+

        Priority ordering (highest to lowest):
        1. deny list  → "deny"
        2. session always-allow set → "auto"
        3. auto_approve list  → "auto"  (overrides mode, even strict)
        4. require_approval list → "gate"  (overrides mode)
        5. mode default

        Returns
        -------
        str
            One of ``"deny"``, ``"auto"``, ``"gate"``.
        """
        name = tool.name

        # 1. Deny always wins.
        if name in self._deny_set:
            return "deny"

        # 2. Session always-allow set (set by human during an interactive run).
        if name in self._always_allow:
            return "auto"

        # 3. Explicit auto_approve overrides mode (including strict).
        if name in self._auto_approve_set:
            return "auto"

        # 4. Explicit require_approval overrides mode default.
        if name in self._require_approval_set:
            return "gate"

        # 5. Mode default — no explicit list entry.
        mode = self._cfg.mode

        if mode is PermissionMode.auto:
            return "auto"

        if mode is PermissionMode.prompt:
            # read_only tools auto-run; anything more dangerous gates.
            if tool.risk == "read_only":
                return "auto"
            return "gate"

        # PermissionMode.strict — gate everything not explicitly auto_approved.
        return "gate"

    def check(
        self,
        tool: Tool,
        call: ToolCall,
        *,
        reasoning: str = "",
    ) -> PermissionDecision:
        """Evaluate a pending tool call and return a :class:`PermissionDecision`.

        This is the single authoritative entry point.  All branches that do
        NOT involve a human TTY return synchronously from policy data alone.

        **Non-hang guarantee:** the branch ``if not self.interactive`` returns
        before :meth:`_prompt` is ever invoked, so no blocking I/O occurs in
        non-interactive environments.

        Parameters
        ----------
        tool:
            The :class:`~agentforge.tools.registry.Tool` the agent wants to
            call.
        call:
            The :class:`~agentforge.messages.ToolCall` carrying the model's
            argument payload.
        reasoning:
            Optional model-supplied justification string shown to the human
            and stored in the trace.

        Returns
        -------
        PermissionDecision
        """
        classification = self.classify(tool)
        args = call.arguments

        # --- Deny by policy -----------------------------------------------
        if classification == "deny":
            return PermissionDecision(
                tool=tool.name,
                args=args,
                outcome=Decision.denied,
                reason="denied by policy",
                auto=True,
            )

        # --- Auto-approve by policy ----------------------------------------
        if classification == "auto":
            return PermissionDecision(
                tool=tool.name,
                args=args,
                outcome=Decision.approved,
                reason=reasoning,
                auto=True,
            )

        # --- Gated call: classification == "gate" --------------------------

        # assume_yes flag (-y / CI auto mode): approve without a prompt.
        if self.assume_yes:
            return PermissionDecision(
                tool=tool.name,
                args=args,
                outcome=Decision.approved,
                reason="auto-approved via --yes flag",
                auto=True,
            )

        # Non-interactive (no TTY): NEVER hang — resolve from policy only.
        # -----------------------------------------------------------------
        # NON-HANG GUARANTEE: this block returns immediately.  _prompt is
        # NOT called.  No input(), no blocking reads, ever.
        if not self.interactive:
            policy = self._cfg.non_interactive
            if policy is NonInteractivePolicy.deny:
                return PermissionDecision(
                    tool=tool.name,
                    args=args,
                    outcome=Decision.denied,
                    reason="non-interactive: gated tool denied",
                    auto=True,
                )
            # NonInteractivePolicy.allow_auto_approved:
            # Only tools in the explicit auto_approve list may run; gated
            # tools are NOT in that list (they got here because classify()
            # returned "gate"), so deny them too.
            return PermissionDecision(
                tool=tool.name,
                args=args,
                outcome=Decision.denied,
                reason="non-interactive: gated tool denied (not in auto_approve list)",
                auto=True,
            )

        # Interactive TTY: ask the human.
        return self._prompt(tool, call, reasoning)

    # ------------------------------------------------------------------
    # Interactive prompt
    # ------------------------------------------------------------------

    def _prompt(
        self,
        tool: Tool,
        call: ToolCall,
        reasoning: str,
    ) -> PermissionDecision:
        """Display a rich prompt and collect the human's decision.

        Options presented:
        - ``[a]`` Approve once
        - ``[d]`` Deny
        - ``[e]`` Edit arguments (then approve with edited args)
        - ``[A]`` Always allow this tool for the rest of the session

        The ``_ask`` callable is injectable — tests replace it with a
        scripted function so no real TTY is needed in the test suite.

        Parameters
        ----------
        tool:
            The tool awaiting approval.
        call:
            The pending tool call with model-supplied arguments.
        reasoning:
            Model's stated justification (may be empty string).

        Returns
        -------
        PermissionDecision
        """
        args = call.arguments

        # Build the rich display panel.
        header = f"[bold yellow]Permission required[/bold yellow]: [bold]{tool.name}[/bold]"
        body_lines = [
            f"[dim]Risk:[/dim] {tool.risk}",
            f"[dim]Args:[/dim]",
        ]
        # Pretty-print args inline; fallback to repr if Pretty fails.
        try:
            args_str = json.dumps(args, indent=2, default=str)
        except Exception:
            args_str = repr(args)
        body_lines.append(args_str)

        if reasoning:
            body_lines.append(f"\n[dim]Reasoning:[/dim] {reasoning}")

        body_lines.append(
            "\n[bold]Choose:[/bold] "
            "\\[a] approve once  "
            "\\[d] deny  "
            "\\[e] edit args  "
            "\\[A] always allow"
        )

        self._console.print(
            Panel(
                "\n".join(body_lines),
                title=header,
                border_style="yellow",
            )
        )

        # Collect the human's choice.
        choice = self._ask("Choice [a/d/e/A]").strip()

        if choice == "A":
            # Always allow this tool for the rest of the session.
            self._always_allow.add(tool.name)
            return PermissionDecision(
                tool=tool.name,
                args=args,
                outcome=Decision.approved,
                reason="always-allow granted by user",
                auto=False,
            )

        if choice == "e":
            # Ask for new args as a JSON string; re-ask once on parse failure.
            return self._collect_edited_args(tool, args, reasoning)

        if choice == "d":
            return PermissionDecision(
                tool=tool.name,
                args=args,
                outcome=Decision.denied,
                reason="denied by user",
                auto=False,
            )

        # Default: "a" or any unrecognised input → approve once.
        return PermissionDecision(
            tool=tool.name,
            args=args,
            outcome=Decision.approved,
            reason=reasoning,
            auto=False,
        )

    def _collect_edited_args(
        self,
        tool: Tool,
        original_args: dict[str, Any],
        reasoning: str,
    ) -> PermissionDecision:
        """Collect edited arguments from the user as a JSON string.

        Attempts to parse the provided JSON once.  If parsing fails, asks
        again.  On a second failure, falls back to a denial so the agent
        can retry rather than proceeding with malformed data.

        Parameters
        ----------
        tool:
            The tool whose arguments are being edited.
        original_args:
            The model-supplied args shown as a reference.
        reasoning:
            Forwarded to the resulting decision's ``reason`` field.

        Returns
        -------
        PermissionDecision
            ``outcome=edited`` with ``edited_args`` set, or ``outcome=denied``
            if both parse attempts failed.
        """
        hint = f"Enter new args as JSON (current: {json.dumps(original_args, default=str)})"

        for attempt in range(2):
            raw = self._ask(hint).strip()
            try:
                new_args = json.loads(raw)
                if not isinstance(new_args, dict):
                    raise ValueError("top-level JSON value must be an object (dict)")
                return PermissionDecision(
                    tool=tool.name,
                    args=original_args,
                    outcome=Decision.edited,
                    reason=reasoning,
                    edited_args=new_args,
                    auto=False,
                )
            except (json.JSONDecodeError, ValueError) as exc:
                if attempt == 0:
                    self._console.print(
                        f"[red]Invalid JSON:[/red] {exc}. Please try again."
                    )
                else:
                    self._console.print(
                        "[red]Invalid JSON on second attempt — treating as deny.[/red]"
                    )

        return PermissionDecision(
            tool=tool.name,
            args=original_args,
            outcome=Decision.denied,
            reason="edit failed: could not parse JSON args (2 attempts)",
            auto=False,
        )
