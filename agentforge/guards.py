"""Runtime safety guards shared by the agent loop and the orchestrator.

Kept in their own module (no heavy imports) so both ``agent.py`` and
``orchestrator.py`` can use them without an import cycle.

- :class:`BudgetGuard` — the financial backstop. Estimates cost before a call
  and measures it after; raises :class:`BudgetExceeded` the moment the hard cap
  would be crossed.
- :class:`RepeatTracker` — repeated-action detection (same tool + same args N
  times in a row).
- :class:`StopReason` — the canonical set of non-fatal stop reasons.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from typing import Any, Optional

from agentforge.cost import ModelPricing, PricingCatalog, cost_usd
from agentforge.messages import Usage


class BudgetExceeded(Exception):
    """Raised when a run would cross its hard USD cap. Caught by the orchestrator,
    which finalizes a partial result + trace. Never a crash."""

    def __init__(self, spent: float, cap: float) -> None:
        self.spent = spent
        self.cap = cap
        super().__init__(f"budget cap exceeded: ${spent:.6f} would exceed ${cap:.6f}")


class StopReason:
    COMPLETED = "completed"
    MAX_ITERATIONS = "max_iterations"
    MAX_TOTAL_ITERATIONS = "max_total_iterations"
    BUDGET_EXCEEDED = "budget_exceeded"
    REPEATED_ACTION = "repeated_action"
    HANDOFF_CYCLE = "handoff_cycle"
    WALL_CLOCK = "wall_clock_timeout"
    ERROR = "error"


@dataclass
class BudgetGuard:
    """Tracks cumulative USD spend against a hard cap.

    ``estimate_before`` is intentionally conservative (it assumes a full
    completion of ``assumed_completion_tokens``) so we abort *before* a call
    that would blow the cap rather than after.
    """

    cap_usd: float
    catalog: PricingCatalog
    assumed_completion_tokens: int = 1024
    total_usd: float = 0.0

    def _pricing(self, model: str) -> ModelPricing:
        p = self.catalog.get(model)
        if p is None:
            # Unknown model: assume zero so we never falsely abort; live runs
            # always have a populated catalog, and unknowns are rare.
            return ModelPricing(0.0, 0.0)
        return p

    def estimate(self, prompt_tokens: int, model: str) -> float:
        pricing = self._pricing(model)
        usage = Usage(prompt_tokens=prompt_tokens, completion_tokens=self.assumed_completion_tokens)
        return cost_usd(usage, pricing)

    def check_before(self, prompt_tokens: int, model: str) -> None:
        """Raise if making this call could cross the cap."""
        projected = self.total_usd + self.estimate(prompt_tokens, model)
        if projected > self.cap_usd:
            raise BudgetExceeded(projected, self.cap_usd)

    def record(self, usage: Usage, model: str) -> float:
        """Add the actual cost of a completed call; return the new total."""
        self.total_usd += cost_usd(usage, self._pricing(model))
        return self.total_usd

    def check_after(self) -> None:
        """Raise if the measured total has crossed the cap."""
        if self.total_usd > self.cap_usd:
            raise BudgetExceeded(self.total_usd, self.cap_usd)


class RepeatTracker:
    """Detects the same tool being called with identical args N times running."""

    def __init__(self, threshold: int = 3) -> None:
        self.threshold = threshold
        self._recent: deque[str] = deque(maxlen=threshold)

    def push(self, tool: str, args: dict[str, Any]) -> bool:
        """Record a call; return True if the last ``threshold`` are identical."""
        key = tool + "::" + json.dumps(args, sort_keys=True, default=str)
        self._recent.append(key)
        return len(self._recent) == self.threshold and len(set(self._recent)) == 1
