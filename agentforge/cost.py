"""Pricing catalog, cost math, and model discovery for the ``agentforge models``
command and orchestrator budget enforcement.

Public surface
--------------
- :class:`ModelPricing`           dataclass holding per-token USD rates + metadata
- :func:`cost_usd`                compute USD cost from a :class:`~agentforge.messages.Usage`
- :class:`PricingCatalog`         dict-backed registry with ``get``/``add``/``from_openrouter``
- :func:`fetch_openrouter_models` fetch live model list from OpenRouter
- :func:`print_models_table`      render a rich Table of tool-capable models + pricing

The orchestrator uses :func:`cost_usd` + :class:`PricingCatalog` to enforce
the hard per-run USD cap. ``agentforge models`` calls :func:`print_models_table`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx

from agentforge.messages import Usage

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hardcoded fallback pricing (used when OpenRouter catalog is unreachable)
# Prices are per-token in USD; the OpenRouter catalog gives $/M tokens as
# strings, so we divide by 1_000_000 when importing from there.
# ---------------------------------------------------------------------------

#: Small set of known-good models so cost math works offline.
# Verified against OpenRouter's live catalog at build time (2026-06). The live
# `agentforge models` probe is the real source of truth; this table only keeps
# cost math working offline / when the catalog is unreachable.
FALLBACK_PRICING: dict[str, "ModelPricing"] = {
    "deepseek/deepseek-v4-flash": None,       # populated after class definition
    "qwen/qwen3.6-flash": None,
    "google/gemma-4-26b-a4b-it": None,
    "minimax/minimax-m3": None,
}


# ---------------------------------------------------------------------------
# Core dataclass
# ---------------------------------------------------------------------------


@dataclass
class ModelPricing:
    """Per-token USD prices and metadata for a single model.

    Parameters
    ----------
    prompt_usd_per_token:
        Cost per input token in USD.
    completion_usd_per_token:
        Cost per output token in USD.
    context_length:
        Maximum context window in tokens (``None`` if unknown).
    supports_tools:
        Whether the model supports native function/tool calling.
    """

    prompt_usd_per_token: float
    completion_usd_per_token: float
    context_length: Optional[int] = None
    supports_tools: bool = True


# Fill in the fallback entries now that ModelPricing is defined.
# deepseek/deepseek-v4-flash: $0.09/$0.18 per million tokens
FALLBACK_PRICING["deepseek/deepseek-v4-flash"] = ModelPricing(
    prompt_usd_per_token=9e-8,          # 0.09 / 1_000_000
    completion_usd_per_token=1.8e-7,    # 0.18 / 1_000_000
    context_length=1_000_000,
    supports_tools=True,
)
# qwen/qwen3.6-flash: $0.1875/$1.125 per million tokens — cheap worker agent
FALLBACK_PRICING["qwen/qwen3.6-flash"] = ModelPricing(
    prompt_usd_per_token=1.875e-7,
    completion_usd_per_token=1.125e-6,
    context_length=1_000_000,
    supports_tools=True,
)
# google/gemma-4-26b-a4b-it: $0.06/$0.33 per million tokens — cheapest tool-capable
FALLBACK_PRICING["google/gemma-4-26b-a4b-it"] = ModelPricing(
    prompt_usd_per_token=6e-8,
    completion_usd_per_token=3.3e-7,
    context_length=262_144,
    supports_tools=True,
)
# minimax/minimax-m3: $0.30/$1.20 per million tokens — strong general agent
FALLBACK_PRICING["minimax/minimax-m3"] = ModelPricing(
    prompt_usd_per_token=3e-7,
    completion_usd_per_token=1.2e-6,
    context_length=1_040_000,
    supports_tools=True,
)


# ---------------------------------------------------------------------------
# Cost computation
# ---------------------------------------------------------------------------


def cost_usd(usage: Usage, pricing: ModelPricing) -> float:
    """Compute the USD cost for a single LLM call.

    Parameters
    ----------
    usage:
        Token counts from :class:`~agentforge.messages.LLMResponse.usage`.
    pricing:
        Per-token rates from the catalog.

    Returns
    -------
    float
        Total cost in USD (prompt + completion).
    """
    return (
        usage.prompt_tokens * pricing.prompt_usd_per_token
        + usage.completion_tokens * pricing.completion_usd_per_token
    )


# ---------------------------------------------------------------------------
# Pricing catalog
# ---------------------------------------------------------------------------


class PricingCatalog:
    """Registry mapping model slugs to :class:`ModelPricing` entries.

    Backed by a plain ``dict``. Build one from the live OpenRouter catalog via
    :meth:`from_openrouter`, or start empty and call :meth:`add` to populate.
    """

    def __init__(self, entries: Optional[dict[str, ModelPricing]] = None) -> None:
        self._data: dict[str, ModelPricing] = dict(entries or {})

    # ------------------------------------------------------------------
    # Instance methods
    # ------------------------------------------------------------------

    def get(self, model: str) -> Optional[ModelPricing]:
        """Return the :class:`ModelPricing` for *model*, or ``None`` if
        the model is not in the catalog.

        Falls back to :data:`FALLBACK_PRICING` when a live catalog was built
        but the specific slug is missing.
        """
        result = self._data.get(model)
        if result is None:
            result = FALLBACK_PRICING.get(model)
        return result

    def add(self, model: str, pricing: ModelPricing) -> None:
        """Register or update pricing for *model*."""
        self._data[model] = pricing

    def __len__(self) -> int:
        return len(self._data)

    # ------------------------------------------------------------------
    # Class methods / factory
    # ------------------------------------------------------------------

    @classmethod
    def from_openrouter(cls, models_json: list[dict[str, Any]]) -> "PricingCatalog":
        """Build a :class:`PricingCatalog` from the OpenRouter ``/models``
        response data list.

        The OpenRouter response returns pricing as strings in USD-per-token
        (e.g. ``"0.0000000900"`` for $0.09/M tokens). ``supported_parameters``
        is a list of capability strings; we check for ``"tools"`` membership to
        set :attr:`ModelPricing.supports_tools`.

        Entries with missing or ``"0"`` prices are included — they represent
        free or uncapped models.

        Parameters
        ----------
        models_json:
            The ``data`` list from ``GET https://openrouter.ai/api/v1/models``.
        """
        catalog = cls(FALLBACK_PRICING.copy())

        for entry in models_json:
            model_id: str = entry.get("id", "")
            if not model_id:
                continue

            pricing_block = entry.get("pricing", {})
            try:
                prompt_per_token = float(pricing_block.get("prompt") or 0)
                completion_per_token = float(pricing_block.get("completion") or 0)
            except (TypeError, ValueError):
                logger.debug("Skipping model %r — unparseable pricing", model_id)
                continue

            context_length: Optional[int] = entry.get("context_length")
            if context_length is not None:
                try:
                    context_length = int(context_length)
                except (TypeError, ValueError):
                    context_length = None

            supported_params: list[str] = entry.get("supported_parameters") or []
            supports_tools = "tools" in supported_params

            catalog.add(
                model_id,
                ModelPricing(
                    prompt_usd_per_token=prompt_per_token,
                    completion_usd_per_token=completion_per_token,
                    context_length=context_length,
                    supports_tools=supports_tools,
                ),
            )

        return catalog


# ---------------------------------------------------------------------------
# Network helpers
# ---------------------------------------------------------------------------

_OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"


def fetch_openrouter_models(
    api_key: Optional[str] = None,
    timeout: float = 30,
) -> list[dict[str, Any]]:
    """Fetch the live model list from OpenRouter.

    The OpenRouter catalog endpoint is public — an API key is optional but
    recommended to avoid anonymous rate limits.

    Parameters
    ----------
    api_key:
        OpenRouter API key. When ``None``, the request is unauthenticated.
    timeout:
        Request timeout in seconds (default: 30).

    Returns
    -------
    list[dict]
        The ``data`` array from the OpenRouter ``/models`` response.

    Raises
    ------
    httpx.HTTPError
        On network or HTTP failure (callers should catch and handle).
    """
    headers: dict[str, str] = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    with httpx.Client(timeout=timeout) as client:
        resp = client.get(_OPENROUTER_MODELS_URL, headers=headers)
        resp.raise_for_status()
        data = resp.json()

    return data.get("data", [])


# ---------------------------------------------------------------------------
# Rich table renderer
# ---------------------------------------------------------------------------


def print_models_table(
    console: Any,
    query: Optional[str] = None,
    tools_only: bool = True,
) -> None:
    """Fetch the OpenRouter model catalog and print a Rich table.

    Columns: Model, In $/M, Out $/M, Context, Tools

    Parameters
    ----------
    console:
        A :class:`rich.console.Console` instance.
    query:
        Optional substring filter applied to model IDs (case-insensitive).
    tools_only:
        When ``True`` (default), only show models that support tool calling.
    """
    from rich.table import Table

    # Attempt live fetch; fall back to the hardcoded catalog on failure
    models_data: list[dict[str, Any]] = []
    live_fetch_ok = False
    try:
        models_data = fetch_openrouter_models()
        live_fetch_ok = True
    except Exception as exc:
        console.print(
            f"[yellow]Warning:[/yellow] could not reach OpenRouter catalog "
            f"({type(exc).__name__}: {exc}). Showing fallback pricing only."
        )

    if live_fetch_ok:
        catalog = PricingCatalog.from_openrouter(models_data)
        # Build display list from live data
        rows = _build_rows_from_live(models_data, query=query, tools_only=tools_only)
    else:
        # Fall back to the hardcoded entries
        rows = _build_rows_from_fallback(query=query, tools_only=tools_only)

    # Sort by prompt price ascending (cheapest first)
    rows.sort(key=lambda r: r["prompt_per_m"])

    table = Table(title="OpenRouter Models", show_lines=False, highlight=True)
    table.add_column("Model", style="bold cyan", no_wrap=True)
    table.add_column("In $/M", style="green", justify="right")
    table.add_column("Out $/M", style="yellow", justify="right")
    table.add_column("Context", justify="right")
    table.add_column("Tools", justify="center")

    for row in rows:
        table.add_row(
            row["id"],
            _fmt_price(row["prompt_per_m"]),
            _fmt_price(row["completion_per_m"]),
            _fmt_context(row["context_length"]),
            "[green]yes[/green]" if row["supports_tools"] else "[dim]no[/dim]",
        )

    if not rows:
        console.print("[dim]No models matched your filter.[/dim]")
        return

    console.print(table)
    console.print(
        f"[dim]{len(rows)} model(s) shown"
        + (" (tool-capable only)" if tools_only else "")
        + ("" if live_fetch_ok else " — offline fallback")
        + "[/dim]"
    )


# ---------------------------------------------------------------------------
# Internal helpers for print_models_table
# ---------------------------------------------------------------------------


def _build_rows_from_live(
    models_data: list[dict[str, Any]],
    *,
    query: Optional[str],
    tools_only: bool,
) -> list[dict[str, Any]]:
    rows = []
    q = query.lower() if query else None

    for entry in models_data:
        model_id: str = entry.get("id", "")
        if not model_id:
            continue
        if q and q not in model_id.lower():
            continue

        supported_params: list[str] = entry.get("supported_parameters") or []
        supports_tools = "tools" in supported_params
        if tools_only and not supports_tools:
            continue

        pricing_block = entry.get("pricing", {})
        try:
            prompt_per_token = float(pricing_block.get("prompt") or 0)
            completion_per_token = float(pricing_block.get("completion") or 0)
        except (TypeError, ValueError):
            continue

        context_length: Optional[int] = entry.get("context_length")
        if context_length is not None:
            try:
                context_length = int(context_length)
            except (TypeError, ValueError):
                context_length = None

        rows.append(
            {
                "id": model_id,
                "prompt_per_m": prompt_per_token * 1_000_000,
                "completion_per_m": completion_per_token * 1_000_000,
                "context_length": context_length,
                "supports_tools": supports_tools,
            }
        )
    return rows


def _build_rows_from_fallback(
    *,
    query: Optional[str],
    tools_only: bool,
) -> list[dict[str, Any]]:
    rows = []
    q = query.lower() if query else None

    for model_id, pricing in FALLBACK_PRICING.items():
        if pricing is None:
            continue
        if q and q not in model_id.lower():
            continue
        if tools_only and not pricing.supports_tools:
            continue
        rows.append(
            {
                "id": model_id,
                "prompt_per_m": pricing.prompt_usd_per_token * 1_000_000,
                "completion_per_m": pricing.completion_usd_per_token * 1_000_000,
                "context_length": pricing.context_length,
                "supports_tools": pricing.supports_tools,
            }
        )
    return rows


def _fmt_price(price_per_m: float) -> str:
    """Format a per-million-token price for display."""
    if price_per_m == 0.0:
        return "free"
    if price_per_m < 0.01:
        return f"${price_per_m:.4f}"
    return f"${price_per_m:.4f}"


def _fmt_context(context_length: Optional[int]) -> str:
    """Format a context window token count for display (e.g. 1M, 131K)."""
    if context_length is None:
        return "—"
    if context_length >= 1_000_000:
        return f"{context_length // 1_000_000}M"
    if context_length >= 1_000:
        return f"{context_length // 1_000}K"
    return str(context_length)
