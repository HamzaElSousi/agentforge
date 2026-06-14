"""Tests for agentforge/cost.py — cost math, PricingCatalog, from_openrouter parsing.

No real network calls are made. fetch_openrouter_models is NOT called; we
exercise from_openrouter with a manually constructed dict.
"""

from __future__ import annotations

import pytest

from agentforge.cost import (
    FALLBACK_PRICING,
    ModelPricing,
    PricingCatalog,
    cost_usd,
)
from agentforge.messages import Usage


# ---------------------------------------------------------------------------
# cost_usd math
# ---------------------------------------------------------------------------


class TestCostUsd:
    def test_basic_cost_calculation(self):
        """cost_usd(Usage(1000, 1000), ModelPricing(1e-6, 2e-6)) == 0.003"""
        usage = Usage(prompt_tokens=1000, completion_tokens=1000)
        pricing = ModelPricing(
            prompt_usd_per_token=1e-6,
            completion_usd_per_token=2e-6,
        )
        result = cost_usd(usage, pricing)
        assert result == pytest.approx(0.003), (
            "1000 * 1e-6 + 1000 * 2e-6 = 0.001 + 0.002 = 0.003"
        )

    def test_zero_tokens_cost_zero(self):
        usage = Usage(prompt_tokens=0, completion_tokens=0)
        pricing = ModelPricing(prompt_usd_per_token=1e-6, completion_usd_per_token=2e-6)
        assert cost_usd(usage, pricing) == pytest.approx(0.0)

    def test_only_prompt_tokens(self):
        usage = Usage(prompt_tokens=500, completion_tokens=0)
        pricing = ModelPricing(prompt_usd_per_token=2e-6, completion_usd_per_token=4e-6)
        assert cost_usd(usage, pricing) == pytest.approx(500 * 2e-6)

    def test_only_completion_tokens(self):
        usage = Usage(prompt_tokens=0, completion_tokens=200)
        pricing = ModelPricing(prompt_usd_per_token=1e-6, completion_usd_per_token=5e-6)
        assert cost_usd(usage, pricing) == pytest.approx(200 * 5e-6)

    def test_asymmetric_pricing(self):
        """Completion tokens typically cost more than prompt tokens."""
        usage = Usage(prompt_tokens=100, completion_tokens=100)
        pricing = ModelPricing(
            prompt_usd_per_token=1e-6,
            completion_usd_per_token=3e-6,
        )
        # 100 * 1e-6 + 100 * 3e-6 = 0.0001 + 0.0003 = 0.0004
        assert cost_usd(usage, pricing) == pytest.approx(0.0004)

    def test_cost_usd_with_real_fallback_pricing(self):
        """Spot-check cost using deepseek fallback pricing: should be tiny."""
        pricing = FALLBACK_PRICING["deepseek/deepseek-v4-flash"]
        usage = Usage(prompt_tokens=1_000_000, completion_tokens=0)
        # 1M tokens * $0.09/M = $0.09
        assert cost_usd(usage, pricing) == pytest.approx(0.09, rel=1e-3)


# ---------------------------------------------------------------------------
# PricingCatalog — add / get round-trip
# ---------------------------------------------------------------------------


class TestPricingCatalogBasics:
    def test_add_then_get_round_trip(self):
        cat = PricingCatalog()
        pricing = ModelPricing(prompt_usd_per_token=1e-6, completion_usd_per_token=2e-6)
        cat.add("my/model", pricing)
        got = cat.get("my/model")
        assert got is not None, "get() must return the added entry"
        assert got.prompt_usd_per_token == pytest.approx(1e-6)
        assert got.completion_usd_per_token == pytest.approx(2e-6)

    def test_get_returns_none_for_unknown_and_not_in_fallback(self):
        cat = PricingCatalog()
        result = cat.get("totally/unknown/slug/xyz")
        assert result is None, "Unknown model not in fallback should return None"

    def test_add_overwrites_existing_entry(self):
        cat = PricingCatalog()
        cat.add("m", ModelPricing(1e-6, 2e-6))
        cat.add("m", ModelPricing(5e-6, 10e-6))
        got = cat.get("m")
        assert got.prompt_usd_per_token == pytest.approx(5e-6)

    def test_len_reflects_added_entries(self):
        cat = PricingCatalog()
        assert len(cat) == 0
        cat.add("m1", ModelPricing(1e-6, 2e-6))
        assert len(cat) == 1
        cat.add("m2", ModelPricing(1e-6, 2e-6))
        assert len(cat) == 2

    def test_catalog_initialized_with_entries(self):
        entries = {"a/model": ModelPricing(1e-6, 2e-6)}
        cat = PricingCatalog(entries)
        assert cat.get("a/model") is not None


# ---------------------------------------------------------------------------
# PricingCatalog.get() — fallback to FALLBACK_PRICING
# ---------------------------------------------------------------------------


class TestPricingCatalogFallback:
    def test_fallback_slug_returned_for_deepseek(self):
        """An empty catalog falls back to FALLBACK_PRICING for known slugs."""
        cat = PricingCatalog()  # empty, nothing added
        got = cat.get("deepseek/deepseek-v4-flash")
        assert got is not None, (
            "deepseek/deepseek-v4-flash must be returned from FALLBACK_PRICING"
        )
        assert isinstance(got, ModelPricing)

    def test_fallback_deepseek_pricing_values(self):
        cat = PricingCatalog()
        got = cat.get("deepseek/deepseek-v4-flash")
        # $0.09/M tokens -> 9e-8 per token
        assert got.prompt_usd_per_token == pytest.approx(9e-8)
        assert got.completion_usd_per_token == pytest.approx(1.8e-7)

    def test_fallback_qwen_model_returned(self):
        cat = PricingCatalog()
        got = cat.get("qwen/qwen3.6-flash")
        assert got is not None
        assert got.supports_tools is True

    def test_catalog_entry_takes_precedence_over_fallback(self):
        """An explicitly added entry must shadow the fallback."""
        custom_pricing = ModelPricing(
            prompt_usd_per_token=9999e-6,
            completion_usd_per_token=9999e-6,
        )
        cat = PricingCatalog()
        cat.add("deepseek/deepseek-v4-flash", custom_pricing)
        got = cat.get("deepseek/deepseek-v4-flash")
        assert got.prompt_usd_per_token == pytest.approx(9999e-6), (
            "Explicit catalog entry must override fallback"
        )


# ---------------------------------------------------------------------------
# PricingCatalog.from_openrouter() — parses fake models_json
# ---------------------------------------------------------------------------


class TestFromOpenRouter:
    def _make_models_json(self, entries: list[dict]) -> list[dict]:
        return entries

    def test_basic_entry_parsed(self):
        models = self._make_models_json([
            {
                "id": "test/model-v1",
                "pricing": {
                    "prompt": "0.0000010000",
                    "completion": "0.0000020000",
                },
                "context_length": 128_000,
                "supported_parameters": ["tools", "temperature"],
            }
        ])
        cat = PricingCatalog.from_openrouter(models)
        got = cat.get("test/model-v1")
        assert got is not None
        assert got.prompt_usd_per_token == pytest.approx(1e-6)
        assert got.completion_usd_per_token == pytest.approx(2e-6)

    def test_supports_tools_true_when_tools_in_supported_parameters(self):
        models = self._make_models_json([
            {
                "id": "tool/model",
                "pricing": {"prompt": "0.000001", "completion": "0.000002"},
                "context_length": 32_000,
                "supported_parameters": ["tools"],
            }
        ])
        cat = PricingCatalog.from_openrouter(models)
        got = cat.get("tool/model")
        assert got.supports_tools is True

    def test_supports_tools_false_when_tools_not_in_supported_parameters(self):
        models = self._make_models_json([
            {
                "id": "no/tools-model",
                "pricing": {"prompt": "0.000001", "completion": "0.000002"},
                "context_length": 8_000,
                "supported_parameters": ["temperature", "max_tokens"],
            }
        ])
        cat = PricingCatalog.from_openrouter(models)
        got = cat.get("no/tools-model")
        assert got.supports_tools is False

    def test_context_length_parsed(self):
        models = self._make_models_json([
            {
                "id": "ctx/model",
                "pricing": {"prompt": "0.000001", "completion": "0.000002"},
                "context_length": 200_000,
                "supported_parameters": ["tools"],
            }
        ])
        cat = PricingCatalog.from_openrouter(models)
        got = cat.get("ctx/model")
        assert got.context_length == 200_000

    def test_entry_with_missing_id_skipped(self):
        models = self._make_models_json([
            {
                # no "id" field
                "pricing": {"prompt": "0.000001", "completion": "0.000002"},
                "context_length": 8_000,
                "supported_parameters": ["tools"],
            },
            {
                "id": "valid/model",
                "pricing": {"prompt": "0.000001", "completion": "0.000002"},
                "context_length": 8_000,
                "supported_parameters": ["tools"],
            },
        ])
        cat = PricingCatalog.from_openrouter(models)
        assert cat.get("valid/model") is not None

    def test_zero_price_entry_included(self):
        """Free/zero-price models must still be included in the catalog."""
        models = self._make_models_json([
            {
                "id": "free/model",
                "pricing": {"prompt": "0", "completion": "0"},
                "context_length": 4_096,
                "supported_parameters": ["tools"],
            }
        ])
        cat = PricingCatalog.from_openrouter(models)
        got = cat.get("free/model")
        assert got is not None
        assert got.prompt_usd_per_token == pytest.approx(0.0)

    def test_from_openrouter_includes_fallback_pricing(self):
        """from_openrouter pre-populates with FALLBACK_PRICING so offline slugs work."""
        cat = PricingCatalog.from_openrouter([])  # empty live data
        got = cat.get("deepseek/deepseek-v4-flash")
        assert got is not None, (
            "from_openrouter catalog must include fallback entries even when live data is empty"
        )

    def test_multiple_models_all_parsed(self):
        models = self._make_models_json([
            {
                "id": f"vendor/model-{i}",
                "pricing": {"prompt": "0.000001", "completion": "0.000002"},
                "context_length": 8_000,
                "supported_parameters": ["tools"],
            }
            for i in range(5)
        ])
        cat = PricingCatalog.from_openrouter(models)
        for i in range(5):
            assert cat.get(f"vendor/model-{i}") is not None


# ---------------------------------------------------------------------------
# ModelPricing dataclass
# ---------------------------------------------------------------------------


class TestModelPricing:
    def test_default_supports_tools_true(self):
        p = ModelPricing(prompt_usd_per_token=1e-6, completion_usd_per_token=2e-6)
        assert p.supports_tools is True

    def test_default_context_length_none(self):
        p = ModelPricing(prompt_usd_per_token=1e-6, completion_usd_per_token=2e-6)
        assert p.context_length is None

    def test_can_set_supports_tools_false(self):
        p = ModelPricing(
            prompt_usd_per_token=1e-6,
            completion_usd_per_token=2e-6,
            supports_tools=False,
        )
        assert p.supports_tools is False
