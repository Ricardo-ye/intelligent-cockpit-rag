from __future__ import annotations

import json

import pytest

from rag_eval.costs import PricingError, estimate_cost, load_pricing


def test_estimate_cost_includes_generation_and_judge_tokens() -> None:
    runtime = {
        "llm": {"model": "demo", "prompt_tokens": 1_000_000, "completion_tokens": 500_000},
        "judge": {"judge_prompt_tokens": 100_000, "judge_completion_tokens": 50_000},
    }
    pricing = {"version": "test", "currency": "USD", "models": {
        "demo": {"prompt_per_million": 2.0, "completion_per_million": 4.0},
    }}
    result = estimate_cost(runtime, pricing)
    assert result["available"] is True
    assert result["generation_cost"] == 4.0
    assert result["judge_cost"] == 0.4
    assert result["estimated_total_cost"] == 4.4


def test_estimate_cost_refuses_unknown_price() -> None:
    result = estimate_cost({"llm": {"model": "unknown"}}, {"currency": "USD", "models": {}})
    assert result == {"available": False, "reason": "价格表未包含模型：unknown", "model": "unknown"}


def test_load_pricing_rejects_invalid_values(tmp_path) -> None:
    path = tmp_path / "prices.json"
    path.write_text(json.dumps({"currency": "USD", "models": {
        "demo": {"prompt_per_million": -1, "completion_per_million": 1},
    }}), encoding="utf-8")
    with pytest.raises(PricingError, match="非负"):
        load_pricing(path)
