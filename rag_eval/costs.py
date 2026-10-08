"""可复现的 LLM 用量成本估算。

模型价格经常变化，评测引擎不内置会过期的厂商价格。调用方必须提供一份
版本化 JSON 价格表；报告会连同该表的版本和来源一起保存，避免把 token 数误当金额。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


class PricingError(ValueError):
    """价格表格式或数值不合法。"""


def load_pricing(path: str | Path) -> Dict[str, Any]:
    """加载并校验价格表。

    最小格式::

        {"version": "2026-10-08", "currency": "USD", "models": {
          "my-model": {"prompt_per_million": 0.2, "completion_per_million": 0.8}
        }}
    """

    source = Path(path)
    if not source.is_file():
        raise PricingError(f"价格表不存在：{source}")
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PricingError(f"无法读取价格表：{exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("models"), dict):
        raise PricingError("价格表必须包含 models 对象。")
    currency = str(data.get("currency") or "").strip().upper()
    if not currency:
        raise PricingError("价格表必须声明 currency。")
    validated: Dict[str, Any] = {**data, "currency": currency, "source": str(source)}
    for model, price in data["models"].items():
        if not isinstance(model, str) or not model.strip() or not isinstance(price, Mapping):
            raise PricingError("models 中每一项必须是非空模型名与对象价格。")
        for field in ("prompt_per_million", "completion_per_million"):
            value = price.get(field)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
                raise PricingError(f"模型 {model} 的 {field} 必须是非负数字。")
    return validated


def estimate_cost(runtime: Mapping[str, Any], pricing: Mapping[str, Any]) -> Dict[str, Any]:
    """从评测报告 ``runtime`` 估算成本，绝不在价格缺失时猜测。"""

    llm = runtime.get("llm") if isinstance(runtime, Mapping) else None
    if not isinstance(llm, Mapping):
        return {"available": False, "reason": "本次未调用 LLM，无需估算成本。"}
    model = str(llm.get("model") or "")
    model_price = (pricing.get("models") or {}).get(model)
    if not isinstance(model_price, Mapping):
        return {"available": False, "reason": f"价格表未包含模型：{model}", "model": model}

    def _n(container: Mapping[str, Any], key: str) -> int:
        value = container.get(key, 0)
        return int(value) if isinstance(value, (int, float)) and value >= 0 else 0

    prompt = _n(llm, "prompt_tokens")
    completion = _n(llm, "completion_tokens")
    judge = runtime.get("judge") if isinstance(runtime.get("judge"), Mapping) else {}
    judge_prompt = _n(judge, "judge_prompt_tokens")
    judge_completion = _n(judge, "judge_completion_tokens")
    prompt_price = float(model_price["prompt_per_million"])
    completion_price = float(model_price["completion_per_million"])
    generation_cost = prompt / 1_000_000 * prompt_price + completion / 1_000_000 * completion_price
    judge_cost = judge_prompt / 1_000_000 * prompt_price + judge_completion / 1_000_000 * completion_price
    return {
        "available": True,
        "currency": pricing["currency"],
        "pricing_version": pricing.get("version") or "unspecified",
        "pricing_source": pricing.get("source") or "inline",
        "model": model,
        "prompt_per_million": prompt_price,
        "completion_per_million": completion_price,
        "generation_cost": round(generation_cost, 8),
        "judge_cost": round(judge_cost, 8),
        "estimated_total_cost": round(generation_cost + judge_cost, 8),
        "tokens": {
            "prompt": prompt,
            "completion": completion,
            "judge_prompt": judge_prompt,
            "judge_completion": judge_completion,
        },
        "note": "估算仅覆盖记录到报告的文本模型 token；Embedding、网络、平台最低消费与税费未包含。",
    }


__all__ = ["PricingError", "load_pricing", "estimate_cost"]
