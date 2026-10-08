"""评测指标：全部为真实计算，无模拟、无占位。

指标分两类，UI 与导出结果中会明确标注来源，避免把启发式代理指标误当 LLM 判官指标：

【确定性指标】（纯本地计算，可复现）
- 检索类：HitRate@K / Precision@K / Recall@K / MRR@K / NDCG@K（基于标注的 relevant_doc_ids）
- 上下文类：ContextPrecision（RAGAS 风格 rank-weighted）、ContextRecall（ground_truth 句子覆盖）
- 答案类：ExactMatch / TokenF1 / RougeL（对比 ground_truth）
- 代理类：Faithfulness(lexical) / AnswerRelevancy(lexical) —— 用词法覆盖率近似，标注为 lexical

【LLM 判官指标】（真实调用 LLM，需 API Key）
- Faithfulness(llm)：把答案拆成原子论断，逐条由 LLM 判定是否被检索上下文支撑
- AnswerRelevancy(llm)：由答案反推若干问题，与原问题做相似度比对

判定相关性统一使用阈值 tau，全部可配置。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence

from .ingest import Chunk
from .llm import LLMClient, LLMError
from .textutils import (
    overlap_ratio, rouge_l, split_sentences, token_f1, truncate,
)

# 默认相关性阈值
TAU_CONTEXT_RELEVANT = 0.30   # 分块被视为「相关」的 token-F1 下限
TAU_SENTENCE_COVERED = 0.45   # ground_truth 句子被视为「被覆盖」的覆盖率下限
TAU_CLAIM_SUPPORTED = 0.45    # 答案论断被视为「被支撑」的覆盖率下限


# ─────────────────────────────────────────────
# 数据结构
# ─────────────────────────────────────────────
@dataclass
class MetricValue:
    """单个指标值 + 来源标注。"""

    name: str
    value: Optional[float]
    family: str          # retrieval | context | answer | faithfulness | relevancy
    method: str          # deterministic | llm-judge
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class JudgeStats:
    calls: int = 0
    latency_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0


# ─────────────────────────────────────────────
# 确定性：检索类 IR 指标
# ─────────────────────────────────────────────
def retrieval_metrics(
    retrieved_doc_ids: Sequence[str],
    relevant_doc_ids: Sequence[str],
    k: int,
) -> Dict[str, Optional[float]]:
    """基于标注的文档级 IR 指标；未标注 relevant_doc_ids 时返回 None（不编造）。

    注意：标注是**文档级**的，而检索结果通常是**分块级**的，同一文档会命中多个分块。
    因此在计算前必须按首次出现顺序去重到文档级，否则会出现 recall / ndcg 大于 1 的错误。
    """
    if not relevant_doc_ids:
        return {"hit_rate": None, "precision_at_k": None, "recall_at_k": None,
                "mrr": None, "ndcg": None}

    # 分块级 -> 文档级去重（保留首次命中时的排名顺序）
    deduped: List[str] = list(dict.fromkeys(retrieved_doc_ids))
    top = deduped[:k]
    rel = set(relevant_doc_ids)

    if not top:
        return {"hit_rate": 0.0, "precision_at_k": 0.0, "recall_at_k": 0.0,
                "mrr": 0.0, "ndcg": 0.0}

    hits = [1 if d in rel else 0 for d in top]

    hit_rate = 1.0 if any(hits) else 0.0
    # 标准 Precision@K 的分母固定为 K；结果不足 K 条也不能因此虚高。
    precision_at_k = sum(hits) / max(k, 1)
    recall_at_k = sum(hits) / len(rel) if rel else 0.0

    mrr = 0.0
    for i, h in enumerate(hits, 1):
        if h:
            mrr = 1.0 / i
            break

    # NDCG：二值增益
    dcg = sum(h / _log2(i + 1) for i, h in enumerate(hits, 1))
    ideal_hits = [1] * min(len(rel), len(top))
    idcg = sum(h / _log2(i + 1) for i, h in enumerate(ideal_hits, 1))
    ndcg = min((dcg / idcg) if idcg > 0 else 0.0, 1.0)

    return {
        "hit_rate": round(hit_rate, 6),
        "precision_at_k": round(min(precision_at_k, 1.0), 6),
        "recall_at_k": round(min(recall_at_k, 1.0), 6),
        "mrr": round(mrr, 6),
        "ndcg": round(ndcg, 6),
    }


def _log2(x: float) -> float:
    import math

    return math.log2(x) if x > 1 else 1.0


# ─────────────────────────────────────────────
# 确定性：上下文类指标
# ─────────────────────────────────────────────
def is_relevant_chunk(chunk: Chunk, ground_truth: str, relevant_doc_ids: Sequence[str]) -> bool:
    """判定某个检索分块是否相关：标注命中 或 与标准答案词法重合达标。"""
    if relevant_doc_ids and chunk.doc_id in set(relevant_doc_ids):
        return True
    return token_f1(chunk.text, ground_truth) >= TAU_CONTEXT_RELEVANT


def context_precision(
    contexts: Sequence[Chunk], ground_truth: str, relevant_doc_ids: Sequence[str]
) -> Optional[float]:
    """RAGAS 风格的上下文精确率（rank-weighted）。"""
    if not contexts:
        return 0.0
    flags = [1 if is_relevant_chunk(c, ground_truth, relevant_doc_ids) else 0 for c in contexts]
    total_rel = sum(flags)
    if total_rel == 0:
        return 0.0
    cum, weighted = 0, 0.0
    for i, f in enumerate(flags, 1):
        cum += f
        if f:
            weighted += (cum / i)
    return round(weighted / total_rel, 6)


def context_recall(
    contexts: Sequence[Chunk], ground_truth: str
) -> Optional[float]:
    """标准答案的句子被检索上下文覆盖的比例。"""
    gold_sents = split_sentences(ground_truth)
    if not gold_sents:
        return 0.0
    joined = [c.text for c in contexts]
    covered = 0
    for s in gold_sents:
        best = max((overlap_ratio(s, c) for c in joined), default=0.0)
        if best >= TAU_SENTENCE_COVERED:
            covered += 1
    return round(covered / len(gold_sents), 6)


# ─────────────────────────────────────────────
# 确定性：答案类指标
# ─────────────────────────────────────────────
def _normalize(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[\s\u3000]+", "", s)
    s = re.sub(r"[，。！？、；：（）()\[\]【】\"'`~!@#$%^&*_+=\-|\\/<>.,;:?!]+", "", s)
    return s


def exact_match(pred: str, gold: str) -> float:
    return 1.0 if _normalize(pred) and _normalize(pred) == _normalize(gold) else 0.0


def answer_metrics(pred: str, gold: str) -> Dict[str, float]:
    return {
        "exact_match": round(exact_match(pred, gold), 6),
        "token_f1": round(token_f1(pred, gold), 6),
        "rouge_l": round(rouge_l(pred, gold), 6),
    }


# ─────────────────────────────────────────────
# 确定性：忠实度 / 相关性（词法代理）
# ─────────────────────────────────────────────
def faithfulness_lexical(answer: str, contexts: Sequence[Chunk]) -> float:
    claims = split_sentences(answer)
    if not claims:
        return 0.0
    ctx = [c.text for c in contexts]
    supported = 0
    for claim in claims:
        best = max((overlap_ratio(claim, c) for c in ctx), default=0.0)
        if best >= TAU_CLAIM_SUPPORTED:
            supported += 1
    return round(supported / len(claims), 6)


def answer_relevancy_lexical(answer: str, question: str) -> float:
    return round(token_f1(answer, question), 6)


# ─────────────────────────────────────────────
# LLM 判官指标
# ─────────────────────────────────────────────
class LLMJudge:
    """真实调用 LLM 的判官（DeepSeek / OpenAI 兼容）。"""

    CLAIM_SYSTEM = (
        "你是信息抽取器。请把给定答案拆解为独立的原子论断（每条只包含一个事实），"
        "保持原意、不要补充、不要合并。只输出 JSON：{\"claims\":[\"论断1\",\"论断2\"]}"
    )
    VERIFY_SYSTEM = (
        "你是严谨的事实核查员。请判断每条论断是否能由【参考资料】直接支撑。"
        "只有在参考资料中明确能找到依据时才判 true，需要推理或资料未提及一律 false。"
        "只输出 JSON：{\"verdicts\":[{\"id\":1,\"supported\":true,\"reason\":\"简短依据\"}]}"
    )
    QUESTION_GEN_SYSTEM = (
        "你是出题人。请根据给定答案，生成 3 个它能够回答的问题，"
        "只输出 JSON：{\"questions\":[\"问题1\",\"问题2\",\"问题3\"]}"
    )

    def __init__(self, llm: LLMClient):
        self.llm = llm
        self.stats = JudgeStats()

    def _call_json(self, system: str, user: str, max_tokens: int = 512) -> Any:
        import time

        t0 = time.perf_counter()
        data = self.llm.chat_json(system, user, max_tokens=max_tokens)
        self.stats.calls += 1
        self.stats.latency_s += time.perf_counter() - t0
        self.stats.prompt_tokens = self.llm.usage.prompt_tokens
        self.stats.completion_tokens = self.llm.usage.completion_tokens
        return data

    def faithfulness(self, answer: str, contexts: Sequence[Chunk]) -> Optional[float]:
        """LLM 版忠实度：拆论断 -> 逐条核查是否被上下文支撑。"""
        if not answer.strip():
            return 0.0
        ctx_text = "\n\n".join(f"[{i}] {c.text}" for i, c in enumerate(contexts, 1))
        try:
            cdata = self._call_json(
                self.CLAIM_SYSTEM, f"答案：{answer}", max_tokens=512,
            )
            claims = cdata.get("claims", []) if isinstance(cdata, dict) else cdata
            claims = [str(c).strip() for c in claims if str(c).strip()]
            if not claims:
                return None

            numbered = "\n".join(f"[{i}] {c}" for i, c in enumerate(claims, 1))
            vdata = self._call_json(
                self.VERIFY_SYSTEM,
                f"【参考资料】\n{truncate(ctx_text, 6000)}\n\n【待核查论断】\n{numbered}",
                max_tokens=256 + 60 * len(claims),
            )
            verdicts = vdata.get("verdicts", []) if isinstance(vdata, dict) else vdata
            valid = {
                int(v.get("id")): bool(v.get("supported"))
                for v in verdicts
                if isinstance(v, dict) and str(v.get("id", "")).isdigit()
            }
            if not verdicts:
                return None
            # 缺失的论断判定视为不支持，避免模型少返回几条导致分数虚高。
            supported = sum(1 for i in range(1, len(claims) + 1) if valid.get(i, False))
            return round(supported / len(claims), 6)
        except LLMError:
            return None

    def answer_relevancy(self, answer: str, question: str) -> Optional[float]:
        """LLM 版答案相关性：由答案反推问题，与原问题比对（词法 F1，口径固定可复现）。"""
        if not answer.strip():
            return 0.0
        try:
            data = self._call_json(
                self.QUESTION_GEN_SYSTEM,
                f"答案：{answer}",
                max_tokens=256,
            )
            qs = data.get("questions", []) if isinstance(data, dict) else data
            qs = [str(q).strip() for q in qs if str(q).strip()]
            if not qs:
                return None
            return round(sum(token_f1(q, question) for q in qs) / len(qs), 6)
        except LLMError:
            return None


__all__ = [
    "MetricValue", "JudgeStats", "LLMJudge",
    "retrieval_metrics", "is_relevant_chunk", "context_precision", "context_recall",
    "exact_match", "answer_metrics", "faithfulness_lexical", "answer_relevancy_lexical",
    "TAU_CONTEXT_RELEVANT", "TAU_SENTENCE_COVERED", "TAU_CLAIM_SUPPORTED",
]
