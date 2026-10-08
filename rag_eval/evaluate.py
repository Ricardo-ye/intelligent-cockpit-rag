"""评测编排器：把「检索 -> 生成 -> 指标」串成一条真实可跑的流水线。

对每个评测样本执行：
    query → 混合检索（真实） → 可选压缩 → 答案生成（真实 LLM 或抽取式）
          → 确定性指标 + 可选 LLM 判官指标 → 逐题记录
最后对全体样本做真实聚合（逐指标求均值，忽略未计算出的指标，不填充假值）。
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from .config import RunConfig, capability_report
from .embedder import Embedder
from .generate import AnswerGenerator
from .index import RetrievalIndex, build_index
from .ingest import Chunk, Document, IngestError, RecursiveChunker, load_json_corpus, load_pdf, load_txt
from .llm import LLMClient, LLMError
from .metrics import (
    LLMJudge, answer_metrics, answer_relevancy_lexical,
    context_precision, context_recall, faithfulness_lexical, retrieval_metrics,
)
from .retrieve import Retriever

ProgressCb = Optional[Callable[[float, str], None]]


class EvalError(RuntimeError):
    pass


# ─────────────────────────────────────────────
# 数据结构
# ─────────────────────────────────────────────
@dataclass
class EvalItem:
    question: str
    ground_truth: str
    relevant_doc_ids: List[str] = field(default_factory=list)
    item_id: str = ""


@dataclass
class EvalRecord:
    """单题真实运行结果。"""

    index: int
    item: EvalItem
    answer: str
    generation_mode: str
    retrieved: List[Dict[str, Any]] = field(default_factory=list)
    generation_contexts: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, Optional[float]] = field(default_factory=dict)
    metric_meta: Dict[str, Dict[str, str]] = field(default_factory=dict)
    timings: Dict[str, float] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "question": self.item.question,
            "ground_truth": self.item.ground_truth,
            "relevant_doc_ids": self.item.relevant_doc_ids,
            "answer": self.answer,
            "generation_mode": self.generation_mode,
            "retrieved": self.retrieved,
            "generation_contexts": self.generation_contexts,
            "metrics": self.metrics,
            "metric_meta": self.metric_meta,
            "timings": self.timings,
            "error": self.error,
        }


@dataclass
class EvalReport:
    config: Dict[str, Any]              # 实际生效的配置（可能因降级而与请求值不同）
    capabilities: Dict[str, Any]
    aggregates: Dict[str, Optional[float]]
    metric_meta: Dict[str, Dict[str, str]]
    metric_breakdown: Dict[str, Dict[str, Dict[str, Any]]] = field(default_factory=dict)
    config_requested: Dict[str, Any] = field(default_factory=dict)  # 用户原始请求的配置
    records: List[EvalRecord] = field(default_factory=list)
    build_stats: Dict[str, Any] = field(default_factory=dict)
    runtime: Dict[str, Any] = field(default_factory=dict)
    started_at: str = ""
    finished_at: str = ""
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": self.config,
            "config_requested": self.config_requested,
            "capabilities": self.capabilities,
            "aggregates": self.aggregates,
            "metric_meta": self.metric_meta,
            "metric_breakdown": self.metric_breakdown,
            "build_stats": self.build_stats,
            "runtime": self.runtime,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "warnings": self.warnings,
            "records": [r.to_dict() for r in self.records],
        }


# ─────────────────────────────────────────────
# 评测集加载
# ─────────────────────────────────────────────
def load_eval_set(path: str | Path) -> List[EvalItem]:
    """加载评测集：JSON 数组 / JSON 对象 / CSV。"""
    p = Path(path)
    if not p.exists():
        raise EvalError(f"评测集不存在：{p}")

    if p.suffix.lower() == ".csv":
        items: List[EvalItem] = []
        with p.open("r", encoding="utf-8-sig", newline="") as f:
            for i, row in enumerate(csv.DictReader(f), 1):
                q = (row.get("question") or row.get("问题") or "").strip()
                gt = (row.get("ground_truth") or row.get("标准答案") or "").strip()
                if not q:
                    continue
                rel = (row.get("relevant_doc_ids") or row.get("相关文档") or "").strip()
                items.append(
                    EvalItem(
                        question=q, ground_truth=gt,
                        relevant_doc_ids=[s.strip() for s in re_split(rel) if s.strip()],
                        item_id=str(row.get("id") or i),
                    )
                )
        if not items:
            raise EvalError("CSV 评测集中没有解析到有效样本。")
        return items

    raw = json.loads(p.read_text(encoding="utf-8"))
    items = _items_from_obj(raw)
    if not items:
        raise EvalError("JSON 评测集中没有解析到有效样本。")
    return items


def _items_from_obj(raw: Any) -> List[EvalItem]:
    seq = raw.get("items", raw.get("questions", raw)) if isinstance(raw, dict) else raw
    if not isinstance(seq, list):
        return []
    out: List[EvalItem] = []
    for i, obj in enumerate(seq, 1):
        if not isinstance(obj, dict):
            continue
        q = str(obj.get("question") or obj.get("问题") or "").strip()
        gt = str(obj.get("ground_truth") or obj.get("标准答案") or obj.get("answer") or "").strip()
        if not q:
            continue
        rel = obj.get("relevant_doc_ids") or obj.get("相关文档") or []
        if isinstance(rel, str):
            rel = [s.strip() for s in re_split(rel) if s.strip()]
        out.append(
            EvalItem(question=q, ground_truth=gt,
                     relevant_doc_ids=[str(r) for r in rel],
                     item_id=str(obj.get("id") or i))
        )
    return out


def re_split(s: str) -> List[str]:
    for sep in (";", "；", ",", "，", "|"):
        if sep in s:
            return s.split(sep)
    return [s]


# ─────────────────────────────────────────────
# 编排器
# ─────────────────────────────────────────────
class Evaluator:
    def __init__(self, cfg: RunConfig):
        self.cfg = cfg
        self.requested_config = cfg.to_dict()   # 留存用户原始请求，便于对比降级情况
        self.caps = capability_report()
        self.warnings: List[str] = []
        self.index: Optional[RetrievalIndex] = None
        self.retriever: Optional[Retriever] = None
        self.generator: Optional[AnswerGenerator] = None
        self.judge: Optional[LLMJudge] = None
        self.llm: Optional[LLMClient] = None
        self.chunks: List[Chunk] = []
        self.embedder: Optional[Embedder] = None

    # ── 阶段一：建索引 ──
    def prepare(self, docs: Sequence[Document], progress: ProgressCb = None) -> Dict[str, Any]:
        def _p(frac: float, msg: str) -> None:
            if progress:
                progress(frac, msg)

        _p(0.05, f"加载语料：{len(docs)} 篇文档")

        chunker = RecursiveChunker(self.cfg.chunk_size, self.cfg.chunk_overlap, self.cfg.min_chunk_chars)
        self.chunks = chunker.chunk_all(docs)
        _p(0.15, f"分块完成：{len(self.chunks)} 个分块（size={self.cfg.chunk_size}, overlap={self.cfg.chunk_overlap}）")

        # LLM 客户端（仅当配置了真实后端）
        if self.cfg.llm_backend != "offline":
            try:
                self.llm = LLMClient(self.cfg)
                _p(0.20, f"LLM 客户端就绪：{self.cfg.llm_model} @ {self.cfg.resolved_base_url()}")
            except LLMError as exc:
                self.llm = None
                self.warnings.append(f"LLM 不可用，已降级为离线抽取式生成：{exc}")
                _p(0.20, f"⚠ LLM 不可用，降级为离线抽取式：{exc}")

        self.embedder = Embedder(self.cfg)

        # LLM 重排依赖真实 LLM：不可用时显式降级为 BM25 词法重排，
        # 避免"配置了 LLM 重排但没 Key"导致每一题都失败、白白跑完整个评测集。
        if self.cfg.rerank_backend == "llm" and self.llm is None:
            self.warnings.append(
                "LLM 重排不可用（缺少可用的 LLM），已自动回退为 BM25 词法重排。"
                "如需真实 LLM 重排，请配置 API Key。"
            )
            self.cfg.rerank_backend = "bm25"
            _p(0.25, "⚠ LLM 重排不可用，已回退为 BM25 词法重排")

        self.index = build_index(
            docs, self.chunks, self.cfg,
            progress=lambda m: _p(0.35, m),
            embedder=self.embedder,
        )
        self.retriever = Retriever(self.cfg, self.index, self.llm)
        self.generator = AnswerGenerator(self.cfg, self.llm)

        if self.cfg.judge_ready() and self.llm is not None:
            self.judge = LLMJudge(self.llm)
        elif self.cfg.use_llm_judge:
            self.warnings.append("启用 LLM 判官失败：LLM 不可用，忠实度/相关性将使用词法代理指标。")

        _p(0.45, "索引就绪，开始逐题评测")
        return {
            "n_docs": len(docs),
            "n_chunks": len(self.chunks),
            "corpus_digest": self.index.corpus_digest,
            **self.index.build_stats,
        }

    # ── 阶段二：跑评测集 ──
    def run(self, items: Sequence[EvalItem], progress: ProgressCb = None) -> EvalReport:
        if not self.index or not self.retriever or not self.generator:
            raise EvalError("请先调用 prepare() 构建索引。")

        started = time.strftime("%Y-%m-%d %H:%M:%S")
        records: List[EvalRecord] = []
        total = len(items)

        for i, item in enumerate(items, 1):
            frac = 0.45 + 0.55 * (i - 1) / max(total, 1)
            if progress:
                progress(frac, f"[{i}/{total}] 检索：{item.question[:28]}")
            records.append(self._run_one(i, item, progress, frac, total))

        if progress:
            progress(0.99, "聚合指标中…")

        aggregates, meta, breakdown = self._aggregate(records)
        runtime = self._runtime_stats(records)

        if progress:
            progress(1.0, f"评测完成：{total} 题")

        return EvalReport(
            config=self.cfg.to_dict(),
            config_requested=self.requested_config,
            capabilities=self.caps,
            aggregates=aggregates,
            metric_meta=meta,
            metric_breakdown=breakdown,
            records=records,
            build_stats=self.index.build_stats,
            runtime=runtime,
            started_at=started,
            finished_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            warnings=self.warnings,
        )

    # ── 单题执行 ──
    def _run_one(self, idx: int, item: EvalItem, progress, frac: float, total: int) -> EvalRecord:
        assert self.retriever and self.generator and self.index
        timings: Dict[str, float] = {}
        record = EvalRecord(index=idx, item=item, answer="", generation_mode="")

        try:
            t0 = time.perf_counter()
            retrieved = self.retriever.retrieve(item.question)
            timings["retrieve_s"] = round(time.perf_counter() - t0, 3)
            timings.update({
                "retrieve_vector_s": self.retriever.last_stats.vector_latency_s,
                "retrieve_bm25_s": self.retriever.last_stats.bm25_latency_s,
                "rerank_s": self.retriever.last_stats.rerank_latency_s,
            })

            if progress:
                progress(frac + 0.2 / max(total, 1), f"[{idx}/{total}] 生成答案…")

            t0 = time.perf_counter()
            gen = self.generator.generate(item.question, retrieved)
            timings["generate_s"] = round(time.perf_counter() - t0, 3)
            # 上下文压缩在 generate() 内部完成，单独拆出来，避免污染「纯生成」耗时口径
            timings["compress_s"] = round(gen.compress_s, 3)
            timings["generate_net_s"] = round(max(0.0, timings["generate_s"] - gen.compress_s), 3)

            record.answer = gen.answer
            record.generation_mode = gen.mode
            record.retrieved = [rc.to_dict() for rc in retrieved]
            record.generation_contexts = [c.to_dict() for c in gen.used_contexts]
            record.error = gen.error

            # —— 指标 ——
            t0 = time.perf_counter()
            contexts = [rc.chunk for rc in retrieved]
            faith_contexts = gen.used_contexts
            metrics: Dict[str, Optional[float]] = {}
            meta: Dict[str, Dict[str, str]] = {}

            def _put(name: str, value: Optional[float], family: str, method: str, detail: str = "") -> None:
                metrics[name] = None if value is None else round(float(value), 6)
                meta[name] = {"family": family, "method": method, "detail": detail}

            # 检索类（需标注 relevant_doc_ids）
            doc_ids = [rc.chunk.doc_id for rc in retrieved]
            rm = retrieval_metrics(doc_ids, item.relevant_doc_ids, self.cfg.top_k)
            for key, val in rm.items():
                _put(key, val, "retrieval", "deterministic",
                     "基于标注 relevant_doc_ids 的真实 IR 计算" if item.relevant_doc_ids else "无标注，不计算")

            # 上下文类
            _put("context_precision", context_precision(contexts, item.ground_truth, item.relevant_doc_ids),
                 "context", "deterministic", f"rank-weighted，相关性阈值 τ={0.30}")
            _put("context_recall", context_recall(contexts, item.ground_truth),
                 "context", "deterministic", "标准答案句子覆盖比例")

            # 答案类
            if item.ground_truth:
                am = answer_metrics(gen.answer, item.ground_truth)
                for key, val in am.items():
                    _put(key, val, "answer", "deterministic", "与 ground_truth 直接比对")
            else:
                for key in ("exact_match", "token_f1", "rouge_l"):
                    _put(key, None, "answer", "deterministic", "缺少 ground_truth，不计算")

            # 忠实度 / 相关性
            if self.judge is not None:
                f_llm = self.judge.faithfulness(gen.answer, faith_contexts)
                r_llm = self.judge.answer_relevancy(gen.answer, item.question)
                _put("faithfulness", f_llm, "faithfulness", "llm-judge",
                     f"LLM 拆论断后核查，模型 {self.cfg.llm_model}")
                _put("answer_relevancy", r_llm, "relevancy", "llm-judge",
                     f"LLM 反推问题，模型 {self.cfg.llm_model}")
                if f_llm is None:
                    _put("faithfulness", faithfulness_lexical(gen.answer, faith_contexts),
                         "faithfulness", "deterministic", "LLM 判官失败，回退词法覆盖")
                if r_llm is None:
                    _put("answer_relevancy", answer_relevancy_lexical(gen.answer, item.question),
                         "relevancy", "deterministic", "LLM 判官失败，回退词法 F1")
            else:
                _put("faithfulness", faithfulness_lexical(gen.answer, faith_contexts),
                     "faithfulness", "deterministic", "词法覆盖代理指标（未启用 LLM 判官）")
                _put("answer_relevancy", answer_relevancy_lexical(gen.answer, item.question),
                     "relevancy", "deterministic", "答案-问题词法 F1（未启用 LLM 判官）")

            # F1（上下文精确率与召回率的调和平均）
            cp, cr = metrics.get("context_precision"), metrics.get("context_recall")
            f1 = None if (cp is None or cr is None) else (
                0.0 if (cp + cr) == 0 else 2 * cp * cr / (cp + cr)
            )
            _put("context_f1", f1, "context", "deterministic", "ContextPrecision 与 ContextRecall 的调和平均")

            timings["metrics_s"] = round(time.perf_counter() - t0, 3)
            record.metrics = metrics
            record.metric_meta = meta
            record.timings = timings

        except Exception as exc:  # 单题失败不影响整体，但如实记录
            record.error = f"{type(exc).__name__}: {exc}"
            record.timings = timings

        return record

    # ── 聚合 ──
    def _aggregate(self, records: Sequence[EvalRecord]) -> tuple:
        keys: List[str] = []
        meta: Dict[str, Dict[str, str]] = {}
        for r in records:
            for k, m in r.metric_meta.items():
                if k not in meta:
                    meta[k] = m
                    keys.append(k)

        aggregates: Dict[str, Optional[float]] = {}
        breakdown: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for k in keys:
            vals = [r.metrics.get(k) for r in records if isinstance(r.metrics.get(k), (int, float))]
            aggregates[k] = round(sum(vals) / len(vals), 6) if vals else None

            by_method: Dict[str, List[float]] = {}
            for r in records:
                value = r.metrics.get(k)
                if not isinstance(value, (int, float)):
                    continue
                method = r.metric_meta.get(k, {}).get("method", "unknown")
                by_method.setdefault(method, []).append(float(value))
            breakdown[k] = {
                method: {"value": round(sum(group) / len(group), 6), "n": len(group)}
                for method, group in by_method.items()
            }
            if len(by_method) > 1:
                methods = "、".join(f"{name}({len(group)})" for name, group in by_method.items())
                meta[k] = {
                    **meta[k],
                    "method": "mixed",
                    "detail": f"本次聚合包含多种计算口径：{methods}；分口径结果见 metric_breakdown",
                }

        return aggregates, {k: meta[k] for k in keys}, breakdown

    def _runtime_stats(self, records: Sequence[EvalRecord]) -> Dict[str, Any]:
        def _sum(key: str) -> float:
            return round(sum(r.timings.get(key, 0.0) for r in records), 3)

        n_ok = sum(1 for r in records if not r.error)
        stats: Dict[str, Any] = {
            "n_items": len(records),
            "n_success": n_ok,
            "n_failed": len(records) - n_ok,
            "total_retrieve_s": _sum("retrieve_s"),
            "total_generate_s": _sum("generate_s"),
            "total_metrics_s": _sum("metrics_s"),
            "avg_retrieve_ms": round(1000 * _sum("retrieve_s") / max(n_ok, 1), 1),
            # 注意：generate_s 是包含上下文压缩的墙钟时间；纯生成口径看 avg_generate_net_ms
            "avg_generate_ms": round(1000 * _sum("generate_s") / max(n_ok, 1), 1),
            "avg_compress_ms": round(1000 * _sum("compress_s") / max(n_ok, 1), 1),
            "avg_generate_net_ms": round(1000 * _sum("generate_net_s") / max(n_ok, 1), 1),
            "total_compress_s": _sum("compress_s"),
        }

        # 真实 LLM 用量：token 消耗、重试、截断，全部来自本次运行
        if self.llm is not None:
            u = self.llm.usage
            stats["llm"] = {
                "model": self.llm.model,
                "base_url": self.llm.base_url,
                "is_reasoning_model": self.llm.is_reasoning,
                "calls": u.calls,
                "retries": u.retries,
                "truncated_calls": u.truncated_calls,
                "max_tokens_raised": u.max_tokens_raised,
                "prompt_tokens": u.prompt_tokens,
                "completion_tokens": u.completion_tokens,
                "total_tokens": u.total_tokens,
                "llm_latency_s": round(u.latency_s, 3),
            }
        if self.judge is not None:
            stats["judge"] = {
                "judge_calls": self.judge.stats.calls,
                "judge_latency_s": round(self.judge.stats.latency_s, 3),
                "judge_prompt_tokens": self.judge.stats.prompt_tokens,
                "judge_completion_tokens": self.judge.stats.completion_tokens,
                "method": "llm-judge",
            }
        elif self.cfg.use_llm_judge:
            stats["judge"] = {"method": "lexical-fallback",
                              "reason": "LLM 判官不可用，已回退词法代理指标"}
        return stats


def load_documents_from(
    pdf_path: Optional[str] = None,
    corpus_json_path: Optional[str] = None,
    txt_path: Optional[str] = None,
    builtin: bool = False,
) -> List[Document]:
    """按优先级加载知识库语料，返回统一的 Document 列表。"""
    from .config import SAMPLE_CORPUS_PATH

    if pdf_path:
        return load_pdf(pdf_path)
    if corpus_json_path:
        return load_json_corpus(corpus_json_path)
    if txt_path:
        return load_txt(txt_path)
    if builtin:
        if not SAMPLE_CORPUS_PATH.exists():
            raise IngestError(f"内置语料缺失：{SAMPLE_CORPUS_PATH}")
        return load_json_corpus(SAMPLE_CORPUS_PATH)
    raise IngestError("未提供任何知识库来源（PDF / JSON / TXT / 内置样例）。")


__all__ = [
    "Evaluator", "EvalItem", "EvalRecord", "EvalReport", "EvalError",
    "load_eval_set", "load_documents_from",
]
