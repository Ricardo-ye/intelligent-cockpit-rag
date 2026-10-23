"""检索层：向量检索 / BM25 检索 / 混合检索（分数融合）+ 真实重排序。

混合检索采用「各自取候选 -> min-max 归一化 -> 加权求和」的经典融合方式，
权重由 bm25_weight 控制，保证可解释、可复现。
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .config import RunConfig
from .index import RetrievalIndex
from .ingest import Chunk
from .llm import LLMClient, LLMError
from .textutils import (
    expand_query_entity_aliases, has_topic_support, minmax_normalize,
    query_coverage, topic_entity_matches, truncate,
)


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float = 0.0
    vector_score: Optional[float] = None
    bm25_score: Optional[float] = None
    rerank_score: Optional[float] = None
    query_coverage: Optional[float] = None
    matched_entities: List[str] = field(default_factory=list)
    rank: int = 0
    source: str = "vector"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk.chunk_id,
            "doc_id": self.chunk.doc_id,
            "doc_title": self.chunk.doc_title,
            "page": self.chunk.page,
            "rank": self.rank,
            "score": round(self.score, 6),
            "vector_score": None if self.vector_score is None else round(self.vector_score, 6),
            "bm25_score": None if self.bm25_score is None else round(self.bm25_score, 6),
            "rerank_score": None if self.rerank_score is None else round(self.rerank_score, 6),
            "query_coverage": None if self.query_coverage is None else round(self.query_coverage, 6),
            "matched_entities": self.matched_entities,
            "source": self.source,
            "text": self.chunk.text,
        }


@dataclass
class RetrieveStats:
    """单次检索的真实耗时与候选规模。"""

    vector_latency_s: float = 0.0
    bm25_latency_s: float = 0.0
    fusion_latency_s: float = 0.0
    rerank_latency_s: float = 0.0
    candidates: int = 0
    final: int = 0
    filtered_low_support: int = 0
    rerank_backend: str = "none"


# ─────────────────────────────────────────────
# 重排序
# ─────────────────────────────────────────────
class Reranker:
    """重排序器：none / bm25（词法）/ llm（真实 LLM 判官打分）。"""

    def __init__(self, cfg: RunConfig, index: RetrievalIndex, llm: Optional[LLMClient] = None):
        self.cfg = cfg
        self.index = index
        self.llm = llm

    def rerank(self, query: str, cands: List[RetrievedChunk], top_k: int) -> List[RetrievedChunk]:
        backend = self.cfg.rerank_backend
        if backend == "none" or not cands:
            return cands[:top_k]

        if backend == "bm25":
            rows = [self.index.chunks.index(c.chunk) for c in cands]
            scored = self.index.lexical_rescore(query, rows)
            order = {r: s for r, s in scored}
            for c, r in zip(cands, rows):
                c.rerank_score = order.get(r)
            cands.sort(key=lambda c: -(c.rerank_score if c.rerank_score is not None else -1e9))
            return cands[:top_k]

        if backend == "llm":
            if self.llm is None or not self.llm.available:
                raise LLMError("LLM 重排需要可用的 LLM 客户端，请配置 API Key 或改用 bm25/none 重排。")
            scores = self._llm_scores(query, cands)
            for c, s in zip(cands, scores):
                c.rerank_score = s
            cands.sort(key=lambda c: -(c.rerank_score or 0.0))
            return cands[:top_k]

        raise ValueError(f"未知重排后端：{backend}")

    def _llm_scores(self, query: str, cands: List[RetrievedChunk]) -> List[float]:
        """让 LLM 对每个候选片段给出 0-10 相关性分（真实调用，解析失败即报错）。"""
        numbered = "\n\n".join(
            f"[{i}] {truncate(c.chunk.text, 400)}" for i, c in enumerate(cands, 1)
        )
        system = (
            "你是检索相关性评审员。请针对用户问题，为每个候选片段打 0-10 的相关性分数，"
            "10 表示直接包含回答问题所需信息，0 表示完全无关。"
            '只输出 JSON：{"scores":[{"id":1,"score":8}, ...]}，不要输出解释。'
        )
        user = f"用户问题：{query}\n\n候选片段：\n{numbered}"
        data = self.llm.chat_json(system, user, max_tokens=64 + 40 * len(cands))  # type: ignore[union-attr]
        raw = data.get("scores", data) if isinstance(data, dict) else data
        mapping: Dict[int, float] = {}
        for item in raw if isinstance(raw, list) else []:
            try:
                mapping[int(item["id"])] = float(item["score"])
            except Exception:
                continue
        if not mapping:
            raise LLMError(f"LLM 重排返回内容无法解析：{truncate(str(data), 200)}")
        return [mapping.get(i, 0.0) for i in range(1, len(cands) + 1)]


# ─────────────────────────────────────────────
# 检索器
# ─────────────────────────────────────────────
class Retriever:
    """组合向量 / BM25 / 混合检索，并可选重排。"""

    def __init__(self, cfg: RunConfig, index: RetrievalIndex, llm: Optional[LLMClient] = None):
        self.cfg = cfg
        self.index = index
        self.reranker = Reranker(cfg, index, llm)
        self.last_stats = RetrieveStats(rerank_backend=cfg.rerank_backend)

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[RetrievedChunk]:
        k = top_k or self.cfg.top_k
        # 只在召回/重排阶段补充“ACC → 自适应巡航”等价词。原问题仍用于界面展示，
        # 生成器也只依据真实手册证据作答，避免查询扩展变成未经证实的回答内容。
        search_query = expand_query_entity_aliases(query)
        stats = RetrieveStats(rerank_backend=self.cfg.rerank_backend)
        # 主题门控也需要看到比最终 Top-K 更大的候选池：若先截断成前 5 个
        # 泛“安全”网页，再做过滤，真正的安全带手册即使排第 6 也没有机会入选。
        # 因此无论是否开启重排，均先扩大召回，再按实体过滤，最后才截断。
        candidate_k = max(k * 4, 20)

        vec_pairs: List[tuple] = []
        bm25_pairs: List[tuple] = []

        if self.cfg.use_hybrid:
            pool = max(candidate_k, k * 4, 20)
            t = time.perf_counter()
            vec_pairs = self.index.vector_search(search_query, pool)
            stats.vector_latency_s = round(time.perf_counter() - t, 3)
            t = time.perf_counter()
            bm25_pairs = self.index.bm25_search(search_query, pool)
            stats.bm25_latency_s = round(time.perf_counter() - t, 3)
        else:
            t = time.perf_counter()
            vec_pairs = self.index.vector_search(search_query, candidate_k)
            stats.vector_latency_s = round(time.perf_counter() - t, 3)

        t = time.perf_counter()
        cands = self._fuse(vec_pairs, bm25_pairs, candidate_k)
        stats.fusion_latency_s = round(time.perf_counter() - t, 3)
        stats.candidates = len(vec_pairs) + len(bm25_pairs)

        # 先过滤不具备实体支撑的候选，再重排；重排器不应浪费预算比较已确定无关的网页。
        cands, dropped = self._filter_low_support(search_query, cands)
        stats.filtered_low_support = dropped

        # 重排
        if self.cfg.rerank_backend != "none":
            t = time.perf_counter()
            # ``rerank_top_k`` 是重排后保留的候选池，不应小于最终展示的两倍；
            # 否则相邻的“条件页”会在最终 Top-K 前被过早丢弃，只留下泛说明页。
            rerank_pool = max(k * 2, self.cfg.rerank_top_k)
            cands = self.reranker.rerank(search_query, cands, rerank_pool)
            stats.rerank_latency_s = round(time.perf_counter() - t, 3)
        else:
            cands = cands[:k]

        cands = self._prioritize_safety_boundary_evidence(search_query, cands)
        cands = self._add_adjacent_page_context(search_query, cands)
        cands = self._limit_per_document(cands, k)

        for i, c in enumerate(cands, 1):
            c.rank = i
        stats.final = len(cands)
        self.last_stats = stats
        return cands

    @staticmethod
    def _prioritize_safety_boundary_evidence(query: str, cands: List[RetrievedChunk]) -> List[RetrievedChunk]:
        """限制/禁用类问题优先保留明确的边界条件原文。

        这是检索排序规则而非回答规则：只有用户明确问“哪些情况、不适合、何时、请勿”
        时才触发，且证据必须出现“无法正常工作、请勿开启、恶劣天气”等原文表达。
        """
        if not re.search(r"(?:情况|不适合|何时|请勿)", query):
            return cands
        boundary = re.compile(r"(?:无法正常工作|请勿开启|恶劣天气|特殊或复杂路况)")
        return sorted(
            cands,
            key=lambda c: (
                1 if boundary.search(c.chunk.text) else 0,
                c.rerank_score if c.rerank_score is not None else c.score,
            ),
            reverse=True,
        )

    def _add_adjacent_page_context(self, query: str, cands: List[RetrievedChunk]) -> List[RetrievedChunk]:
        """为限制类问题保留命中块的同页下一分块，恢复被 chunk_size 切断的规则列表。"""
        if not re.search(r"(?:情况|不适合|何时|请勿)", query):
            return cands
        boundary = re.compile(r"(?:无法正常工作|请勿开启|恶劣天气|特殊或复杂路况)")
        expanded: List[RetrievedChunk] = []
        seen = set()
        for candidate in cands:
            if candidate.chunk.chunk_id not in seen:
                expanded.append(candidate)
                seen.add(candidate.chunk.chunk_id)
            if not boundary.search(candidate.chunk.text):
                continue
            row = self.index.chunks.index(candidate.chunk)
            if row + 1 >= len(self.index.chunks):
                continue
            neighbor = self.index.chunks[row + 1]
            if (neighbor.doc_id != candidate.chunk.doc_id or neighbor.page != candidate.chunk.page
                    or neighbor.chunk_id in seen):
                continue
            neighbor_text = f"{neighbor.doc_title}\n{neighbor.text}"
            expanded.append(RetrievedChunk(
                chunk=neighbor,
                score=max(0.0, candidate.score - 0.0001),
                query_coverage=query_coverage(query, neighbor_text),
                matched_entities=topic_entity_matches(query, neighbor_text),
                source=f"{candidate.source}+adjacent-page",
            ))
            seen.add(neighbor.chunk_id)
        return expanded

    def _filter_low_support(self, query: str, cands: List[RetrievedChunk]) -> tuple[List[RetrievedChunk], int]:
        threshold = float(self.cfg.min_query_coverage)
        kept: List[RetrievedChunk] = []
        for c in cands:
            candidate_text = f"{c.chunk.doc_title}\n{c.chunk.text}"
            coverage = query_coverage(query, candidate_text)
            c.query_coverage = coverage
            c.matched_entities = topic_entity_matches(query, candidate_text)
            # 即使用户将覆盖率阈值设为 0，也不能关闭“主题实体必须命中”这条
            # 安全底线；0 只表示关闭比例门槛，不能允许无关主题进入生成器。
            if coverage >= threshold and has_topic_support(query, candidate_text):
                kept.append(c)
        return kept, len(cands) - len(kept)

    def _limit_per_document(self, cands: List[RetrievedChunk], k: int) -> List[RetrievedChunk]:
        """保留排序的同时限制单篇文档占位，给不同来源留下互证空间。"""
        seen: Dict[str, int] = {}
        out: List[RetrievedChunk] = []
        for c in cands:
            n = seen.get(c.chunk.doc_id, 0)
            if n >= self.cfg.max_chunks_per_doc:
                continue
            seen[c.chunk.doc_id] = n + 1
            out.append(c)
            if len(out) >= k:
                break
        return out

    def _fuse(self, vec_pairs, bm25_pairs, k: int) -> List[RetrievedChunk]:
        """分数融合。仅向量/仅 BM25/混合三种路径。"""
        chunks = self.index.chunks

        if not self.cfg.use_hybrid:
            return [
                RetrievedChunk(chunk=chunks[r], score=s, vector_score=s, source="vector")
                for r, s in vec_pairs
            ]

        w_bm25 = float(self.cfg.bm25_weight)
        w_vec = 1.0 - w_bm25

        vec_norm = dict(zip([r for r, _ in vec_pairs], minmax_normalize([s for _, s in vec_pairs])))
        bm25_norm = dict(zip([r for r, _ in bm25_pairs], minmax_normalize([s for _, s in bm25_pairs])))
        vec_raw = dict(vec_pairs)
        bm25_raw = dict(bm25_pairs)

        rows = list(dict.fromkeys([r for r, _ in vec_pairs] + [r for r, _ in bm25_pairs]))
        fused: List[RetrievedChunk] = []
        for r in rows:
            score = w_vec * vec_norm.get(r, 0.0) + w_bm25 * bm25_norm.get(r, 0.0)
            fused.append(
                RetrievedChunk(
                    chunk=chunks[r],
                    score=score,
                    vector_score=vec_raw.get(r),
                    bm25_score=bm25_raw.get(r),
                    source="hybrid",
                )
            )
        fused.sort(key=lambda c: -c.score)
        return fused[:k]


__all__ = ["Retriever", "Reranker", "RetrievedChunk", "RetrieveStats"]
