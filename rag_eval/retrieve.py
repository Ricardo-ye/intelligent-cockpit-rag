"""检索层：向量检索 / BM25 检索 / 混合检索（分数融合）+ 真实重排序。

混合检索采用「各自取候选 -> min-max 归一化 -> 加权求和」的经典融合方式，
权重由 bm25_weight 控制，保证可解释、可复现。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .config import RunConfig
from .index import RetrievalIndex
from .ingest import Chunk
from .llm import LLMClient, LLMError
from .textutils import minmax_normalize, truncate


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float = 0.0
    vector_score: Optional[float] = None
    bm25_score: Optional[float] = None
    rerank_score: Optional[float] = None
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
        stats = RetrieveStats(rerank_backend=self.cfg.rerank_backend)

        vec_pairs: List[tuple] = []
        bm25_pairs: List[tuple] = []

        if self.cfg.use_hybrid:
            pool = max(k * 4, 20)
            t = time.perf_counter()
            vec_pairs = self.index.vector_search(query, pool)
            stats.vector_latency_s = round(time.perf_counter() - t, 3)
            t = time.perf_counter()
            bm25_pairs = self.index.bm25_search(query, pool)
            stats.bm25_latency_s = round(time.perf_counter() - t, 3)
        else:
            t = time.perf_counter()
            vec_pairs = self.index.vector_search(query, k)
            stats.vector_latency_s = round(time.perf_counter() - t, 3)

        t = time.perf_counter()
        cands = self._fuse(vec_pairs, bm25_pairs, k)
        stats.fusion_latency_s = round(time.perf_counter() - t, 3)
        stats.candidates = len(vec_pairs) + len(bm25_pairs)

        # 重排
        if self.cfg.rerank_backend != "none":
            t = time.perf_counter()
            cands = self.reranker.rerank(query, cands, self.cfg.rerank_top_k)
            stats.rerank_latency_s = round(time.perf_counter() - t, 3)
        else:
            cands = cands[:k]

        for i, c in enumerate(cands, 1):
            c.rank = i
        stats.final = len(cands)
        self.last_stats = stats
        return cands

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
