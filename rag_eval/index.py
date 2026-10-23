"""索引层：FAISS 向量索引 + BM25 词法索引 + 磁盘缓存。

- 向量索引：优先 FAISS `IndexFlatIP`（向量已 L2 归一化，内积即余弦）；
  未安装 faiss 时自动退化为 numpy 精确检索，结果一致只是速度略慢。
- BM25：rank_bm25 的 BM25Okapi，按 jieba 分词建索引。
- 缓存：以「语料指纹 + 向量化参数签名」为键落盘，参数一变即自动失效，
  杜绝「改了 chunk_size 却复用旧向量」这类静默错误。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import CACHE_DIR, RunConfig
from .embedder import Embedder
from .ingest import Chunk, corpus_hash
from .textutils import tokenize


class IndexError_(RuntimeError):
    """索引构建/检索失败。"""


CACHE_SCHEMA_VERSION = "v3"


# ─────────────────────────────────────────────
# 向量索引
# ─────────────────────────────────────────────
class VectorIndex:
    """稠密向量索引（FAISS 优先，numpy 兜底）。"""

    def __init__(self, dim: int):
        self.dim = int(dim)
        self._index = None
        self._matrix: Optional[np.ndarray] = None
        self.backend = "numpy"

    def build(self, vectors: np.ndarray) -> "VectorIndex":
        if vectors.ndim != 2:
            raise IndexError_("向量矩阵必须是二维。")
        if vectors.shape[1] != self.dim:
            raise IndexError_(f"向量维度不一致：期望 {self.dim}，实际 {vectors.shape[1]}")
        try:
            import faiss  # type: ignore

            index = faiss.IndexFlatIP(self.dim)
            index.add(vectors.astype(np.float32))
            self._index, self.backend = index, "faiss"
        except Exception:
            self._matrix, self.backend = vectors.astype(np.float32), "numpy"
        return self

    def search(self, query_vec: np.ndarray, k: int) -> List[Tuple[int, float]]:
        """返回 [(行号, 相似度)]，按相似度降序、真实计算。"""
        if k <= 0:
            return []
        q = np.asarray(query_vec, dtype=np.float32).reshape(1, -1)
        # 归一化以兼容未归一化的查询向量
        n = float(np.linalg.norm(q))
        if n > 0:
            q = q / n

        if self._index is not None:
            sims, idxs = self._index.search(q, min(k, self._index.ntotal))
            return [(int(i), float(s)) for i, s in zip(idxs[0], sims[0]) if i >= 0]

        if self._matrix is None or len(self._matrix) == 0:
            return []
        sims = self._matrix @ q[0]
        order = np.argsort(-sims)[: min(k, len(sims))]
        return [(int(i), float(sims[i])) for i in order]


# ─────────────────────────────────────────────
# BM25 索引
# ─────────────────────────────────────────────
class BM25Index:
    """词法索引（BM25Okapi）。"""

    def __init__(self):
        self._bm25 = None
        self._tokenized: List[List[str]] = []

    def build(self, texts: Sequence[str]) -> "BM25Index":
        try:
            from rank_bm25 import BM25Okapi  # type: ignore
        except Exception as exc:
            raise IndexError_(
                f"缺少 rank_bm25，无法启用混合检索：{exc}。请执行 pip install rank_bm25。"
            ) from exc
        self._tokenized = [tokenize(t) for t in texts]
        # 全空语料会让 BM25 除零，这里塞一个占位 token 保证数值稳定
        safe = [t if t else ["空"] for t in self._tokenized]
        self._bm25 = BM25Okapi(safe)
        return self

    def search(self, query: str, k: int) -> List[Tuple[int, float]]:
        if self._bm25 is None or k <= 0:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        order = np.argsort(-np.asarray(scores))[: min(k, len(scores))]
        return [(int(i), float(scores[i])) for i in order]


# ─────────────────────────────────────────────
# 缓存
# ─────────────────────────────────────────────
@dataclass
class CacheInfo:
    hit: bool
    path: Optional[str]
    key: str


def cache_key(cfg: RunConfig, corpus_digest: str) -> str:
    return f"{CACHE_SCHEMA_VERSION}__{corpus_digest}__{cfg.signature()}".replace("|", "_").replace("/", "_")


def load_cached_vectors(
    cfg: RunConfig,
    corpus_digest: str,
    n_chunks: int,
    expected_dim: Optional[int] = None,
) -> Optional[np.ndarray]:
    """命中缓存则返回向量矩阵，否则 None。会校验形状以防脏缓存。"""
    if not cfg.use_cache:
        return None
    p = CACHE_DIR / f"{cache_key(cfg, corpus_digest)}.npy"
    if not p.exists():
        return None
    try:
        arr = np.load(p)
        if arr.ndim != 2:
            return None
        dimension_matches = expected_dim is None or arr.shape[1] == expected_dim
        if arr.shape[0] == n_chunks and dimension_matches:
            return arr
    except Exception:
        pass
    return None


def save_cached_vectors(cfg: RunConfig, corpus_digest: str, vectors: np.ndarray) -> Optional[Path]:
    if not cfg.use_cache:
        return None
    try:
        p = CACHE_DIR / f"{cache_key(cfg, corpus_digest)}.npy"
        np.save(p, vectors.astype(np.float32))
        (CACHE_DIR / f"{cache_key(cfg, corpus_digest)}.meta.json").write_text(
            json.dumps(
                {
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "n_chunks": int(vectors.shape[0]),
                    "dim": int(vectors.shape[1]),
                    "corpus_digest": corpus_digest,
                    "signature": cfg.signature(),
                    "cache_schema": CACHE_SCHEMA_VERSION,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return p
    except Exception:
        return None


# ─────────────────────────────────────────────
# 统一构建入口
# ─────────────────────────────────────────────
@dataclass
class RetrievalIndex:
    """一次运行所需的全部索引产物。"""

    chunks: List[Chunk]
    embedder: Embedder
    vectors: np.ndarray
    vector_index: VectorIndex
    bm25_index: Optional[BM25Index]
    corpus_digest: str
    cache_hit: bool
    build_stats: Dict[str, Any]

    def vector_search(self, query: str, k: int) -> List[Tuple[int, float]]:
        return self.vector_index.search(self.embedder.encode_one(query), k)

    def bm25_search(self, query: str, k: int) -> List[Tuple[int, float]]:
        if self.bm25_index is None:
            return []
        return self.bm25_index.search(query, k)

    def lexical_rescore(self, query: str, candidate_rows: Sequence[int]) -> List[Tuple[int, float]]:
        """对候选集用 BM25 重新打分（用于 bm25 重排）。"""
        if self.bm25_index is None or not candidate_rows:
            return [(r, 0.0) for r in candidate_rows]
        from .textutils import tokenize as _tok  # noqa

        scores = self.bm25_index._bm25.get_scores(_tok(query))  # type: ignore[union-attr]
        pairs = [(r, float(scores[r])) for r in candidate_rows]
        pairs.sort(key=lambda x: -x[1])
        return pairs

    def dense_scores(self, query: str, candidate_rows: Sequence[int]) -> Dict[int, float]:
        """返回候选块的稠密相似度（用于混合检索融合）。"""
        qv = self.embedder.encode_one(query)
        out: Dict[int, float] = {}
        for r in candidate_rows:
            out[r] = float(np.dot(self.vectors[r], qv))
        return out


def build_index(
    docs,
    chunks: List[Chunk],
    cfg: RunConfig,
    progress=None,
    embedder: Optional[Embedder] = None,
) -> RetrievalIndex:
    """真实构建索引：向量化 -> FAISS -> BM25，并落盘/命中缓存。"""
    def _p(msg: str) -> None:
        if progress:
            progress(msg)

    digest = corpus_hash(docs)
    stats: Dict[str, Any] = {}

    # 1) 向量化（带缓存）
    t0 = time.perf_counter()
    emb = embedder or Embedder(cfg)
    # 无论是否命中缓存，本地后端都需先拟合词表，以便编码查询。
    # 这也让我们能校验缓存的实际维度，防止算法升级后读到旧向量。
    # 标题是稳定的主题标签。把它纳入索引表示可避免正文切到中段后丢失“这段在讲什么”。
    # Chunk.text 仍保持原文，供引用与生成使用。
    index_texts = [f"{c.doc_title}\n{c.text}" for c in chunks]
    emb.fit(index_texts)
    expected_dim = emb.dim if cfg.embed_backend == "local-tfidf-svd" else None
    cached = load_cached_vectors(cfg, digest, len(chunks), expected_dim=expected_dim)
    if cached is not None:
        cache_hit = True
        vectors = cached
        _p(f"命中向量缓存（{vectors.shape[0]}×{vectors.shape[1]}）")
    else:
        cache_hit = False
        _p(f"向量化中：{emb.describe()['name']}，共 {len(chunks)} 个分块")
        vectors = emb.encode(index_texts)
        save_cached_vectors(cfg, digest, vectors)
    stats["embed_latency_s"] = round(time.perf_counter() - t0, 3)
    stats["embed_backend"] = emb.describe()["name"]
    stats["cache_hit"] = cache_hit

    # 2) FAISS / numpy 向量索引
    t1 = time.perf_counter()
    vindex = VectorIndex(vectors.shape[1]).build(vectors)
    stats["vector_index_backend"] = vindex.backend
    stats["vector_index_latency_s"] = round(time.perf_counter() - t1, 3)
    _p(f"向量索引就绪：{vindex.backend}（{vectors.shape[0]} 条）")

    # 3) BM25
    bm25 = None
    if cfg.use_hybrid or cfg.rerank_backend == "bm25":
        t2 = time.perf_counter()
        bm25 = BM25Index().build(index_texts)
        stats["bm25_latency_s"] = round(time.perf_counter() - t2, 3)
        _p(f"BM25 索引就绪（{len(chunks)} 条）")
    stats["index_total_latency_s"] = round(stats["embed_latency_s"] + stats["vector_index_latency_s"], 3)

    return RetrievalIndex(
        chunks=chunks, embedder=emb, vectors=vectors,
        vector_index=vindex, bm25_index=bm25,
        corpus_digest=digest, cache_hit=cache_hit, build_stats=stats,
    )


__all__ = [
    "VectorIndex", "BM25Index", "RetrievalIndex", "IndexError_",
    "build_index", "cache_key", "load_cached_vectors", "save_cached_vectors",
]

