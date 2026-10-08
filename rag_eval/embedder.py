"""向量化器：把文本变成真实向量（稠密）。

两种后端：
1. ``local-tfidf-svd`` —— 在本机用 TF-IDF + TruncatedSVD（LSA）拟合出稠密语义向量，
   完全离线、结果确定可复现、可跑 FAISS。
2. ``api`` —— 调用 OpenAI 兼容的 Embedding 接口，真实网络请求。

两种后端都产出 L2 归一化向量，因此内积即余弦相似度。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import numpy as np

from .config import RunConfig
from .textutils import tokenize


class EmbeddingError(RuntimeError):
    pass


@dataclass
class EmbedStats:
    backend: str
    dim: int
    fitted_on: int = 0
    api_calls: int = 0
    latency_s: float = 0.0


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


class Embedder:
    """统一的向量化接口（fit 之后 encode）。"""

    def __init__(self, cfg: RunConfig):
        self.cfg = cfg
        self.backend = cfg.embed_backend
        self.dim = int(cfg.embed_dim)
        self._vectorizer = None
        self._svd = None
        self._fitted = False
        self._api_client = None
        self.stats = EmbedStats(backend=self.backend, dim=self.dim)

    # ─────────────────────────────────────
    # 拟合（仅本地后端需要）
    # ─────────────────────────────────────
    def fit(self, corpus_texts: Sequence[str]) -> "Embedder":
        if self.backend == "api":
            self._require_api_client()
            self._fitted = True
            self.stats.fitted_on = len(corpus_texts)
            return self

        if self.backend != "local-tfidf-svd":
            raise EmbeddingError(f"未知向量后端：{self.backend}")

        try:
            from sklearn.decomposition import TruncatedSVD  # type: ignore
            from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
        except Exception as exc:
            raise EmbeddingError(
                f"本地向量化需要 scikit-learn，但它未安装：{exc}。"
                "请执行 pip install scikit-learn，或改用 api 向量后端。"
            ) from exc

        texts = [t if t.strip() else "空" for t in corpus_texts]
        vectorizer = TfidfVectorizer(
            tokenizer=lambda s: tokenize(s, drop_stopwords=True),
            token_pattern=None,
            lowercase=True,
            sublinear_tf=True,
            ngram_range=(1, 2),
            min_df=1,
        )
        tfidf = vectorizer.fit_transform(texts)

        n_samples, n_features = tfidf.shape
        # 极小语料无法做有意义的降维，直接保留 TF-IDF 稠密向量。
        # 这同时避免 n_features=1 时强制 n_components=2 的维度越界。
        if n_samples < 2 or n_features < 2:
            self._vectorizer, self._svd, self._fitted = vectorizer, None, True
            self.dim = int(n_features)
            self.stats.dim = self.dim
            self.stats.fitted_on = n_samples
            self.stats.effective_dim_note = "tiny-corpus-tfidf"  # type: ignore[attr-defined]
            return self

        # 保守限制到样本数与特征数之下，确保不同 sklearn 版本行为一致。
        max_dim = max(1, min(n_samples - 1, n_features - 1))
        dim = int(min(self.dim, max_dim))
        svd = TruncatedSVD(n_components=dim, random_state=self.cfg.seed)
        svd.fit(tfidf)

        self._vectorizer, self._svd, self._fitted = vectorizer, svd, True
        self.dim = dim
        self.stats.dim = dim
        self.stats.fitted_on = n_samples
        self.stats.effective_dim_note = dim  # type: ignore[attr-defined]
        return self

    # ─────────────────────────────────────
    # 编码
    # ─────────────────────────────────────
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not self._fitted:
            raise EmbeddingError("Embedder 尚未 fit，无法 encode。")

        if self.backend == "api":
            return self._encode_api(list(texts))

        tfidf = self._vectorizer.transform([t if t.strip() else "空" for t in texts])  # type: ignore[union-attr]
        dense = (
            self._svd.transform(tfidf)  # type: ignore[union-attr]
            if self._svd is not None
            else tfidf.toarray()
        )
        return _l2_normalize(np.asarray(dense, dtype=np.float32))

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]

    # ─────────────────────────────────────
    # API 后端
    # ─────────────────────────────────────
    def _require_api_client(self) -> None:
        key = self.cfg.embed_api_key()
        base = self.cfg.embed_api_base()
        if not key:
            raise EmbeddingError(
                "api 向量后端需要 OPENAI_API_KEY（或 EMBEDDING_API_KEY）。"
                "DeepSeek 官方不提供 embedding 接口，请改用 local-tfidf-svd，或填入支持 embedding 的兼容服务。"
            )
        if not base:
            raise EmbeddingError("api 向量后端需要 OPENAI_BASE_URL（或 EMBEDDING_BASE_URL）。")
        try:
            from openai import OpenAI  # type: ignore
        except Exception as exc:
            raise EmbeddingError(f"缺少 openai 包：{exc}") from exc
        self._api_client = OpenAI(api_key=key, base_url=base, max_retries=2, timeout=120)
        self._api_model = self.cfg.embed_api_model

    def _encode_api(self, texts: List[str], batch: int = 32) -> np.ndarray:
        import time

        assert self._api_client is not None
        out: List[List[float]] = []
        t0 = time.perf_counter()
        for i in range(0, len(texts), batch):
            chunk = [t if t.strip() else "空" for t in texts[i:i + batch]]
            try:
                resp = self._api_client.embeddings.create(model=self._api_model, input=chunk)
            except Exception as exc:
                raise EmbeddingError(f"Embedding 接口调用失败（{self._api_model}）：{exc}") from exc
            out.extend([d.embedding for d in resp.data])
            self.stats.api_calls += 1
        self.stats.latency_s += time.perf_counter() - t0
        arr = np.asarray(out, dtype=np.float32)
        self.dim = int(arr.shape[1])
        self.stats.dim = self.dim
        return _l2_normalize(arr)

    # ─────────────────────────────────────
    def describe(self) -> Dict[str, Any]:
        if self.backend == "local-tfidf-svd":
            name = f"本地 TF-IDF+SVD（{self.dim}维）"
        else:
            name = f"API Embedding：{getattr(self, '_api_model', self.cfg.embed_api_model)}（{self.dim}维）"
        return {"backend": self.backend, "name": name, "dim": self.dim}


__all__ = ["Embedder", "EmbeddingError", "EmbedStats"]
