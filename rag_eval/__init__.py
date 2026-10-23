"""智能座舱 RAG 能力评估 —— 真实评测引擎。

对外主入口：
    from rag_eval import RunConfig, Evaluator, load_documents_from, load_eval_set

流水线：语料加载 → 边界感知分块 → 真实向量化 → FAISS + BM25 索引
        → 混合检索 → 重排 → 真实答案生成 → 真实指标计算 → 聚合报告
"""

from __future__ import annotations

from .config import (
    CACHE_DIR, DATA_DIR, EMBED_API_MODELS, EMBED_BACKENDS, LLM_PRESETS, PROJECT_ROOT,
    RERANK_BACKENDS, SAMPLE_CORPUS_PATH, SAMPLE_EVAL_PATH,
    RunConfig, capability_report,
)
from .embedder import Embedder, EmbeddingError
from .evaluate import (
    EvalError, EvalItem, EvalRecord, EvalReport, Evaluator,
    load_documents_from, load_eval_set,
)
from .generate import (
    AnswerGenerator, CHAT_SYSTEM_PROMPT, ContextCompressor, GeneratedAnswer,
    build_retrieval_query, looks_like_followup,
)
from .index import BM25Index, RetrievalIndex, VectorIndex, build_index
from .ingest import (
    LOADERS, SUPPORTED_EXTENSIONS, Chunk, Document, IngestError, RecursiveChunker, load_any,
)
from .kb import KB_DIR, ImportResult, KnowledgeBase
from .llm import LLMClient, LLMError
from .metrics import LLMJudge, MetricValue
from .retrieve import Reranker, RetrievedChunk, Retriever
from .security import SecurityError, resolve_upload_path, safe_upload_filename, validate_public_http_url
from .webimport import (
    ENGINE_CHAIN, SearchHit, WebDoc, WebImportError,
    fetch_url, is_direct_url, search_and_fetch, web_search,
)

__version__ = "2.5.0"

__all__ = [
    "__version__",
    "RunConfig", "capability_report", "LLM_PRESETS", "EMBED_API_MODELS", "EMBED_BACKENDS",
    "RERANK_BACKENDS",
    "DATA_DIR", "CACHE_DIR", "PROJECT_ROOT", "SAMPLE_CORPUS_PATH", "SAMPLE_EVAL_PATH",
    "Evaluator", "EvalItem", "EvalRecord", "EvalReport", "EvalError",
    "load_documents_from", "load_eval_set",
    "Document", "Chunk", "RecursiveChunker", "IngestError",
    "Embedder", "EmbeddingError",
    "VectorIndex", "BM25Index", "RetrievalIndex", "build_index",
    "Retriever", "Reranker", "RetrievedChunk",
    "AnswerGenerator", "GeneratedAnswer", "ContextCompressor",
    "build_retrieval_query", "looks_like_followup", "CHAT_SYSTEM_PROMPT",
    "LLMClient", "LLMError", "LLMJudge", "MetricValue",
    # 知识库与联网导入
    "KnowledgeBase", "ImportResult", "KB_DIR",
    "SUPPORTED_EXTENSIONS", "LOADERS", "load_any",
    "WebDoc", "SearchHit", "WebImportError",
    "fetch_url", "web_search", "search_and_fetch", "is_direct_url", "ENGINE_CHAIN",
    "SecurityError", "resolve_upload_path", "safe_upload_filename", "validate_public_http_url",
]

