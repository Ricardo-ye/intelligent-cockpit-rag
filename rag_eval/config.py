"""运行配置与环境变量加载。

本模块集中定义：
- 项目路径常量
- 一次评测运行的完整参数（RunConfig）
- LLM / Embedding 后端的可用性探测
- .env 解析（零依赖，不引入 python-dotenv）
"""

from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional

# ─────────────────────────────────────────────
# 路径
# ─────────────────────────────────────────────
PKG_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PKG_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
SAMPLE_CORPUS_PATH = DATA_DIR / "sample_corpus.json"
SAMPLE_EVAL_PATH = DATA_DIR / "sample_eval_set.json"

for _d in (DATA_DIR, CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# .env 加载（极简实现）
# ─────────────────────────────────────────────
def load_dotenv(path: Optional[Path] = None) -> Dict[str, str]:
    """把项目根目录 .env 中的键值注入 os.environ（不覆盖已存在的真实环境变量）。"""
    env_path = path or (PROJECT_ROOT / ".env")
    loaded: Dict[str, str] = {}
    if not env_path.exists():
        return loaded
    for raw in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val
        loaded[key] = val
    return loaded


load_dotenv()


# ─────────────────────────────────────────────
# 后端预设
# ─────────────────────────────────────────────
# OpenAI 兼容协议的服务商预设：base_url + 默认模型 + 对应环境变量名
LLM_PRESETS: Dict[str, Dict[str, Any]] = {
    "deepseek": {
        "label": "DeepSeek（OpenAI 兼容）",
        "base_url": "https://api.deepseek.com/v1",
        "models": ["deepseek-chat", "deepseek-reasoner"],
        "default_model": "deepseek-chat",
        "env_key": "DEEPSEEK_API_KEY",
        "needs_key": True,
        "supports_embedding": False,   # DeepSeek 官方不提供 embedding 接口
    },
    "openai-compatible": {
        "label": "自定义 OpenAI 兼容（中转/自建）",
        "base_url": os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        "models": ["gpt-4o-mini", "gpt-4o", "qwen-max", "glm-4-plus"],
        "default_model": os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        "env_key": "OPENAI_API_KEY",
        "needs_key": True,
        "supports_embedding": True,
    },
    "offline": {
        "label": "离线抽取式（无需任何 Key）",
        "base_url": "",
        "models": ["extractive"],
        "default_model": "extractive",
        "env_key": "",
        "needs_key": False,
        "supports_embedding": False,
    },
}

# 向量后端
EMBED_BACKENDS: Dict[str, Dict[str, Any]] = {
    "local-tfidf-svd": {
        "label": "本地 TF-IDF + SVD 语义向量（离线·真算）",
        "needs_key": False,
        "dims": [64, 128, 256, 512, 1024],
    },
    "api": {
        "label": "OpenAI 兼容 Embedding 接口（需 Key）",
        "needs_key": True,
        "dims": [1024, 1536, 768, 512],
    },
}

# 重排后端
RERANK_BACKENDS: Dict[str, Dict[str, Any]] = {
    "none": {"label": "不使用重排序"},
    "llm": {"label": "LLM 判官重排（DeepSeek/兼容，真调用）"},
    "bm25": {"label": "词法重排（BM25 重新打分，离线）"},
}

EMBED_API_MODELS = ["text-embedding-3-small", "text-embedding-3-large", "text-embedding-v3"]


# ─────────────────────────────────────────────
# 生成预算策略（单一事实来源）
# ─────────────────────────────────────────────
# 思维链模型（reasoner / R1 / o1 之类）会先产出内部推理 token，而这些 token 同样计入
# max_tokens。上限太小会导致推理耗尽预算、finish_reason="length" 而 content 为空——
# 这是非常隐蔽的静默失败，因此对这类模型强制一个 token 下限。
REASONING_MODEL_HINTS: tuple = ("reasoner", "-r1", "r1-", "thinking", "-o1", "-o3", "qwq")
MIN_MAX_TOKENS_REASONING = 2048
DEFAULT_MAX_TOKENS = 1024

# 各模型的推荐 token 预算（未列出的用 DEFAULT_MAX_TOKENS）
MODEL_MAX_TOKENS: Dict[str, int] = {
    "deepseek-chat": 1024,
    "deepseek-reasoner": 4096,
}


def is_reasoning_model(model: str) -> bool:
    m = (model or "").lower()
    return any(h in m for h in REASONING_MODEL_HINTS)


def recommended_max_tokens(model: str) -> int:
    return MODEL_MAX_TOKENS.get(model, DEFAULT_MAX_TOKENS)


# ─────────────────────────────────────────────
# 运行配置
# ─────────────────────────────────────────────
@dataclass
class RunConfig:
    """一次评测运行的全部参数（可完整序列化，用于复现）。"""

    # —— 分块 ——
    chunk_size: int = 300
    chunk_overlap: int = 60
    min_chunk_chars: int = 40

    # —— 向量化 ——
    embed_backend: str = "local-tfidf-svd"   # local-tfidf-svd | api
    embed_dim: int = 256
    embed_api_model: str = "text-embedding-3-small"

    # —— 检索 ——
    top_k: int = 5
    use_hybrid: bool = True
    bm25_weight: float = 0.4                 # 混合检索中 BM25 的权重（0~1）

    # —— 重排 ——
    rerank_backend: str = "none"             # none | llm | bm25
    rerank_top_k: int = 3

    # —— 上下文压缩 ——
    use_compression: bool = False
    max_context_chars: int = 3000

    # —— 生成 ——
    llm_backend: str = "offline"             # deepseek | openai-compatible | offline
    llm_model: str = "extractive"
    llm_base_url: str = ""
    llm_temperature: float = 0.0
    llm_max_tokens: int = DEFAULT_MAX_TOKENS
    llm_max_attempts: int = 3              # 含首次尝试；仅对暂时性错误重试
    llm_retry_base_delay: float = 1.0      # 指数退避基数（秒）

    # —— 评判（LLM 判官指标）——
    use_llm_judge: bool = False              # 需 LLM 可用；faithfulness / answer_relevancy 走 LLM
    judge_request_timeout: int = 90

    # —— 其它 ——
    use_cache: bool = True
    seed: int = 20260924

    def __post_init__(self) -> None:
        """在所有入口统一校验配置，避免无效值进入分块、索引或网络调用。"""
        self.validate()

    def validate(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size 必须大于 0。")
        if self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap 必须满足 0 <= chunk_overlap < chunk_size。")
        if self.min_chunk_chars <= 0 or self.min_chunk_chars > self.chunk_size:
            raise ValueError("min_chunk_chars 必须满足 0 < min_chunk_chars <= chunk_size。")
        if self.embed_backend not in EMBED_BACKENDS:
            raise ValueError(f"未知向量后端：{self.embed_backend}")
        if self.embed_dim <= 0:
            raise ValueError("embed_dim 必须大于 0。")
        if self.top_k <= 0 or self.rerank_top_k <= 0:
            raise ValueError("top_k 与 rerank_top_k 必须大于 0。")
        if not 0.0 <= self.bm25_weight <= 1.0:
            raise ValueError("bm25_weight 必须位于 [0, 1]。")
        if self.rerank_backend not in RERANK_BACKENDS:
            raise ValueError(f"未知重排后端：{self.rerank_backend}")
        if self.max_context_chars <= 0:
            raise ValueError("max_context_chars 必须大于 0。")
        if self.llm_backend not in LLM_PRESETS:
            raise ValueError(f"未知 LLM 后端：{self.llm_backend}")
        if not 0.0 <= self.llm_temperature <= 2.0:
            raise ValueError("llm_temperature 必须位于 [0, 2]。")
        if self.llm_max_tokens <= 0 or self.llm_max_attempts <= 0:
            raise ValueError("llm_max_tokens 与 llm_max_attempts 必须大于 0。")
        if self.llm_retry_base_delay < 0:
            raise ValueError("llm_retry_base_delay 不能为负数。")
        if self.judge_request_timeout <= 0:
            raise ValueError("judge_request_timeout 必须大于 0。")

    def resolved_base_url(self) -> str:
        if self.llm_base_url:
            return self.llm_base_url.rstrip("/")
        preset = LLM_PRESETS.get(self.llm_backend)
        if preset:
            return str(preset.get("base_url") or "").rstrip("/")
        return ""

    def resolved_api_key(self) -> str:
        """按 后端预设环境变量 -> OPENAI_API_KEY -> DEEPSEEK_API_KEY 的顺序取 Key。"""
        preset = LLM_PRESETS.get(self.llm_backend) or {}
        env_key = preset.get("env_key")
        if env_key and os.environ.get(env_key):
            return os.environ[env_key]
        return os.environ.get("OPENAI_API_KEY") or os.environ.get("DEEPSEEK_API_KEY") or ""

    def llm_ready(self) -> bool:
        return self.llm_backend != "offline" and bool(self.resolved_api_key())

    def recommended_max_tokens(self) -> int:
        """当前模型的推荐 token 预算（思维链模型更大）。"""
        return recommended_max_tokens(self.llm_model)

    def is_reasoning(self) -> bool:
        return is_reasoning_model(self.llm_model)

    def judge_ready(self) -> bool:
        return self.use_llm_judge and self.llm_ready()

    def embed_api_key(self) -> str:
        return os.environ.get("OPENAI_API_KEY") or os.environ.get("EMBEDDING_API_KEY") or ""

    def embed_api_base(self) -> str:
        return (os.environ.get("OPENAI_BASE_URL") or os.environ.get("EMBEDDING_BASE_URL") or "").rstrip("/")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def signature(self) -> str:
        """影响索引内容的参数签名，用于缓存键校验。"""
        return (
            f"{self.embed_backend}|{self.embed_dim}|{self.embed_api_model}|"
            f"{self.chunk_size}|{self.chunk_overlap}|{self.min_chunk_chars}"
        )

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RunConfig":
        allowed = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in allowed})


# ─────────────────────────────────────────────
# 能力探测
# ─────────────────────────────────────────────
def capability_report() -> Dict[str, Any]:
    """探测各可选依赖是否就绪，供 UI 与 CLI 展示真实运行能力。"""
    caps: Dict[str, Any] = {}

    def _probe(name: str) -> bool:
        try:
            __import__(name)
            return True
        except Exception:
            return False

    for mod, label in [
        ("pypdf", "PDF 解析"),
        ("jieba", "中文分词"),
        ("rank_bm25", "BM25 检索"),
        ("faiss", "FAISS 向量索引"),
        ("sklearn", "TF-IDF/SVD 向量"),
        ("openai", "OpenAI 兼容客户端"),
        ("streamlit", "Streamlit 运行时"),
        ("pandas", "数据导出"),
    ]:
        caps[label] = _probe(mod)

    key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
    caps["LLM API Key"] = bool(key)
    caps["API Key 来源"] = (
        "DEEPSEEK_API_KEY" if os.environ.get("DEEPSEEK_API_KEY")
        else ("OPENAI_API_KEY" if os.environ.get("OPENAI_API_KEY") else "未配置")
    )
    return caps


__all__ = [
    "PROJECT_ROOT", "DATA_DIR", "CACHE_DIR", "SAMPLE_CORPUS_PATH", "SAMPLE_EVAL_PATH",
    "LLM_PRESETS", "EMBED_BACKENDS", "RERANK_BACKENDS", "EMBED_API_MODELS",
    "MODEL_MAX_TOKENS", "DEFAULT_MAX_TOKENS", "MIN_MAX_TOKENS_REASONING",
    "REASONING_MODEL_HINTS", "is_reasoning_model", "recommended_max_tokens",
    "RunConfig", "capability_report", "load_dotenv",
]
