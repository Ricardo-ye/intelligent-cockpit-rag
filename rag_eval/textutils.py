"""文本处理工具：中文分词、分句、相似度。

统一供 BM25、Embedding、指标计算复用，避免各模块各自实现导致口径不一致。
jieba 缺失时自动降级为字符 n-gram，保证离线可用。
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import List, Sequence

# ─────────────────────────────────────────────
# 分词
# ─────────────────────────────────────────────
_STOPWORDS = set(
    "的 了 和 是 在 我 有 就 不 人 都 一 一个 上 也 很 到 说 要 去 你 会 着 没有 看 好 "
    "自己 这 那 与 及 或 而 但 并 被 把 对 从 为 以 之 其 该 等 如何 怎么 怎样 吗 呢 啊".split()
)

_JIEBA_READY = False


def _ensure_jieba() -> bool:
    global _JIEBA_READY
    if _JIEBA_READY:
        return True
    try:
        import jieba  # type: ignore

        jieba.setLogLevel(60)  # 关闭 jieba 首次运行的日志噪音
        _JIEBA_READY = True
        return True
    except Exception:
        return False


_TOKEN_RE = re.compile(r"[a-zA-Z]+|\d+")
_QUERY_NOISE = set("的了和是在有把被向与及或而但并从为以之其该等什么怎么怎样如何哪些哪个吗呢啊请问一下")


def tokenize(text: str, drop_stopwords: bool = True) -> List[str]:
    """中文分词 + 英文/数字切分 + 去停用词。用于 BM25 与词法相似度。"""
    if not text:
        return []
    tokens: List[str] = []
    if _ensure_jieba():
        import jieba  # type: ignore

        tokens = [t.strip() for t in jieba.lcut(text) if t.strip()]
    else:
        tokens = _TOKEN_RE.findall(text) + list(text)

    tokens = [t for t in tokens if not re.fullmatch(r"\s+", t)]
    if drop_stopwords:
        tokens = [t for t in tokens if t.lower() not in _STOPWORDS and len(t) > 0]
    return tokens


def char_ngrams(text: str, n: int = 2) -> List[str]:
    """字符 n-gram，用于 jieba 不可用时的检索兜底与 F1 计算。"""
    cleaned = re.sub(r"\s+", "", text or "")
    if len(cleaned) < n:
        return [cleaned] if cleaned else []
    return [cleaned[i:i + n] for i in range(len(cleaned) - n + 1)]


# ─────────────────────────────────────────────
# 分句
# ─────────────────────────────────────────────
_SENT_SPLIT_RE = re.compile(r"(?<=[。！？!?；;\n])")


def split_sentences(text: str, min_len: int = 4) -> List[str]:
    """把文本切成句子（保留真实语义边界），过滤过短片段。"""
    if not text:
        return []
    parts: List[str] = []
    for seg in _SENT_SPLIT_RE.split(text):
        for line in seg.split("\n"):
            s = line.strip()
            if len(s) >= min_len:
                parts.append(s)
    return parts or ([text.strip()] if text.strip() else [])


# ─────────────────────────────────────────────
# 相似度 / 重合度
# ─────────────────────────────────────────────
def token_f1(pred: str, gold: str) -> float:
    """SQuAD 风格 token 级 F1（按字符 n-gram + 词法 token 混合计数）。"""
    p_tokens = _counter(pred)
    g_tokens = _counter(gold)
    if not p_tokens or not g_tokens:
        return 0.0
    common = sum((p_tokens & g_tokens).values())
    if common == 0:
        return 0.0
    precision = common / sum(p_tokens.values())
    recall = common / sum(g_tokens.values())
    return round(2 * precision * recall / (precision + recall), 6)


def _counter(text: str) -> Counter:
    """混合 token 计数器：中文用字级 2-gram，英文/数字用词。"""
    c: Counter = Counter()
    c.update(char_ngrams(text, 2))
    c.update(t.lower() for t in _TOKEN_RE.findall(text or ""))
    return c


def overlap_ratio(a: str, b: str) -> float:
    """a 的内容被 b 覆盖的比例（用于判定某句/某块是否被支撑）。"""
    a_tokens = _counter(a)
    if not a_tokens:
        return 0.0
    b_tokens = _counter(b)
    common = sum((a_tokens & b_tokens).values())
    return common / sum(a_tokens.values())


def query_coverage(query: str, text: str) -> float:
    """计算正文对问题关键信号的覆盖率（0~1）。

    混合检索的 min-max 分数只反映同一批候选的相对名次：即使所有候选都无关，
    第一名也可能是 1。因此这里独立使用中文二元组及英文/数字词做“能否支撑问题”
    的下限门控。疑问词、助词不会计入分母，避免“什么/怎么”制造伪相关。
    """
    q = (query or "").lower()
    body = (text or "").lower()
    latin = {t for t in _TOKEN_RE.findall(q) if len(t) > 1}
    cjk = "".join(ch for ch in q if "\u4e00" <= ch <= "\u9fff" and ch not in _QUERY_NOISE)
    signals = set(latin)
    signals.update(cjk[i:i + 2] for i in range(len(cjk) - 1))
    if not signals:
        return 1.0
    matched = sum(1 for signal in signals if signal in body)
    return matched / len(signals)


def lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    """最长公共子序列长度（用于 ROUGE-L）。"""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        for j in range(1, len(b) + 1):
            cur[j] = prev[j - 1] + 1 if a[i - 1] == b[j - 1] else max(prev[j], cur[j - 1])
        prev = cur
    return prev[-1]


def rouge_l(pred: str, gold: str) -> float:
    """ROUGE-L F1。"""
    p, g = tokenize(pred, drop_stopwords=False), tokenize(gold, drop_stopwords=False)
    if not p or not g:
        return 0.0
    lcs = lcs_length(p, g)
    if lcs == 0:
        return 0.0
    precision, recall = lcs / len(p), lcs / len(g)
    return round(2 * precision * recall / (precision + recall), 6)


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """余弦相似度（输入若已 L2 归一化则等价于点积）。"""
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def minmax_normalize(scores: Sequence[float]) -> List[float]:
    """把一组分数线性归一化到 [0,1]，用于混合检索的分数融合。"""
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-12:
        return [1.0] * len(scores)
    return [(s - lo) / (hi - lo) for s in scores]


def truncate(text: str, max_chars: int) -> str:
    return text if len(text) <= max_chars else text[:max_chars] + "…"


__all__ = [
    "tokenize", "char_ngrams", "split_sentences", "token_f1", "overlap_ratio", "query_coverage",
    "lcs_length", "rouge_l", "cosine", "minmax_normalize", "truncate",
]

