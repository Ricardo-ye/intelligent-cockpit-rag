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
# 这些短语经常描述的是提问或操作形式，而不是车辆功能本身。若把它们当作
# "主题命中"，一篇 ACC 长文只要出现“开启方式”便可能通过“座椅加热怎么开”的
# 检索门槛，造成答非所问。它们仍会参与 query_coverage 的粗筛，但不能单独证明
# 资料覆盖了问题所问的功能实体。
_GENERIC_TOPIC_SIGNALS = {
    "如何", "怎么", "怎样", "什么", "哪些", "哪个", "哪种", "几种",
    "开启", "关闭", "方式", "方法", "操作", "使用", "功能", "可以",
    "是否", "问题", "原因", "情况", "注意", "需要", "支持", "设置",
}

# 面向智能座驾的功能实体词表。它不是用来回答事实的知识库，而是检索准入的
# 安全护栏：例如“安全带”绝不能只因“安全”“正确”两个泛词就被电池文章放行。
# 后续可由车型/业务词表配置化扩展；这里保留跨小米电车系列通用的功能名称。
_DOMAIN_ENTITY_TERMS = {
    "自动紧急制动", "自适应巡航", "车道保持", "抬头显示", "座椅加热", "座椅通风", "座椅按摩",
    "安全带", "安全气囊", "儿童锁", "胎压监测", "整车ota", "语音助手", "驾驶辅助",
    "自动泊车", "后视镜", "方向盘", "雨刮", "内循环", "外循环", "空调", "导航", "蓝牙",
    "车机", "充电", "磷酸铁锂", "三元锂", "电池", "小米", "su7", "yu7", "acc", "aeb", "hud", "lka",
}

# 用户习惯用缩写询问，但手册通常只使用全称。若只按字面匹配，
# “ACC 在哪些情况下不能使用”会把写有“自适应巡航”的正式章节挡在门外。
# 这里只做检索准入的同义归一，并不把缩写当作事实知识扩写。
_DOMAIN_ENTITY_ALIASES = {
    "acc": ("acc", "自适应巡航"),
    "aeb": ("aeb", "自动紧急制动"),
    "hud": ("hud", "抬头显示"),
    "lka": ("lka", "车道保持"),
    "ota": ("ota", "整车ota"),
}
_GENERIC_ENTITY_TOKENS = _GENERIC_TOPIC_SIGNALS | {
    "应该", "正确", "日常", "相关", "一下", "一个", "一些", "哪些", "什么", "怎么", "如何",
    "有无", "是否", "需要", "能否", "注意事项", "步骤", "开启", "关闭", "使用", "操作",
}


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


def query_topic_signals(query: str) -> List[str]:
    """提取用于“主题实体”校验的查询信号。

    ``query_coverage`` 适合作为宽松的召回下限，但其中的“开启/方式”等通用操作词
    不能证明候选确实在谈同一个功能。本函数只保留更具区分度的中文二元词和英文/
    数字词，供检索层在送入生成模型前作第二道准入校验。
    """
    q = (query or "").lower()
    latin = {t for t in _TOKEN_RE.findall(q) if len(t) > 1}
    # 必须先剔除完整短语，再生成二元词。否则“几种开启方式”会在词边界
    # 生成“种开/启方”，仍可被无关页面的“多种开启方式”误命中。
    cjk_raw = "".join(ch for ch in q if "\u4e00" <= ch <= "\u9fff")
    for term in sorted(_GENERIC_TOPIC_SIGNALS, key=len, reverse=True):
        cjk_raw = cjk_raw.replace(term, " ")
    cjk_parts = [
        "".join(ch for ch in part if ch not in _QUERY_NOISE)
        for part in cjk_raw.split()
    ]
    cjk_signals = {
        part[i:i + 2]
        for part in cjk_parts
        for i in range(len(part) - 1)
    }
    return sorted(latin | (cjk_signals - _GENERIC_TOPIC_SIGNALS))


def query_topic_entities(query: str) -> List[str]:
    """提取问题中的功能实体，用于阻止泛词造成的跨主题召回。

    优先匹配汽车领域的完整功能名；只有没有命中领域词表时才退回分词结果。
    这样“安全带应该怎么正确佩戴”必须命中完整的“安全带”，而不能被“安全”
    “正确”等通用表达误判为有资料支撑。
    """
    q = (query or "").strip().casefold()
    if not q:
        return []
    explicit = sorted({term for term in _DOMAIN_ENTITY_TERMS if term in q}, key=lambda x: (-len(x), x))
    if explicit:
        return explicit
    tokens = {
        token.casefold()
        for token in tokenize(q)
        if len(token.strip()) >= 2 and token.casefold() not in _GENERIC_ENTITY_TOKENS
    }
    # 对未知功能保守兜底：三个字符以上的连续中文比二元词更不易被“安全/正确”碰撞。
    cjk_runs = re.findall(r"[\u4e00-\u9fff]{3,}", q)
    for run in cjk_runs:
        for n in (4, 3):
            tokens.update(run[i:i + n] for i in range(max(0, len(run) - n + 1)))
    return sorted(tokens)


def topic_entity_matches(query: str, text: str) -> List[str]:
    """返回候选正文实际覆盖的查询功能实体，供准入与可观测性共用。"""
    body = (text or "").casefold()
    matched: List[str] = []
    for entity in query_topic_entities(query):
        aliases = _DOMAIN_ENTITY_ALIASES.get(entity, (entity,))
        if any(alias in body for alias in aliases):
            matched.append(entity)
    return matched


def expand_query_entity_aliases(query: str) -> str:
    """为检索补充手册常用全称，原问题仍保留用于展示与生成。

    例如用户输入 ``ACC``，官方手册通常只写“自适应巡航”。此函数仅追加
    等价检索词，不改写用户的意图，也不会把等价词当成回答事实。
    """
    q = (query or "").strip()
    if not q:
        return q
    lower = q.casefold()
    additions: List[str] = []
    for short, aliases in _DOMAIN_ENTITY_ALIASES.items():
        if short in lower:
            additions.extend(alias for alias in aliases[1:] if alias not in q)
    return " ".join([q, *dict.fromkeys(additions)])


def has_topic_support(query: str, text: str) -> bool:
    """候选是否真正命中了问题的功能实体。

    对含多个实体信号的问题，至少要求命中两个，避免只因“座椅”或“开启”之类的
    单个泛词放行；只有一个或两个信号的短问题（如 ``ACC 怎么开启``、``HUD 重影``）
    则要求该信号本身命中。没有可判定实体的模糊独立问题一律不放入生成上下文。
    """
    # 功能实体为第一准入条件；存在明确实体时，必须有完整实体命中。
    # 未识别出领域实体的模糊提问才沿用更保守的双信号兜底。
    entities = query_topic_entities(query)
    if entities:
        return bool(topic_entity_matches(query, text))
    signals = query_topic_signals(query)
    if not signals:
        return False
    body = (text or "").lower()
    return sum(1 for signal in signals if signal in body) >= 2


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
    "query_topic_signals", "query_topic_entities", "topic_entity_matches", "expand_query_entity_aliases", "has_topic_support", "lcs_length", "rouge_l", "cosine", "minmax_normalize", "truncate",
]
