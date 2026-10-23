"""问答页推荐题的可回答性过滤。

内置评测题仅能在对应的内置语料存在时展示。对用户知识库，不能把演示题误作
“当前可回答的问题”，否则按钮本身会制造答非所问的 badcase。
"""

from __future__ import annotations

from typing import Iterable, List, Sequence

from .evaluate import EvalItem
from .ingest import Document


def supported_sample_questions(
    items: Sequence[EvalItem], docs: Iterable[Document], *, limit: int = 4,
) -> List[EvalItem]:
    """返回有明确标注资料支撑的内置示例题。

    不尝试通过语义相似度“猜”一篇资料是否能回答推荐题；只有评测项声明的
    ``relevant_doc_ids`` 全部在当前语料中时才展示，宁可少推荐，也不误导用户。
    """
    available = {doc.doc_id for doc in docs}
    return [
        item for item in items
        if item.relevant_doc_ids and set(item.relevant_doc_ids).issubset(available)
    ][:max(0, int(limit))]


__all__ = ["supported_sample_questions"]
