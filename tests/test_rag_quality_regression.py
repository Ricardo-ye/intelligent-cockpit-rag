"""答非所问回归测试：无需 sklearn / 外部模型，可在最小离线环境运行。

覆盖真实故障链：上一轮 ACC → 本轮短的座椅加热独立问题；以及无关问题拒答。
运行：python tests/test_rag_quality_regression.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_eval.config import RunConfig  # noqa: E402
from rag_eval.generate import AnswerGenerator, build_retrieval_query  # noqa: E402
from rag_eval.ingest import Chunk, clean_retrieval_text  # noqa: E402
from rag_eval.kb import assess_document_quality  # noqa: E402
from rag_eval.retrieve import Retriever  # noqa: E402


class _Index:
    """固定候选索引：只模拟索引边界，检验 Retriever 的真实融合后门控逻辑。"""

    def __init__(self) -> None:
        self.chunks = [
            Chunk("acc::c000", "acc", "ACC 自适应巡航", "ACC 通过方向盘按键开启，并设定巡航速度。"),
            Chunk("seat::c000", "seat", "座椅加热与通风", "开启座椅加热有三种常见方式：中央显示屏、手机 App 与实体按键或旋钮。"),
        ]

    def vector_search(self, query: str, k: int):
        return [(0, 0.91), (1, 0.84)][:k]

    def bm25_search(self, query: str, k: int):
        return [(0, 4.2), (1, 3.8)][:k]

    def lexical_rescore(self, query: str, candidate_rows):
        # 座椅块在二阶段词法重排中应胜出，用于检验重排实际看到了候选池。
        return [(row, 9.0 if row == 1 else 1.0) for row in candidate_rows]


def test_independent_question() -> None:
    history = [
        {"role": "user", "content": "ACC 自适应巡航应该怎么开启？"},
        {"role": "assistant", "content": "通过方向盘按键开启并设定速度。"},
    ]
    question = "座椅加热有哪几种开启方式？"
    query, rewritten = build_retrieval_query(question, history, llm=None)
    assert query == question and not rewritten, query

    cfg = RunConfig(llm_backend="offline", min_query_coverage=0.12)
    retrieved = Retriever(cfg, _Index()).retrieve(query)
    assert [item.chunk.doc_id for item in retrieved] == ["seat"], [item.to_dict() for item in retrieved]
    answer = AnswerGenerator(cfg).generate(question, retrieved, history=history, retrieval_query=query)
    assert "三种" in answer.answer and "ACC" not in answer.answer, answer.to_dict()


def test_unrelated_question_abstains() -> None:
    cfg = RunConfig(llm_backend="offline", min_query_coverage=0.12)
    question = "火星基地种植番茄的光照方案是什么？"
    retrieved = Retriever(cfg, _Index()).retrieve(question)
    assert not retrieved, [item.to_dict() for item in retrieved]
    answer = AnswerGenerator(cfg).generate(question, retrieved)
    assert answer.answer == "资料不足，无法回答。", answer.to_dict()


def test_web_boilerplate_is_cleaned_and_flagged() -> None:
    raw = "座椅加热可以通过中控屏开启。\n内容合作\n责任编辑：测试\n座椅加热可以通过中控屏开启。"
    clean = clean_retrieval_text(raw)
    quality = assess_document_quality(raw)
    assert "内容合作" not in clean and clean.count("座椅加热") == 1, clean
    assert quality["warnings"], quality


def test_reranker_receives_a_candidate_pool() -> None:
    """Top-K=1 时，正确块排在向量第二名也必须有机会被重排救回。"""
    cfg = RunConfig(
        llm_backend="offline", top_k=1, use_hybrid=False,
        rerank_backend="bm25", rerank_top_k=1, min_query_coverage=0.0,
    )
    retrieved = Retriever(cfg, _Index()).retrieve("座椅加热怎么开启")
    assert [item.chunk.doc_id for item in retrieved] == ["seat"], [item.to_dict() for item in retrieved]


def main() -> int:
    test_independent_question()
    test_unrelated_question_abstains()
    test_web_boilerplate_is_cleaned_and_flagged()
    test_reranker_receives_a_candidate_pool()
    print("PASS: 独立短问题不串题；无关问题拒答；网页噪声清洗与告警；重排候选池")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

