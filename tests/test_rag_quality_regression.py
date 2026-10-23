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
from rag_eval.textutils import expand_query_entity_aliases, has_topic_support, query_topic_entities  # noqa: E402


class _Index:
    """固定候选索引：只模拟索引边界，检验 Retriever 的真实融合后门控逻辑。"""

    def __init__(self) -> None:
        self.chunks = [
            # 故意带上“开启方式”：旧版仅按覆盖率门控时会把它误判为座椅问题的候选。
            Chunk("acc::c000", "acc", "ACC 自适应巡航", "ACC 可通过方向盘按键开启，有多种开启方式。"),
            Chunk("seat::c000", "seat", "座椅加热与通风", "开启座椅加热有三种常见方式：中央显示屏、手机 App 与实体按键或旋钮。"),
        ]

    def vector_search(self, query: str, k: int):
        return [(0, 0.91), (1, 0.84)][:k]

    def bm25_search(self, query: str, k: int):
        return [(0, 4.2), (1, 3.8)][:k]

    def lexical_rescore(self, query: str, candidate_rows):
        # 座椅块在二阶段词法重排中应胜出，用于检验重排实际看到了候选池。
        return [(row, 9.0 if row == 1 else 1.0) for row in candidate_rows]


class _WideSafetyIndex:
    """正确安全带资料排在初始 Top-5 之后，模拟真实噪声知识库。"""

    def __init__(self) -> None:
        self.chunks = [
            Chunk(f"noise::{i}", f"noise-{i}", "泛安全网页", "电池安全与正确使用的参考说明。")
            for i in range(5)
        ] + [
            Chunk("belt::c000", "belt", "安全带使用手册", "安全带肩带应跨过肩部中央，腰带应贴紧髋部。"),
        ]

    def vector_search(self, query: str, k: int):
        return [(i, 0.99 - i * 0.01) for i in range(min(k, len(self.chunks)))]

    def bm25_search(self, query: str, k: int):
        return self.vector_search(query, k)

    def lexical_rescore(self, query: str, candidate_rows):
        return [(row, 1.0) for row in candidate_rows]


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


def test_generic_operation_words_cannot_pass_topic_gate() -> None:
    """“开启方式”相同不能证明 ACC 文档能支撑座椅加热问题。"""
    question = "座椅加热有哪几种开启方式？"
    assert not has_topic_support(question, _Index().chunks[0].text)
    assert has_topic_support(question, _Index().chunks[1].text)


def test_safety_belt_requires_full_feature_entity() -> None:
    """不能把“安全”“正确”这种泛词当作“安全带”资料。"""
    question = "安全带应该怎么正确佩戴？"
    battery_text = "磷酸铁锂电池安全性更高，但并非绝对安全，仍需正确使用和维护。"
    belt_text = "安全带的肩带应跨过肩部中央，腰带应贴紧髋部。"
    assert "安全带" in query_topic_entities(question)
    assert not has_topic_support(question, battery_text)
    assert has_topic_support(question, belt_text)


def test_extractive_generator_refuses_if_wrong_context_bypasses_retrieval() -> None:
    """生成层是最后一道防线：即便错误分块被强行传入，也不能直接摘答。"""
    question = "安全带应该怎么正确佩戴？"
    wrong = Chunk("battery::c000", "battery", "电池养护", "磷酸铁锂电池安全性更高，但仍需正确使用和维护。")
    cfg = RunConfig(llm_backend="offline")
    answer = AnswerGenerator(cfg).generate(question, [])
    # 空上下文路径同样必须显式标为拒答。
    assert answer.abstained and answer.answer == "资料不足，无法回答。"
    from rag_eval.retrieve import RetrievedChunk
    guarded = AnswerGenerator(cfg).generate(question, [RetrievedChunk(chunk=wrong)])
    assert guarded.abstained and guarded.answer == "资料不足，无法回答。", guarded.to_dict()


def test_entity_gate_runs_before_final_top_k() -> None:
    """正确资料排在初始 Top-K 之后，也应在过滤噪声后进入最终结果。"""
    cfg = RunConfig(llm_backend="offline", top_k=1, use_hybrid=False, rerank_backend="none")
    retrieved = Retriever(cfg, _WideSafetyIndex()).retrieve("安全带应该怎么正确佩戴？")
    assert [item.chunk.doc_id for item in retrieved] == ["belt"], [item.to_dict() for item in retrieved]


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


def test_extractive_generator_keeps_manual_steps_together() -> None:
    """离线模式必须保留连续步骤，不能把手册标题和跨页碎句拼在一起。"""
    from rag_eval.retrieve import RetrievedChunk

    manual = Chunk(
        "belt::c000", "belt", "小米 SU7 手册第41页",
        "解锁安全带\n1. 握住安全带锁舌，按下安全带锁扣按钮。\n"
        "2. 待安全带锁扣解锁后，将安全带锁舌放置到原位。\n注意:\n请勿松开安全带锁舌。",
        page=41,
    )
    answer = AnswerGenerator(RunConfig(llm_backend="offline")).generate(
        "小米SU7如何解锁安全带？", [RetrievedChunk(chunk=manual)]
    )
    assert "按下安全带锁扣按钮" in answer.answer, answer.to_dict()
    assert "第 41 页" in answer.answer, answer.to_dict()


def test_extractive_generator_keeps_heading_description_and_list() -> None:
    """“哪些功能”应保留标题、入口与连续功能列表。"""
    from rag_eval.retrieve import RetrievedChunk

    manual = Chunk(
        "ac::c000", "ac", "小米 SU7 手册第178页",
        "远程空调\n打开小米汽车 APP 进入车辆>空调，可对车辆空调进行远程控制。\n"
        "1. 开启或关闭方向盘加热。\n2. 开启或关闭座椅加热。\n3. 调节空调温度。",
        page=178,
    )
    answer = AnswerGenerator(RunConfig(llm_backend="offline")).generate(
        "远程空调可以控制哪些功能？", [RetrievedChunk(chunk=manual)]
    )
    assert "座椅加热" in answer.answer and "空调温度" in answer.answer, answer.to_dict()


def test_acc_alias_matches_manual_full_name() -> None:
    """用户说 ACC、手册写“自适应巡航”时，仍应通过实体准入。"""
    assert has_topic_support("ACC 在哪些情况下不适合使用？", "自适应巡航在恶劣天气下请勿开启。")
    assert "自适应巡航" in expand_query_entity_aliases("ACC 在哪些情况下不适合使用？")


def test_followup_keeps_previous_function_entity_not_old_intent() -> None:
    history = [{"role": "user", "content": "ACC 如何开启？"}]
    query, rewritten = build_retrieval_query("那哪些情况下不适合使用？", history, llm=None)
    assert rewritten and "acc" in query.casefold() and "如何开启" not in query, query


def test_condition_question_prefers_explicit_do_not_use_evidence() -> None:
    """安全边界问题应优先引用明确“请勿开启”的证据，而非泛泛的功能说明。"""
    from rag_eval.retrieve import RetrievedChunk

    generic = Chunk("acc::c000", "acc", "说明", "自适应巡航功能可以辅助驾驶。", page=147)
    boundary = Chunk(
        "acc::c001", "acc", "限制", "自适应巡航在以下特殊或复杂路况下请勿开启："
        "\n• 处于恶劣天气，如大雨、冰雪或大雾。", page=148,
    )
    answer = AnswerGenerator(RunConfig(llm_backend="offline")).generate(
        "ACC 在哪些情况下不适合使用？",
        [RetrievedChunk(chunk=generic), RetrievedChunk(chunk=boundary)],
    )
    assert "恶劣天气" in answer.answer and "第 148 页" in answer.answer, answer.to_dict()


def main() -> int:
    test_independent_question()
    test_generic_operation_words_cannot_pass_topic_gate()
    test_safety_belt_requires_full_feature_entity()
    test_extractive_generator_refuses_if_wrong_context_bypasses_retrieval()
    test_entity_gate_runs_before_final_top_k()
    test_unrelated_question_abstains()
    test_web_boilerplate_is_cleaned_and_flagged()
    test_reranker_receives_a_candidate_pool()
    print("PASS: 独立短问题不串题；泛操作词与泛安全词不放行；无关问题拒答；网页噪声清洗与告警；重排候选池")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
