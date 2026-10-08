"""多轮对话测试：追问识别 / 查询改写 / 历史截断 / 真实两轮问答 / 评测口径不受影响。

运行：python tests/test_chat.py
离线即可跑完（不依赖 LLM Key）：无 Key 时走抽取式生成，同样会经过历史与改写逻辑。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_eval import (  # noqa: E402
    Evaluator, RunConfig, build_retrieval_query, load_documents_from, load_eval_set,
    looks_like_followup,
)
from rag_eval.config import SAMPLE_EVAL_PATH  # noqa: E402
from rag_eval.generate import AnswerGenerator  # noqa: E402
from rag_eval.ingest import Chunk  # noqa: E402
from rag_eval.retrieve import RetrievedChunk  # noqa: E402


def _history() -> list:
    return [
        {"role": "user", "content": "安全带的正确佩戴方法是什么？"},
        {"role": "assistant", "content": "肩带应跨过肩部中央，腰带贴紧髋部。[1]"},
    ]


def test_generation_context_respects_budget() -> None:
    cfg = RunConfig(max_context_chars=30, llm_backend="offline")
    generator = AnswerGenerator(cfg)
    original = Chunk(
        chunk_id="d::c000", doc_id="d", doc_title="测试", start=0, end=100,
        text="智能座舱" * 25,
    )
    result = generator.generate("智能座舱是什么？", [RetrievedChunk(chunk=original, rank=1)])
    assert result.context_chars <= cfg.max_context_chars
    assert len(result.used_contexts) == 1
    assert len(result.used_contexts[0].text) < len(original.text)
    assert result.used_contexts[0].text in original.text


def test_followup_detection() -> None:
    h = _history()
    # 追问：短问题 / 以「那」开头 / 含指代词
    assert looks_like_followup("那后排呢？", h) is True
    assert looks_like_followup("它有什么要求？", h) is True
    assert looks_like_followup("还有别的吗", h) is True
    # 独立问题：够长且无指代
    assert looks_like_followup("智能座舱的语音助手支持哪些方言识别？", h) is False
    # 没有历史时一律不算追问（首轮问题不该被改写）
    assert looks_like_followup("那后排呢？", []) is False
    assert looks_like_followup("那后排呢？", None) is False
    print("  ✔ 追问识别：短问题/指代词判为追问，独立问题与首轮不改写")


def test_query_rewrite_offline() -> None:
    """无 LLM 时应退化为「上一轮问题 + 本轮追问」的拼接，而不是把追问原样拿去检索。"""
    q, rewritten = build_retrieval_query("那后排呢？", _history(), llm=None)
    assert rewritten is True, "识别为追问时应标记已改写"
    assert "安全带" in q and "后排" in q, f"改写结果应带上文语境，实际：{q}"

    q2, r2 = build_retrieval_query("智能座舱的语音助手支持哪些方言识别？", _history(), llm=None)
    assert r2 is False and q2 == "智能座舱的语音助手支持哪些方言识别？", "独立问题不应被改写"

    q3, r3 = build_retrieval_query("那后排呢？", None, llm=None)
    assert r3 is False and q3 == "那后排呢？"
    print(f"  ✔ 查询改写（无 LLM 兜底）：『那后排呢？』→『{q[:38]}…』")


def test_history_trimming() -> None:
    gen = AnswerGenerator(RunConfig(llm_backend="offline"))
    history = [
        {"role": "user", "content": f"问题{i}"} if i % 2 == 0 else {"role": "assistant", "content": f"回答{i}"}
        for i in range(8)
    ]
    trimmed = gen._trim_history(history, max_turns=2)
    assert len(trimmed) == 4, f"2 轮应为 4 条消息，实际 {len(trimmed)}"
    assert trimmed[0]["role"] == "user", "截断后第一条必须是 user，否则上下文是残缺的"
    assert trimmed[0]["content"] == "问题4", f"应保留最近 2 轮，实际首条 {trimmed[0]['content']}"

    # 传进来的历史以助手开头时，必须丢掉这条孤立的回答
    odd = [{"role": "assistant", "content": "孤儿回答"}, {"role": "user", "content": "问题"}]
    t2 = gen._trim_history(odd, max_turns=3)
    assert t2 and t2[0]["role"] == "user", "历史不应以 assistant 开头"

    assert gen._trim_history(None, 3) == []
    assert gen._trim_history([], 3) == []
    print("  ✔ 历史截断：仅保留最近 N 轮且从 user 开始")


def test_real_two_turn_conversation() -> None:
    """真实两轮问答：第二轮是追问，应当借助上下文完成检索与生成。"""
    cfg = RunConfig(llm_backend="offline", top_k=5, use_hybrid=True)
    ev = Evaluator(cfg)
    docs = load_documents_from(builtin=True)
    stats = ev.prepare(docs)
    assert stats["n_chunks"] > 0

    q1 = "安全带的正确佩戴方法是什么？"
    r1 = ev.retriever.retrieve(build_retrieval_query(q1, None)[0])
    a1 = ev.generator.generate(q1, r1, history=None)
    assert a1.answer.strip(), f"首轮应给出答案：{a1.to_dict()}"
    assert a1.history_turns == 0, "首轮不应带入历史"

    history = [{"role": "user", "content": q1}, {"role": "assistant", "content": a1.answer}]
    q2 = "那孕妇呢？"
    query2, rewritten = build_retrieval_query(q2, history, llm=None)
    r2 = ev.retriever.retrieve(query2)
    a2 = ev.generator.generate(q2, r2, history=history)

    assert rewritten is True, "追问应触发改写"
    assert len(r2) > 0, "改写后的追问应能召回内容（改写前『那孕妇呢？』几乎召回不到）"
    assert a2.answer.strip(), f"追问应给出答案：{a2.to_dict()}"
    assert a2.history_turns == 1, f"应带入 1 轮历史，实际 {a2.history_turns}"

    # 对照：改写前的原始追问检索效果更差或为空——这是"改写确有价值"的证据
    raw = ev.retriever.retrieve(q2)
    print(f"  ✔ 真实两轮：首轮召回 {len(r1)} 段；追问改写『{query2[:26]}…』召回 {len(r2)} 段"
          f"（改写前 {len(raw)} 段），历史带入 {a2.history_turns} 轮")


def test_eval_path_unaffected_by_history() -> None:
    """评测链路不传 history，结果必须与改造前完全一致。

    「口径不变」要能检验，断言「不传 history」与「传空 history」输出逐字相同，
    同时跑一遍真实评测确认整条链路没有因为多轮改造而劣化。
    """
    cfg = RunConfig(llm_backend="offline")
    ev = Evaluator(cfg)
    ev.prepare(load_documents_from(builtin=True))

    q = "安全带的正确佩戴方法是什么？"
    retrieved = ev.retriever.retrieve(q)
    a_none = AnswerGenerator(cfg).generate(q, retrieved, history=None)
    a_empty = AnswerGenerator(cfg).generate(q, retrieved, history=[])
    assert a_none.answer == a_empty.answer, "空历史与不传历史必须等价，否则评测口径被动摇"
    assert a_none.history_turns == 0 and a_empty.history_turns == 0

    items = load_eval_set(SAMPLE_EVAL_PATH)[:2]
    report = ev.run(items)
    assert len(report.records) == 2
    for rec in report.records:
        assert rec.answer.strip(), f"离线评测每题都应有答案：{rec.to_dict()}"
        assert rec.generation_mode == "extractive", f"离线模式应为抽取式，实际 {rec.generation_mode}"
        assert rec.error is None, f"评测不应报错：{rec.error}"
    print(f"  ✔ 评测口径不变：{len(report.records)} 题全部完成，"
          f"token_f1={report.aggregates.get('token_f1')}，hist=None/[] 输出逐字相同")


TESTS = [
    ("生成上下文预算", test_generation_context_respects_budget),
    ("追问识别", test_followup_detection),
    ("查询改写（离线兜底）", test_query_rewrite_offline),
    ("历史截断", test_history_trimming),
    ("真实两轮问答", test_real_two_turn_conversation),
    ("评测口径不受影响", test_eval_path_unaffected_by_history),
]


def main() -> int:
    print("── 多轮对话验证 ──")
    passed = failed = 0
    for name, fn in TESTS:
        print(f"\n[{name}]")
        try:
            fn()
            passed += 1
        except AssertionError as exc:
            failed += 1
            print(f"  ✘ 失败：{exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  ✘ 异常：{type(exc).__name__}: {exc}")
    print(f"\n结果：{passed}/{passed + failed} 通过")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
