"""真实评测引擎的冒烟测试与一致性校验。

覆盖：
1. 分块器契约（不超长、有重叠、可切分）
2. IR 指标的文档级去重正确性（历史上出现过 recall/ndcg > 1 的 bug，这里做回归防护）
3. 全部指标落在 [0, 1]
4. 端到端真实评测（离线链路）：能跑通、无失败、指标合理
5. 缺少 API Key 时 LLMClient 必须明确报错，而不是静默伪造
6. LLM 重排不可用时必须显式降级为 BM25，并把降级写进警告
7. Streamlit 应用冒烟：脚本无异常，且点击「开始真实评测」能产出报告

运行：
    python tests/test_smoke.py          # 直接运行
    python -m pytest tests/test_smoke.py -q
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 用临时空知识库目录：工作台默认语料是「已导入知识库」（库里非空时），
# 而内置评测集的 relevant_doc_ids 只对应内置语料 —— 本机真实知识库在的话，
# 默认语料会被换成知识库，检索类指标全 0，这个用例就不再验证任何东西了。
_KB_TMP = Path(tempfile.mkdtemp(prefix="smoke_kb_"))
os.environ["RAG_KB_DIR"] = str(_KB_TMP)

from rag_eval import (  # noqa: E402
    Evaluator, RunConfig, RecursiveChunker, load_documents_from, load_eval_set,
)
from rag_eval.config import SAMPLE_EVAL_PATH  # noqa: E402
from rag_eval.llm import LLMClient, LLMError  # noqa: E402
from rag_eval.metrics import retrieval_metrics  # noqa: E402

APP_PATH = ROOT / "rag_eval_app.py"


# ─────────────────────────────────────────────
def test_chunker_contract() -> None:
    docs = load_documents_from(builtin=True)
    chunker = RecursiveChunker(chunk_size=200, chunk_overlap=40, min_chunk_chars=30)
    chunks = chunker.chunk_all(docs)
    assert chunks, "分块结果不应为空"
    oversize = [c for c in chunks if len(c.text) > 260]   # 允许重叠导致的少量超出
    assert len(oversize) <= max(1, len(chunks) // 10), f"过多分块明显超长：{len(oversize)}/{len(chunks)}"
    assert all(c.chunk_id and c.doc_id for c in chunks), "分块必须带可溯源的 id"
    print(f"  ✔ 分块契约：{len(chunks)} 块，最大长度 {max(len(c.text) for c in chunks)}")


def test_config_validation() -> None:
    invalid = [
        {"chunk_size": 0},
        {"chunk_size": 100, "chunk_overlap": 100},
        {"top_k": 0},
        {"bm25_weight": 1.1},
        {"embed_dim": 0},
        {"llm_temperature": -0.1},
    ]
    for kwargs in invalid:
        try:
            RunConfig(**kwargs)
            raise AssertionError(f"无效配置未被拒绝：{kwargs}")
        except ValueError:
            pass


def test_retrieval_metrics_dedupe() -> None:
    """回归防护：同一文档的多个分块命中，不能让 recall / ndcg 超过 1。"""
    retrieved = ["d1", "d1", "d1", "d1", "d1"]      # 5 个分块同属文档 d1
    rel = ["d1"]
    m = retrieval_metrics(retrieved, rel, k=5)
    assert m["recall_at_k"] == 1.0, f"recall 应为 1.0，实际 {m['recall_at_k']}"
    assert m["ndcg"] == 1.0, f"ndcg 应为 1.0，实际 {m['ndcg']}"
    assert m["precision_at_k"] == 0.2, f"标准 Precision@5 应为 0.2，实际 {m['precision_at_k']}"
    assert m["mrr"] == 1.0

    # 未标注相关文档时，不允许编造数值
    assert retrieval_metrics(retrieved, [], k=5)["hit_rate"] is None
    print(f"  ✔ IR 指标去重正确：{m}")


def test_metric_bounds() -> None:
    docs = load_documents_from(builtin=True)
    items = load_eval_set(SAMPLE_EVAL_PATH)
    cfg = RunConfig(chunk_size=300, chunk_overlap=60, top_k=5, llm_backend="offline")
    ev = Evaluator(cfg)
    ev.prepare(docs)
    report = ev.run(items)
    assert report.aggregates, "聚合指标不应为空"
    bad = {k: v for k, v in report.aggregates.items()
           if v is not None and not (0.0 <= v <= 1.0)}
    assert not bad, f"存在越界指标：{bad}"
    print(f"  ✔ 指标全部落在 [0,1]：{[f'{k}={v}' for k, v in list(report.aggregates.items())[:6]]}")


def test_end_to_end_offline() -> None:
    docs = load_documents_from(builtin=True)
    items = load_eval_set(SAMPLE_EVAL_PATH)
    cfg = RunConfig(chunk_size=300, chunk_overlap=60, top_k=5,
                    llm_backend="offline", rerank_backend="none")
    report = Evaluator(cfg)
    build = report.prepare(docs)
    rep = report.run(items)

    assert build["n_chunks"] > 0
    assert rep.runtime["n_items"] == len(items)
    assert rep.runtime["n_failed"] == 0, f"不应有失败项：{[r.error for r in rep.records if r.error]}"
    assert all(r.answer for r in rep.records), "每题都应有真实答案"
    assert all(r.retrieved for r in rep.records), "每题都应有真实召回结果"
    # 抽取式答案来自上下文原文，忠实度应当很高（这是可验证的行为而非巧合）
    assert rep.aggregates["faithfulness"] >= 0.9, f"抽取式忠实度异常：{rep.aggregates['faithfulness']}"
    print(f"  ✔ 端到端离线评测：{rep.runtime['n_success']}/{rep.runtime['n_items']} 成功，"
          f"hit_rate={rep.aggregates['hit_rate']} token_f1={rep.aggregates['token_f1']}")


def test_llm_requires_key() -> None:
    cfg = RunConfig(llm_backend="deepseek", llm_model="deepseek-chat")
    import os

    saved = {k: os.environ.pop(k, None) for k in ("DEEPSEEK_API_KEY", "OPENAI_API_KEY")}
    try:
        try:
            LLMClient(cfg)
            raise AssertionError("缺少 Key 时应当抛出 LLMError")
        except LLMError as exc:
            assert "API Key" in str(exc), f"报错信息应可指导用户，实际：{exc}"
            print(f"  ✔ 缺 Key 明确报错：{str(exc)[:60]}…")
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v


def test_rerank_degrades_when_llm_missing() -> None:
    """回归防护：LLM 重排不可用时必须降级为 BM25 并告警。

    注意必须隔离环境变量：本机可能已配置真实 Key，否则「LLM 可用」会让本用例前提失效。
    """
    import os

    saved = {k: os.environ.pop(k, None) for k in ("DEEPSEEK_API_KEY", "OPENAI_API_KEY")}
    try:
        docs = load_documents_from(builtin=True)
        cfg = RunConfig(llm_backend="deepseek", llm_model="deepseek-chat",
                        rerank_backend="llm", use_hybrid=True)
        ev = Evaluator(cfg)
        ev.prepare(docs)
        assert ev.cfg.rerank_backend == "bm25", "LLM 重排不可用时应降级为 bm25"
        assert any("重排" in w for w in ev.warnings), "降级必须出现在警告里"
        print("  ✔ LLM 重排缺失时显式降级为 BM25 并告警")
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v


def test_streamlit_app_smoke() -> None:
    """用 Streamlit 官方 AppTest 真实执行页面脚本，并模拟点击评测按钮。"""
    try:
        from streamlit.testing.v1 import AppTest
    except Exception as exc:  # pragma: no cover
        print(f"  ⚠ 跳过 AppTest（不可用）：{exc}")
        return

    at = AppTest.from_file(str(APP_PATH), default_timeout=180)
    at.run()
    assert not at.exception, f"页面首次渲染异常：{[e.value for e in at.exception]}"
    print("  ✔ 页面首屏渲染无异常")

    # 定位「开始真实评测」按钮（默认参数即为内置真实样例语料 + 内置真实评测集）
    run_btn = None
    for b in at.button:
        if "开始真实评测" in (b.label or ""):
            run_btn = b
            break
    assert run_btn is not None, "未找到开始评测按钮"

    run_btn.click().run(timeout=180)
    assert not at.exception, f"点击评测后异常：{[e.value for e in at.exception]}"
    report = at.session_state.get("report")
    assert report, "点击评测后应产出真实报告"
    assert report["runtime"]["n_items"] > 0, "报告应有评测样本"
    assert report["records"], "报告应包含逐题真实结果"

    # 失败项必须可诊断：把真实报错打出来，而不是只报数量
    if report["runtime"]["n_failed"]:
        print(f"  ⚠ 失败项详情（{report['runtime']['n_failed']} 条）：")
        for r in report["records"]:
            if r["error"]:
                print(f"    Q{r['index']}: {r['error'][:400]}")

    assert report["runtime"]["n_failed"] == 0, (
        f"UI 链路存在失败项：{[r['error'] for r in report['records'] if r['error']]}"
    )

    # 检索类指标必须有意义：全 0 说明「评测集标注」和「当前语料」对不上，
    # 这种报告看着"跑成功了"其实什么都没验证到，必须拦住。
    agg = report["aggregates"]
    assert agg.get("hit_rate", 0) > 0, (
        f"UI 默认配置下 hit_rate={agg.get('hit_rate')}，说明评测集标注与默认语料不匹配"
        f"（默认应用内置语料 + 内置评测集，二者应当匹配）"
    )
    print(f"  ✔ UI 全链路真实评测：{report['runtime']['n_success']}/{report['runtime']['n_items']} 成功，"
          f"生成模式={report['records'][0]['generation_mode']}，"
          f"hit_rate={agg.get('hit_rate')} recall_at_k={agg.get('recall_at_k')} "
          f"token_f1={agg.get('token_f1')}")


# ─────────────────────────────────────────────
def main() -> int:
    tests = [
        test_chunker_contract,
        test_config_validation,
        test_retrieval_metrics_dedupe,
        test_metric_bounds,
        test_end_to_end_offline,
        test_llm_requires_key,
        test_rerank_degrades_when_llm_missing,
        test_streamlit_app_smoke,
    ]
    print("── 冒烟测试 ──")
    failed = 0
    for t in tests:
        print(f"\n[{t.__name__}]")
        try:
            t()
        except Exception as exc:  # noqa: BLE001
            failed += 1
            import traceback

            print(f"  ✘ 失败：{type(exc).__name__}: {exc}")
            traceback.print_exc()
    print(f"\n结果：{len(tests) - failed}/{len(tests)} 通过")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
