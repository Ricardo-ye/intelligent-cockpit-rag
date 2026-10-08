"""真实评测命令行入口（无界面，便于自动化验证与批量对比）。

示例：
    # 用内置语料与评测集跑一遍（离线真实计算，无需任何 Key）
    python run_eval_cli.py

    # 用自己的 PDF + 评测集，并启用 DeepSeek 生成与 LLM 判官
    python run_eval_cli.py --pdf 手册.pdf --evalset eval.json --llm deepseek --model deepseek-chat --judge

    # 参数扫描：对比不同 chunk_size 的真实指标
    python run_eval_cli.py --sweep chunk_size=128,256,512

输出：在 data/reports/ 下落一份完整 JSON 报告，并在终端打印真实指标汇总。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from rag_eval import (  # noqa: E402
    Evaluator, RunConfig, capability_report,
    load_documents_from, load_eval_set,
)
from rag_eval.config import SAMPLE_CORPUS_PATH, SAMPLE_EVAL_PATH  # noqa: E402
from rag_eval.costs import PricingError, estimate_cost, load_pricing  # noqa: E402


def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="智能座舱 RAG 真实评测 CLI")
    p.add_argument("--pdf", help="知识库 PDF 路径")
    p.add_argument("--corpus", help="知识库 JSON 语料路径")
    p.add_argument("--txt", help="知识库 TXT 路径")
    p.add_argument("--evalset", help="评测集 JSON/CSV 路径（默认用内置样例）")
    p.add_argument("--chunk-size", type=int, default=300)
    p.add_argument("--overlap", type=int, default=60)
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--no-hybrid", action="store_true", help="关闭混合检索，仅用向量")
    p.add_argument("--bm25-weight", type=float, default=0.4)
    p.add_argument("--rerank", default="none", choices=["none", "bm25", "llm"])
    p.add_argument("--rerank-topk", type=int, default=3)
    p.add_argument("--compress", action="store_true", help="启用上下文压缩")
    p.add_argument("--embed", default="local-tfidf-svd",
                   choices=["local-tfidf-svd", "api"])
    p.add_argument("--embed-dim", type=int, default=256)
    p.add_argument("--llm", default="offline",
                   choices=["offline", "deepseek", "openai-compatible"])
    p.add_argument("--model", default=None, help="LLM 模型名，缺省取后端默认值")
    p.add_argument("--judge", action="store_true", help="启用 LLM 判官指标")
    p.add_argument("--no-cache", action="store_true", help="禁用向量缓存")
    p.add_argument("--sweep", help="参数扫描，如 chunk_size=128,256,512；只接受一个参数名")
    p.add_argument("--pricing", help="版本化价格表 JSON；提供后才会估算金额，不提供则只记录 token")
    p.add_argument("--compare-models", help="以同一配置依次评测的模型名，逗号分隔；仅适用于非 offline LLM")
    p.add_argument("--experiment-name", help="写入报告的实验名称，便于归档比较")
    p.add_argument("--out", help="报告输出路径")
    return p.parse_args()


def make_base_cfg(a: argparse.Namespace) -> RunConfig:
    from rag_eval.config import LLM_PRESETS

    model = a.model or LLM_PRESETS[a.llm]["default_model"]
    return RunConfig(
        chunk_size=a.chunk_size,
        chunk_overlap=a.overlap,
        embed_backend=a.embed,
        embed_dim=a.embed_dim,
        top_k=a.topk,
        use_hybrid=not a.no_hybrid,
        bm25_weight=a.bm25_weight,
        rerank_backend=a.rerank,
        rerank_top_k=a.rerank_topk,
        use_compression=a.compress,
        llm_backend=a.llm,
        llm_model=model,
        use_llm_judge=a.judge,
        use_cache=not a.no_cache,
    )


def coerce_sweep_value(current, raw: str):
    """按字段真实类型解析扫描值，避免 bool('false') 仍为 True。"""
    if isinstance(current, bool):
        value = raw.strip().lower()
        if value in {"1", "true", "yes", "on"}:
            return True
        if value in {"0", "false", "no", "off"}:
            return False
        raise ValueError(f"布尔值应为 true/false，实际：{raw}")
    return type(current)(raw) if current is not None else raw


def print_caps() -> None:
    caps = capability_report()
    print("── 运行时能力自检 ──")
    for k, v in caps.items():
        print(f"  {'✔' if v is True else ('✘' if v is False else '·')} {k}"
              + ("" if isinstance(v, bool) else f"：{v}"))
    print()


def load_data(a: argparse.Namespace):
    if a.pdf:
        docs = load_documents_from(pdf_path=a.pdf)
    elif a.corpus:
        docs = load_documents_from(corpus_json_path=a.corpus)
    elif a.txt:
        docs = load_documents_from(txt_path=a.txt)
    else:
        docs = load_documents_from(builtin=True)
        print(f"· 使用内置语料：{SAMPLE_CORPUS_PATH}")

    if a.evalset:
        items = load_eval_set(a.evalset)
    else:
        items = load_eval_set(SAMPLE_EVAL_PATH)
        print(f"· 使用内置评测集：{SAMPLE_EVAL_PATH}")
    return docs, items


def run_once(cfg: RunConfig, docs, items, quiet: bool = False, pricing: dict | None = None,
             experiment_name: str | None = None) -> dict:
    ev = Evaluator(cfg)

    def cb(frac: float, msg: str) -> None:
        if not quiet:
            print(f"  [{frac * 100:5.1f}%] {msg}")

    build = ev.prepare(docs, progress=cb)
    report = ev.run(items, progress=cb)
    d = report.to_dict()
    d["build_stats"] = {**build, **d["build_stats"]}
    if experiment_name:
        d["experiment_name"] = experiment_name
    if pricing is not None:
        d["cost_estimate"] = estimate_cost(d.get("runtime", {}), pricing)
    return d


def print_summary(d: dict) -> None:
    aggs, meta = d["aggregates"], d["metric_meta"]
    print("\n── 真实指标汇总 ──")
    for k, v in aggs.items():
        m = meta.get(k, {})
        tag = ("LLM判官" if m.get("method") == "llm-judge"
               else ("混合口径" if m.get("method") == "mixed" else "确定性"))
        val = "—" if v is None else f"{v:.4f}"
        print(f"  {k:<20} {val:>8}   [{tag}] {m.get('detail', '')}")

    rt = d["runtime"]
    print("\n── 运行统计 ──")
    print(f"  语料分块：{d['build_stats'].get('n_chunks')}　评测题数：{rt['n_items']}（成功 {rt['n_success']} / 失败 {rt['n_failed']}）")
    print(f"  平均检索：{rt['avg_retrieve_ms']} ms　平均生成：{rt['avg_generate_ms']} ms（含压缩）")
    if rt.get("avg_compress_ms"):
        print(f"  其中上下文压缩：{rt['avg_compress_ms']} ms／题　纯生成：{rt['avg_generate_net_ms']} ms／题")
    print(f"  向量后端：{d['build_stats'].get('embed_backend')}　索引后端：{d['build_stats'].get('vector_index_backend')}　缓存命中：{d['build_stats'].get('cache_hit')}")

    llm = rt.get("llm")
    if llm:
        print("\n── LLM 真实用量 ──")
        print(f"  模型：{llm['model']}（思维链模型：{llm['is_reasoning_model']}）@ {llm['base_url']}")
        print(f"  调用 {llm['calls']} 次　重试 {llm['retries']} 次　输出截断 {llm['truncated_calls']} 次　"
              f"自动抬高 token 上限 {llm['max_tokens_raised']} 次")
        print(f"  tokens：prompt={llm['prompt_tokens']}　completion={llm['completion_tokens']}　"
              f"total={llm['total_tokens']}")
        print(f"  LLM 累计耗时：{llm['llm_latency_s']}s　平均每题："
              f"{round(llm['llm_latency_s'] / max(llm['calls'], 1), 3)}s")
    else:
        print("\n── 生成模式 ──")
        print("  离线抽取式（本次未调用任何 LLM，指标中的忠实度/相关性为词法代理）")

    judge = rt.get("judge")
    if judge:
        print(f"  判官口径：{judge.get('method')}　调用 {judge.get('judge_calls', '-')} 次　"
              f"耗时 {judge.get('judge_latency_s', '-')}s")
    else:
        print("  判官口径：未启用 LLM 判官（忠实度/相关性使用词法代理指标）")

    cost = d.get("cost_estimate")
    if cost:
        if cost.get("available"):
            print("\n── 成本估算（可复现）──")
            print(f"  估算金额：{cost['estimated_total_cost']:.8f} {cost['currency']}（生成 {cost['generation_cost']:.8f} / 判官 {cost['judge_cost']:.8f}）")
            print(f"  价格表：{cost['pricing_version']} @ {cost['pricing_source']}")
        else:
            print(f"\n── 成本估算 ──\n  未生成金额：{cost['reason']}")

    for w in d.get("warnings", []):
        print(f"  ⚠ {w}")

    # 请求配置 vs 实际生效配置：把自动降级摊开说清楚
    req, eff = d.get("config_requested") or {}, d.get("config") or {}
    diffs = {k: (req.get(k), eff.get(k)) for k in req if k in eff and req[k] != eff[k]}
    if diffs:
        print("\n── 配置降级对比（请求 -> 生效）──")
        for k, (a, b) in diffs.items():
            print(f"  {k}: {a} -> {b}")

    print("\n── 逐题真实结果（前 3 题）──")
    for r in d["records"][:3]:
        print(f"\n  Q{r['index']} {r['question']}")
        print(f"    答案：{(r['answer'] or '')[:110]}")
        print(f"    标准：{(r['ground_truth'] or '')[:110]}")
        print(f"    指标：token_f1={r['metrics'].get('token_f1')} ctx_prec={r['metrics'].get('context_precision')} "
              f"ctx_rec={r['metrics'].get('context_recall')} faithful={r['metrics'].get('faithfulness')}")
        top = r["retrieved"][0] if r["retrieved"] else None
        if top:
            print(f"    首位召回：{top['doc_title']} (score={top['score']})")


def main() -> int:
    a = build_args()
    print_caps()
    docs, items = load_data(a)

    pricing = None
    if a.pricing:
        try:
            pricing = load_pricing(a.pricing)
        except PricingError as exc:
            print(f"价格表无效：{exc}")
            return 2
    if a.compare_models and a.llm == "offline":
        print("--compare-models 需要非 offline 的 --llm 后端。")
        return 2

    out_dir = SCRIPT_DIR / "data" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")

    if a.sweep:
        if "=" not in a.sweep:
            print("--sweep 格式应为 name=v1,v2,...")
            return 2
        name, _, vals = a.sweep.partition("=")
        name = name.strip()
        if name not in RunConfig.__dataclass_fields__:
            print(f"不支持的扫描参数：{name}")
            return 2
        results = {}
        for v in [x.strip() for x in vals.split(",") if x.strip()]:
            cfg = make_base_cfg(a)
            cur = getattr(cfg, name)
            try:
                setattr(cfg, name, coerce_sweep_value(cur, v))
                cfg.validate()
            except ValueError as exc:
                print(f"扫描值无效：{name}={v}（{exc}）")
                return 2
            print(f"\n═════ {name} = {v} ═════")
            d = run_once(cfg, docs, items, quiet=True, pricing=pricing, experiment_name=a.experiment_name)
            print_summary(d)
            results[f"{name}={v}"] = {"aggregates": d["aggregates"], "runtime": d["runtime"], "cost_estimate": d.get("cost_estimate")}
        p = out_dir / f"sweep_{name}_{ts}.json"
        p.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n扫描结果已保存：{p}")
        return 0

    cfg = make_base_cfg(a)
    print(f"· 配置：{json.dumps(cfg.to_dict(), ensure_ascii=False)}")
    print()
    if a.compare_models:
        models = [m.strip() for m in a.compare_models.split(",") if m.strip()]
        if len(models) < 2:
            print("--compare-models 至少需要两个模型名。")
            return 2
        comparison = {"experiment_name": a.experiment_name or f"model-comparison-{ts}", "runs": []}
        for model in models:
            run_cfg = make_base_cfg(a)
            run_cfg.llm_model = model
            run_cfg.validate()
            print(f"\n═════ 模型：{model} ═════")
            d = run_once(run_cfg, docs, items, pricing=pricing, experiment_name=comparison["experiment_name"])
            print_summary(d)
            comparison["runs"].append({
                "model": model, "config": d.get("config"), "aggregates": d["aggregates"],
                "runtime": d["runtime"], "cost_estimate": d.get("cost_estimate"), "warnings": d.get("warnings", []),
            })
        p = Path(a.out) if a.out else (out_dir / f"comparison_{ts}.json")
        p.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n跨模型对照已保存：{p}")
        return 0

    d = run_once(cfg, docs, items, pricing=pricing, experiment_name=a.experiment_name)
    print_summary(d)

    p = Path(a.out) if a.out else (out_dir / f"report_{ts}.json")
    p.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n完整报告已保存：{p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
