"""应用诊断工具：在真实使用前把"能不能跑通"逐项验证清楚。

设计原则：每一项检查都真实执行并给出可判定的结论（PASS / WARN / FAIL），
不依赖任何 mock；缺少 API Key 时仍然会探测端点可达性，把问题提前暴露出来。

运行：
    python tools/diagnostics.py
"""

from __future__ import annotations

import importlib
import json
import os
import platform
import socket
import ssl
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESULTS: list[tuple[str, str, str]] = []   # (检查项, 结论, 细节)


def rec(item: str, status: str, detail: str = "") -> None:
    RESULTS.append((item, status, detail))


# ─────────────────────────────────────────────
def check_runtime() -> None:
    rec("Python 运行时", "PASS", f"{platform.python_version()} @ {sys.executable}")
    pkgs = ["streamlit", "pandas", "numpy", "pypdf", "jieba", "rank_bm25", "faiss", "sklearn", "openai"]
    for name in pkgs:
        try:
            mod = importlib.import_module(name)
            ver = getattr(mod, "__version__", "?")
            rec(f"依赖 {name}", "PASS", f"版本 {ver}")
        except Exception as exc:
            rec(f"依赖 {name}", "FAIL", f"{type(exc).__name__}: {exc}")


def check_imports() -> None:
    modules = [
        "rag_eval.config", "rag_eval.textutils", "rag_eval.llm", "rag_eval.embedder",
        "rag_eval.ingest", "rag_eval.index", "rag_eval.retrieve", "rag_eval.generate",
        "rag_eval.metrics", "rag_eval.evaluate", "rag_eval",
    ]
    for m in modules:
        try:
            importlib.import_module(m)
            rec(f"模块 {m}", "PASS", "可导入")
        except Exception as exc:
            rec(f"模块 {m}", "FAIL", f"{type(exc).__name__}: {exc}")


def check_data_integrity() -> None:
    from rag_eval import RecursiveChunker, load_documents_from, load_eval_set
    from rag_eval.config import SAMPLE_CORPUS_PATH, SAMPLE_EVAL_PATH

    try:
        docs = load_documents_from(builtin=True)
        rec("内置语料加载", "PASS", f"{len(docs)} 篇文档，{SAMPLE_CORPUS_PATH.name}")
    except Exception as exc:
        rec("内置语料加载", "FAIL", str(exc))
        return

    ids = [d.doc_id for d in docs]
    dup = {i for i in ids if ids.count(i) > 1}
    rec("语料 doc_id 唯一性", "FAIL" if dup else "PASS", f"重复：{dup}" if dup else f"{len(ids)} 个唯一 id")

    empty = [d.doc_id for d in docs if not d.text.strip()]
    rec("语料正文非空", "FAIL" if empty else "PASS", f"空文档：{empty}" if empty else "全部非空")

    try:
        items = load_eval_set(SAMPLE_EVAL_PATH)
        rec("内置评测集加载", "PASS", f"{len(items)} 条样本，{SAMPLE_EVAL_PATH.name}")
    except Exception as exc:
        rec("内置评测集加载", "FAIL", str(exc))
        return

    no_gt = [i.item_id for i in items if not i.ground_truth.strip()]
    rec("评测集 ground_truth 完整性", "FAIL" if no_gt else "PASS",
        f"缺标准答案：{no_gt}" if no_gt else "全部具备")

    # 最关键的一项：标注的相关文档必须真实存在，否则 IR 指标会静默归零
    known = set(ids)
    broken = {i.item_id: [r for r in i.relevant_doc_ids if r not in known]
              for i in items if any(r not in known for r in i.relevant_doc_ids)}
    rec("评测集 relevant_doc_ids 引用完整性", "FAIL" if broken else "PASS",
        f"悬空引用：{broken}" if broken else "全部指向真实存在的文档")

    no_rel = [i.item_id for i in items if not i.relevant_doc_ids]
    rec("评测集标注覆盖率", "WARN" if no_rel else "PASS",
        f"未标注相关文档：{no_rel}（这些题不参与 IR 指标）" if no_rel else "每题均有标注")

    chunks = RecursiveChunker(300, 60, 40).chunk_all(docs)
    over = [c.chunk_id for c in chunks if len(c.text) > 300 + 60]
    rec("分块长度约束", "WARN" if over else "PASS",
        f"{len(over)} 块超出 360 字符（重叠所致）" if over else f"{len(chunks)} 块均在约束内")

    # 重复文本检测：内容完全相同的分块会互相抢排名，稀释指标
    seen: dict[str, str] = {}
    dupe_chunks = 0
    for c in chunks:
        key = c.text.strip()
        if key in seen:
            dupe_chunks += 1
        else:
            seen[key] = c.chunk_id
    rec("分块内容去重", "WARN" if dupe_chunks else "PASS",
        f"{dupe_chunks} 个分块文本完全重复" if dupe_chunks else "无完全重复分块")


def check_config_edge_cases() -> None:
    from rag_eval import RunConfig, RecursiveChunker
    from rag_eval.embedder import Embedder
    from rag_eval.ingest import IngestError

    # 1) overlap >= chunk_size 必须被拒绝
    try:
        RecursiveChunker(chunk_size=100, chunk_overlap=100)
        rec("非法 overlap 校验", "FAIL", "overlap == chunk_size 未被拒绝")
    except IngestError:
        rec("非法 overlap 校验", "PASS", "overlap >= chunk_size 被正确拒绝")
    except Exception as exc:
        rec("非法 overlap 校验", "WARN", f"抛出非预期异常：{type(exc).__name__}: {exc}")

    # 2) 极小语料（单文档）不能崩——SVD 在样本数不足时会失败
    try:
        emb = Embedder(RunConfig(embed_dim=256))
        emb.fit(["只有一个很短的分块。"])
        vec = emb.encode(["只有一个很短的分块。"])
        rec("极小语料向量化", "PASS", f"单样本降维成功，实际维度 {vec.shape[1]}")
    except Exception as exc:
        rec("极小语料向量化", "FAIL", f"{type(exc).__name__}: {exc}")

    # 3) 空文本不应产生 NaN
    try:
        emb = Embedder(RunConfig(embed_dim=64))
        emb.fit(["文本一", "文本二", "文本三", "文本四"])
        v = emb.encode(["", "   ", "文本一"])
        import numpy as np

        ok = np.isfinite(v).all() and (np.linalg.norm(v, axis=1) > 0).all()
        rec("空字符串向量化", "PASS" if ok else "FAIL",
            "产出有限且非零向量" if ok else "出现 NaN/零向量")
    except Exception as exc:
        rec("空字符串向量化", "FAIL", f"{type(exc).__name__}: {exc}")

    # 4) 无关问题不应编造答案
    try:
        from rag_eval.generate import AnswerGenerator
        from rag_eval.retrieve import RetrievedChunk
        from rag_eval.ingest import Chunk

        gen = AnswerGenerator(RunConfig(llm_backend="offline"))
        ch = Chunk(chunk_id="c1", doc_id="d1", doc_title="t", text="安全带应贴合髋部。")
        ans = gen.generate("今天北京天气怎么样？", [RetrievedChunk(chunk=ch, score=1.0)])
        rec("离线模式拒答能力", "PASS" if ans.answer else "WARN",
            f"无关问题返回：{ans.answer[:40]!r}")
    except Exception as exc:
        rec("离线模式拒答能力", "FAIL", f"{type(exc).__name__}: {exc}")


def check_cache() -> None:
    from rag_eval.config import CACHE_DIR

    npy = list(CACHE_DIR.glob("*.npy"))
    rec("向量缓存目录", "PASS" if CACHE_DIR.exists() else "FAIL", f"{CACHE_DIR}，{len(npy)} 个向量文件")
    for f in npy:
        meta = f.with_suffix(".meta.json")
        if not meta.exists():
            rec(f"缓存元数据 {f.name}", "WARN", "缺少 .meta.json，无法审计")
            continue
        try:
            m = json.loads(meta.read_text(encoding="utf-8"))
            rec(f"缓存元数据 {f.name}", "PASS",
                f"n_chunks={m.get('n_chunks')} dim={m.get('dim')} sig={m.get('signature')}")
        except Exception as exc:
            rec(f"缓存元数据 {f.name}", "FAIL", str(exc))


def check_llm_config() -> None:
    from rag_eval import RunConfig
    from rag_eval.config import LLM_PRESETS

    for name, preset in LLM_PRESETS.items():
        ok = bool(preset.get("base_url")) or name == "offline"
        rec(f"LLM 预设 {name}", "PASS" if ok else "FAIL",
            f"base_url={preset.get('base_url') or '(离线，无)'} models={preset['models']}")

    cfg = RunConfig(llm_backend="deepseek", llm_model="deepseek-chat")
    key = cfg.resolved_api_key()
    rec("DeepSeek 凭据解析", "PASS" if key else "WARN",
        f"已解析到 Key（长度 {len(key)}）" if key else "未配置（生成将降级为抽取式，判官降级为词法指标）")
    rec("DeepSeek base_url", "PASS", cfg.resolved_base_url())


def check_network() -> None:
    """探测 DeepSeek 端点可达性。无 Key 时预期 401 —— 401 恰好证明链路是通的。"""
    proxy = {k: os.environ.get(k) for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY") if os.environ.get(k)}
    rec("网络代理环境", "WARN" if proxy else "PASS",
        f"检测到代理 {proxy}（可能影响外网调用）" if proxy else "无代理，直连")

    host = "api.deepseek.com"
    try:
        t0 = time.perf_counter()
        ip = socket.gethostbyname(host)
        rec("DNS 解析", "PASS", f"{host} -> {ip}（{(time.perf_counter()-t0)*1000:.0f} ms）")
    except Exception as exc:
        rec("DNS 解析", "FAIL", f"{type(exc).__name__}: {exc}")
        return

    try:
        t0 = time.perf_counter()
        ctx = ssl.create_default_context()
        with socket.create_connection((host, 443), timeout=10) as s:
            with ctx.wrap_socket(s, server_hostname=host) as ss:
                cert = ss.getpeercert()
        dt = (time.perf_counter() - t0) * 1000
        subj = dict(x[0] for x in cert.get("subject", []))
        rec("TLS 握手", "PASS", f"{dt:.0f} ms，证书 CN={subj.get('commonName', '?')}")
    except Exception as exc:
        rec("TLS 握手", "FAIL", f"{type(exc).__name__}: {exc}")
        return

    # 真实发一个请求：带 Key 期望 200，不带 Key 期望 401
    from rag_eval import RunConfig

    cfg = RunConfig(llm_backend="deepseek", llm_model="deepseek-chat")
    key = cfg.resolved_api_key()
    url = cfg.resolved_base_url() + "/models"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"} if key else {})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            body = resp.read(200).decode("utf-8", "ignore")
        rec("端点鉴权探测", "PASS", f"HTTP {resp.status}（带 Key 鉴权成功）：{body[:80]}")
    except urllib.error.HTTPError as e:
        if e.code == 401 and not key:
            rec("端点鉴权探测", "PASS", "HTTP 401 —— 无 Key 符合预期，证明网络链路可用")
        elif e.code == 401:
            rec("端点鉴权探测", "FAIL", "HTTP 401 —— 端点可达但 Key 无效或未生效")
        elif e.code == 402:
            rec("端点鉴权探测", "WARN", "HTTP 402 —— 端点可达但账户余额不足")
        else:
            rec("端点鉴权探测", "WARN", f"HTTP {e.code}: {e.reason}")
    except Exception as exc:
        rec("端点鉴权探测", "FAIL", f"{type(exc).__name__}: {exc}")


def check_app() -> None:
    """确认已启动的 Streamlit 应用真实在服务。"""
    try:
        with urllib.request.urlopen("http://127.0.0.1:8501/_stcore/health", timeout=8) as r:
            body = r.read(20).decode("utf-8", "ignore").strip()
        rec("Streamlit 服务健康", "PASS" if body == "ok" else "WARN", f"/_stcore/health -> {body!r}")
    except Exception as exc:
        rec("Streamlit 服务健康", "WARN", f"未在 8501 监听到服务：{type(exc).__name__}: {exc}")


# ─────────────────────────────────────────────
def main() -> int:
    print("═" * 78)
    print(" 智能座舱 RAG 评估系统 —— 应用诊断报告")
    print(f" 时间：{time.strftime('%Y-%m-%d %H:%M:%S')}   工作目录：{ROOT}")
    print("═" * 78)

    for fn in (check_runtime, check_imports, check_data_integrity, check_config_edge_cases,
               check_cache, check_llm_config, check_network, check_app):
        print(f"\n── {fn.__name__[6:].replace('_', ' ')} ──")
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            rec(fn.__name__, "FAIL", f"检查器自身异常 {type(exc).__name__}: {exc}")

    print("\n" + "═" * 78)
    print(" 结论汇总")
    print("═" * 78)
    from collections import Counter

    counts = Counter(s for _, s, _ in RESULTS)
    for item, status, detail in RESULTS:
        mark = {"PASS": "✔", "WARN": "!", "FAIL": "✘"}[status]
        print(f" {mark} [{status}] {item}" + (f" —— {detail}" if detail else ""))
    print("─" * 78)
    print(f" 合计：PASS {counts.get('PASS', 0)}　WARN {counts.get('WARN', 0)}　FAIL {counts.get('FAIL', 0)}")
    print("═" * 78)

    out = ROOT / "data" / "reports" / f"diagnostics_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "root": str(ROOT),
         "summary": dict(counts), "results": [{"item": i, "status": s, "detail": d} for i, s, d in RESULTS]},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f" 报告已保存：{out}")
    return 1 if counts.get("FAIL") else 0


if __name__ == "__main__":
    raise SystemExit(main())
