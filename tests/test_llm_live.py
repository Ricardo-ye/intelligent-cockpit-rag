"""LLM 实时连通性验证（需要真实 API Key，缺失时自动跳过）。

验证内容：
1. .env / 环境变量中的 Key 是否被正确解析
2. Base URL 的 DNS 与 HTTPS 可达性（真实网络探测）
3. 基础对话调用（chat）
4. JSON 模式调用（chat_json）——重点验证 response_format 兼容性
5. 多个模型的兼容性差异（如 deepseek-chat vs deepseek-reasoner）
6. 网络不通时的报错是否可读、是否被误判为鉴权失败

运行：
    python tests/test_llm_live.py
"""

from __future__ import annotations

import json
import socket
import ssl
import sys
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_eval import LLMClient, LLMError, RunConfig, capability_report  # noqa: E402
from rag_eval.config import LLM_PRESETS  # noqa: E402


def net_probe(base_url: str) -> dict:
    """真实探测 base_url 的 DNS 与 TLS 连通性（绕过可能是 502 的转发代理）。"""
    host = urlparse(base_url).hostname or ""
    result: dict = {"host": host, "dns": None, "tcp443": None, "tls": None}
    try:
        result["dns"] = socket.gethostbyname(host)
    except Exception as exc:
        result["dns"] = f"FAILED: {exc}"
        return result
    try:
        with socket.create_connection((host, 443), timeout=10):
            result["tcp443"] = "ok"
    except Exception as exc:
        result["tcp443"] = f"FAILED: {exc}"
        return result
    try:
        ctx = ssl.create_default_context()
        with socket.create_connection((host, 443), timeout=10) as sock:
            with ctx.wrap_socket(sock, server_hostname=host) as ss:
                result["tls"] = f"ok ({ss.version()})"
    except Exception as exc:
        result["tls"] = f"FAILED: {exc}"
    return result


def raw_http_probe(base_url: str, api_key: str) -> dict:
    """不依赖 openai SDK，直接用 urllib 打一次真实请求，用于判断是否 SDK 层的问题。"""
    body = json.dumps({
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": "回复两个字：正常"}],
        "max_tokens": 16,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        return {
            "ok": True,
            "http_status": 200,
            "latency_s": round(time.perf_counter() - t0, 3),
            "reply": (payload.get("choices") or [{}])[0].get("message", {}).get("content", "")[:60],
            "usage": payload.get("usage"),
        }
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")[:400]
        return {"ok": False, "http_status": exc.code, "latency_s": round(time.perf_counter() - t0, 3),
                "error": detail}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                "latency_s": round(time.perf_counter() - t0, 3)}


def main() -> int:
    out: dict = {"steps": []}

    def log(name: str, data) -> None:
        out["steps"].append({"check": name, "result": data})
        print(f"\n[{name}]")
        print(json.dumps(data, ensure_ascii=False, indent=2))

    # 1) 凭据解析
    caps = capability_report()
    cfg = RunConfig(llm_backend="deepseek", llm_model="deepseek-chat", use_llm_judge=True)
    key = cfg.resolved_api_key()
    log("1. 凭据解析", {
        "capability_report_LLM_API_Key": caps.get("LLM API Key"),
        "key_source": caps.get("API Key 来源"),
        "resolved_base_url": cfg.resolved_base_url(),
        "key_len": len(key),
        "key_prefix": key[:6] + "…" if key else "",
        "llm_ready": cfg.llm_ready(),
        "judge_ready": cfg.judge_ready(),
    })

    if not key:
        print("\n未检测到 Key，跳过实时调用验证。")
        (ROOT / "data" / "reports").mkdir(parents=True, exist_ok=True)
        (ROOT / "data" / "reports" / "llm_live_check.json").write_text(
            json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        return 0

    base = cfg.resolved_base_url()

    # 2) 网络可达性
    log("2. 网络探测", net_probe(base))

    # 3) 绕过 SDK 的裸 HTTP 调用
    log("3. 裸 HTTP 调用（urllib，不经 SDK）", raw_http_probe(base, key))

    # 4) SDK 构建
    try:
        client = LLMClient(cfg, timeout=60)
        log("4. LLMClient 构建", {"ok": True, "base_url": client.base_url, "model": client.model})
    except LLMError as exc:
        log("4. LLMClient 构建", {"ok": False, "error": str(exc)})
        return _dump(out)

    # 5) 基础对话
    try:
        log("5. 基础对话 chat()", client.connectivity_check())
    except LLMError as exc:
        log("5. 基础对话 chat()", {"ok": False, "error": str(exc)})

    # 6) JSON 模式（重点）
    try:
        data = client.chat_json(
            "你是信息抽取器。只输出 JSON：{\"claims\":[\"论断\"]}",
            "答案：安全带必须先调整座椅再佩戴。",
            max_tokens=128,
        )
        log("6. JSON 模式 chat_json()", {"ok": True, "parsed": data,
                                        "usage": {"prompt": client.usage.prompt_tokens,
                                                  "completion": client.usage.completion_tokens}})
    except LLMError as exc:
        log("6. JSON 模式 chat_json()", {"ok": False, "error": str(exc)})

    # 7) 多模型兼容性
    for model in LLM_PRESETS["deepseek"]["models"]:
        c2 = RunConfig(llm_backend="deepseek", llm_model=model)
        try:
            cl = LLMClient(c2, timeout=60)
            t0 = time.perf_counter()
            txt = cl.chat([{"role": "user", "content": "回复两个字：正常"}], max_tokens=32)
            log(f"7. 模型 {model} 基础对话", {"ok": True, "reply": txt[:40],
                                            "latency_s": round(time.perf_counter() - t0, 3)})
        except LLMError as exc:
            log(f"7. 模型 {model} 基础对话", {"ok": False, "error": str(exc)[:300]})
        try:
            cl = LLMClient(c2, timeout=60)
            d = cl.chat_json("只输出 JSON：{\"ok\":true}", "测试")
            log(f"7b. 模型 {model} JSON 模式", {"ok": True, "parsed": d})
        except LLMError as exc:
            log(f"7b. 模型 {model} JSON 模式", {"ok": False, "error": str(exc)[:300]})

    # 8) Embedding 接口探测（预期不可用，用于确认降级路径）
    try:
        from rag_eval import Embedder

        e = Embedder(RunConfig(embed_backend="api", embed_api_model="text-embedding-3-small"))
        e.fit(["测试"])
        e.encode(["测试"])
        log("8. Embedding API", {"ok": True})
    except Exception as exc:
        log("8. Embedding API", {"ok": False, "error": str(exc)[:200]})

    return _dump(out)


def _dump(out: dict) -> int:
    p = ROOT / "data" / "reports" / "llm_live_check.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已写入：{p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
