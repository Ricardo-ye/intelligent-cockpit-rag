"""联网导入模块测试：链接判定 / 噪声过滤 / 相关性过滤 / doc_id 稳定性 / 抓取失败归因。

默认全部离线（不打网络请求），运行：python tests/test_webimport.py
需要真实联网的用例用环境变量开启：RAG_LIVE_NET=1 python tests/test_webimport.py
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_eval.kb import KnowledgeBase  # noqa: E402
from rag_eval.security import (  # noqa: E402
    SecurityError, resolve_upload_path, safe_upload_filename, validate_public_http_url,
)
from rag_eval.webimport import (  # noqa: E402
    ENGINE_CHAIN, SearchHit, WebDoc, _is_search_page, _is_widget_junk,
    _query_tokens, _relevance_ok, fetch_url, is_direct_url, to_documents, web_search,
)

LIVE = os.environ.get("RAG_LIVE_NET") == "1"


# ─────────────────────────── 链接判定 ───────────────────────────

def test_network_and_upload_security_boundaries() -> None:
    for url in ("http://127.0.0.1/admin", "http://[::1]/", "file:///etc/passwd"):
        try:
            validate_public_http_url(url)
            raise AssertionError(f"危险 URL 未被拒绝：{url}")
        except SecurityError:
            pass

    with tempfile.TemporaryDirectory(prefix="upload_safe_") as tmp:
        root = Path(tmp)
        target = resolve_upload_path(root, "../../../../evil.txt")
        assert target.parent == root.resolve()
        assert safe_upload_filename("..\\..\\CON.txt").startswith("_")

def test_is_direct_url_classifies_redirect_chains() -> None:
    """百度/360/搜狗的 link?url= 跳转链在部分网络下抓不动，必须被标成「非直连」。"""
    redirects = [
        "http://www.baidu.com/link?url=abc",
        "http://www.baidu.com/baidu.php?url=abc",      # 广告位
        "https://www.so.com/link?m=abc",
        "https://www.sogou.com/link?url=abc",
    ]
    direct = [
        "https://blog.csdn.net/a/b",
        "https://news.yiche.com/hao/wenzhang/104104150/",
    ]
    for u in redirects:
        assert not is_direct_url(u), f"应判为跳转链：{u}"
    for u in direct:
        assert is_direct_url(u), f"应判为可直连：{u}"
    assert not is_direct_url("ftp://x/y"), "非 http(s) 一律不算可直连"
    assert SearchHit(title="t", url=redirects[0]).is_direct is False
    print(f"  ✔ 跳转链判定：{len(redirects)} 个跳转链识别为不可直连，{len(direct)} 个正常链接放行")


def test_search_page_and_widget_filtering() -> None:
    """站内搜索页与引擎自家挂件页要剔除；跳转链是真实结果，必须保留。"""
    assert _is_search_page("https://tv.360kan.com/s?q=%E6%99%BA%E8%83%BD")
    assert _is_search_page("https://www.example.com/web?query=abc&x=1")
    assert not _is_search_page("https://blog.csdn.net/GUOSA0928/article/details/130964839")
    assert not _is_search_page("https://www.example.com/s"), "只有 /s 没有 q= 参数，不是搜索页"

    widget = SearchHit(title="360热门榜单", url="http://news.so.com/ns?q=abc")
    assert _is_widget_junk(widget), "引擎自家挂件页应剔除"
    assert _is_widget_junk(SearchHit(title="榜单", url="http://info.so.com/feedback.html"))
    assert _is_widget_junk(SearchHit(title="视频", url="https://tv.360kan.com/s?q=x"))

    redirect_hit = SearchHit(title="宝马 HUD 重影", url="http://www.baidu.com/link?url=abc")
    assert not _is_widget_junk(redirect_hit), "跳转链是真实搜索结果，不能当挂件丢掉"
    real = SearchHit(title="智能座舱语音交互技术解析", url="https://blog.csdn.net/a/b")
    assert not _is_widget_junk(real)
    print("  ✔ 噪声过滤：站内搜索页/引擎挂件页剔除，跳转链与真实文章保留")


def test_relevance_filter_drops_offtopic_hits() -> None:
    """实测 Bing 在中文多词查询下会返回完全跑题的结果，相关性过滤要拦住它。"""
    tokens = _query_tokens("智能座舱 语音交互 设计")
    assert "智能" in tokens and "座舱" in tokens, f"中文应切成 bigram：{sorted(tokens)[:8]}"
    assert "什么" not in _query_tokens("这是什么"), "疑问词应被剔除"

    good = SearchHit(title="VibeVoice在汽车领域的应用：智能座舱语音交互设计", url="https://a.com/1")
    bad = SearchHit(title="visio怎么画雷达图 - CSDN文库", url="https://b.com/2")
    assert _relevance_ok(good, tokens), "命中查询片段的结果应保留"
    assert not _relevance_ok(bad, tokens), "与查询毫无重叠的结果应剔除"

    # 摘要命中也算：标题没词但摘要里有，不能误杀
    snippet_hit = SearchHit(title="一篇评测", url="https://c.com/3", snippet="智能座舱语音交互体验")
    assert _relevance_ok(snippet_hit, tokens)

    # 查询切不出片段时（例如纯符号）不过滤，避免把所有结果一刀切光
    assert _relevance_ok(bad, set())
    print("  ✔ 相关性过滤：跑题结果（visio雷达图）被拦，标题或摘要命中即放行")


# ─────────────────────────── doc_id 稳定性 ───────────────────────────

def test_web_doc_id_stable_and_hashable_source() -> None:
    """doc_id 必须是 URL 的确定性函数。

    早期用内置 hash(url)（按进程随机加盐），同一 URL 换个进程就变成新文档，
    导致「同来源覆盖」失效、知识库堆重复。这里同时验证「函数确定性」和
    「真的能覆盖」——后者才是这个 bug 的真实后果。
    """
    url = "https://blog.csdn.net/a/article/details/123"
    docs = [WebDoc(url=url, title="标题", text="正文" * 50, ok=True)]
    ids = [to_documents(docs)[0].doc_id for _ in range(3)]
    assert len(set(ids)) == 1, f"同进程内 doc_id 应一致：{ids}"
    assert ids[0] == WebDoc(url=url, title="x", text="y", ok=True).doc_id()

    # 端到端：同一 URL 二次导入必须走「更新」而不是「新增」
    tmp = Path(tempfile.mkdtemp(prefix="kb_web_"))
    try:
        kb = KnowledgeBase(root=tmp)
        r1 = kb.add_documents(to_documents([WebDoc(url=url, title="标题", text="第一版正文内容。", ok=True)]),
                              source_type="联网抓取")
        r2 = kb.add_documents(to_documents([WebDoc(url=url, title="标题", text="第二版正文内容已更新。", ok=True)]),
                              source_type="联网抓取")
        assert (r1.added, r1.updated) == (1, 0), r1.summary()
        assert (r2.added, r2.updated) == (0, 1), f"同一 URL 应覆盖更新而非新增：{r2.summary()}"
        assert kb.stats()["n_docs"] == 1, "覆盖后不应出现两篇同源文档"
        print(f"  ✔ doc_id 稳定：同一 URL 二次导入 -> 新增{r1.added}/更新{r2.updated}，文档数保持 1")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_to_documents_skips_failed_and_empty() -> None:
    ok = WebDoc(url="https://a.com/1", title="T", text="正文内容足够长" * 10, ok=True)
    failed = WebDoc(url="https://a.com/2", title="T2", text="", ok=False, error="HTTP 403")
    empty = WebDoc(url="https://a.com/3", title="T3", text="   ", ok=True)
    docs = to_documents([ok, failed, empty])
    assert len(docs) == 1 and docs[0].source == "https://a.com/1"
    print("  ✔ 入库转换：抓取失败与空正文不入库，仅保留有效网页")


def test_fetch_url_failure_is_reported_not_raised() -> None:
    """抓取失败必须返回 ok=False + 中文归因，绝不抛异常（否则会中断整批导入）。

    不要断言具体的失败归因文案：同一个不可达地址，直连网络得到的是
    「请求失败（ConnectionError）」，而带代理的环境会被代理拦成 HTTP 502。
    这里只断言真正的不变量——不抛异常、ok=False、有归因文案。
    """
    d = fetch_url("http://127.0.0.1:9/never-listening", timeout=3)
    assert d.ok is False, "不可达地址应判为失败"
    assert d.error, "失败必须带归因文案（用户据此判断是否改走粘贴导入）"
    assert d.status_code is None or d.status_code >= 400
    assert len(d.text) == 0, "失败时不应带入任何正文"

    d2 = fetch_url("", timeout=3)
    assert d2.ok is False and "为空" in d2.error

    d3 = fetch_url("ftp://example.com/x", timeout=3)
    assert d3.ok is False, "非 http(s) 不应尝试抓取"
    print(f"  ✔ 失败归因：不可达 -> 「{d.error[:30]}…」，空 URL 与非 http 也被明确拒绝")


def test_engine_chain_order_and_members() -> None:
    """引擎链首位必须是 360：实测中文多词查询里它相关性最好且给真实地址。"""
    names = [n for n, _ in ENGINE_CHAIN]
    assert names[0] == "360", f"360 应排首位，实际 {names}"
    assert "baidu" in names and "bing" in names, "百度/Bing 应保留在链上作补充"
    assert len(names) == len(set(names)), "引擎名不应重复"
    print(f"  ✔ 引擎链：{' → '.join(names)}")


# ─────────────────────────── 真实联网（可选） ───────────────────────────

def test_live_search_and_direct_hits() -> None:
    if not LIVE:
        print("  · 跳过真实联网用例（设 RAG_LIVE_NET=1 开启）")
        return
    hits = web_search("智能座舱 语音交互", n=6)
    assert hits, "应至少返回 1 条"
    tokens = _query_tokens("智能座舱 语音交互")
    for h in hits:
        assert _relevance_ok(h, tokens), f"结果与查询无关却放行：{h.title}"
        assert not _is_widget_junk(h), f"噪声条目却放行：{h.url}"
    direct = [h for h in hits if h.is_direct]
    assert direct, "应至少有 1 条可直连抓取的结果"
    print(f"  ✔ 真实搜索：{len(hits)} 条命中（可直连 {len(direct)} 条），首位引擎 {hits[0].engine}")


def test_live_fetch_real_page() -> None:
    if not LIVE:
        print("  · 跳过真实抓取用例（设 RAG_LIVE_NET=1 开启）")
        return
    hits = [h for h in web_search("智能座舱 语音交互", n=8) if h.is_direct]
    assert hits, "应能找到可直连的结果"
    for h in hits[:4]:
        d = fetch_url(h.url, timeout=25, min_chars=200)
        if d.ok and len(d.text) >= 200:
            assert d.title or d.site
            print(f"  ✔ 真实抓取：{d.site} -> {len(d.text)} 字（标题「{(d.title or '')[:24]}…」）")
            return
    raise AssertionError("连试 4 个直连地址都没抓到正文（网络或站点反爬）")


TESTS = [
    ("网络与上传安全边界", test_network_and_upload_security_boundaries),
    ("跳转链判定", test_is_direct_url_classifies_redirect_chains),
    ("噪声过滤", test_search_page_and_widget_filtering),
    ("相关性过滤", test_relevance_filter_drops_offtopic_hits),
    ("doc_id 稳定与覆盖", test_web_doc_id_stable_and_hashable_source),
    ("入库转换", test_to_documents_skips_failed_and_empty),
    ("失败归因", test_fetch_url_failure_is_reported_not_raised),
    ("引擎链", test_engine_chain_order_and_members),
    ("真实搜索", test_live_search_and_direct_hits),
    ("真实抓取", test_live_fetch_real_page),
]


def main() -> int:
    print("═══ 联网导入测试 ═══")
    if not LIVE:
        print("（只跑离线用例；RAG_LIVE_NET=1 可加上真实联网用例）")
    passed = failed = 0
    for name, fn in TESTS:
        try:
            print(f"\n▶ {name}")
            fn()
            passed += 1
        except AssertionError as exc:
            failed += 1
            print(f"  ✘ 断言失败：{exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  ✘ {type(exc).__name__}: {exc}")
    print(f"\n结果：{passed}/{len(TESTS)} 通过" + (f"，{failed} 失败" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
