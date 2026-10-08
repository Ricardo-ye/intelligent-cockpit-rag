"""知识库命令行入口（无界面）：导入 / 联网抓取 / 检索 / 维护。

与工作台的「📚 知识库」页共用同一个存储目录（默认 data/kb，
可用环境变量 RAG_KB_DIR 覆盖），因此命令行导入的内容会立刻出现在工作台里。

示例：
    # 看当前知识库状态
    python run_kb_cli.py stats

    # 导入本地文件（支持 .txt .md .markdown .json .csv .pdf .docx .html .htm）
    python run_kb_cli.py add-file 手册.pdf --tags 座舱,手册
    python run_kb_cli.py add-file docs/ ./notes/          # 目录递归导入

    # 粘贴一段文本
    python run_kb_cli.py add-text --title "后排空调设定" --file notes.txt
    python run_kb_cli.py add-text --title "临时结论" --text "内容……"

    # 联网：先搜索，命中标题+摘要入档；--fetch 会把网页正文也抓下来
    python run_kb_cli.py search "智能座舱 语音交互 规范" --n 8
    python run_kb_cli.py search "车机 OTA 升级注意事项" --n 6 --import-top 3 --fetch

    # 直接按 URL 抓取正文
    python run_kb_cli.py add-url https://example.com/a.html https://example.com/b.html

    # 维护
    python run_kb_cli.py list
    python run_kb_cli.py remove <doc_id>
    python run_kb_cli.py export --out backup.json
    python run_kb_cli.py builtin          # 把内置真实样例语料导入知识库
    python run_kb_cli.py clear --yes
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# Windows 控制台默认 GBK，中文与 ✔/✘ 会炸；强制 UTF-8
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except Exception:
        pass

from rag_eval import KnowledgeBase, SUPPORTED_EXTENSIONS  # noqa: E402
from rag_eval.kb import MAX_DOC_CHARS, default_kb_dir  # noqa: E402
from rag_eval.webimport import WebImportError, search_and_fetch, web_search  # noqa: E402


# ─────────────────────────── 输出工具 ───────────────────────────

def hr(title: str = "") -> None:
    if title:
        print(f"\n── {title} ──")
    else:
        print()


def print_result(res, verbose: bool = True) -> None:
    """打印一次导入的结果（字段名与 kb.ImportResult / 清单条目一致）。"""
    print(f"  {res.summary()}")
    for d in res.docs:
        flag = {"added": "+", "updated": "~", "skipped": "="}.get(d.get("action", ""), "·")
        print(f"    [{flag}] {d.get('title', '')[:60]}"
              f"  ({d.get('chars', 0)} 字, v{d.get('version', 1)})")
        print(f"        id={d.get('doc_id', '')}")
    for e in res.errors:
        print(f"    ✘ {e}")
    if verbose:
        for m in res.messages:
            print(f"    · {m}")


def print_stats(kb: KnowledgeBase) -> None:
    s = kb.stats()
    hr("知识库状态")
    print(f"  存储目录：{s.get('root', default_kb_dir())}")
    print(f"  文档数　：{s.get('n_docs', 0)}")
    print(f"  总字数　：{s.get('total_chars', 0)}")
    print(f"  内容指纹：{kb.fingerprint()}")
    print(f"  最后更新：{s.get('updated_at') or '（空）'}")
    print(f"  历史版本：{s.get('n_versions', 0)} 次更新　截断文档：{s.get('n_truncated', 0)} 篇")
    by = s.get("by_type") or {}
    if by:
        chars = s.get("chars_by_type") or {}
        print("  来源分布：" + "　".join(
            f"{k}={v}篇/{chars.get(k, 0)}字" for k, v in sorted(by.items())))


# ─────────────────────────── 子命令 ───────────────────────────

def cmd_stats(kb: KnowledgeBase, a) -> int:
    print_stats(kb)
    return 0


def cmd_list(kb: KnowledgeBase, a) -> int:
    entries = kb.entries()
    hr(f"文档列表（{len(entries)} 篇）")
    if not entries:
        print("  （空）用 add-file / add-url / search --import-top 导入内容。")
        return 0
    for i, e in enumerate(entries, 1):
        print(f"  {i:>3}. {e.get('title', '')[:56]}")
        print(f"       id={e.get('doc_id')}　{e.get('chars', 0)} 字　v{e.get('version', 1)}　"
              f"来源={e.get('source_type', '-')}　更新={e.get('updated_at') or e.get('added_at')}")
        if e.get("source"):
            print(f"       source={str(e['source'])[:90]}")
        if e.get("truncated"):
            print("       ⚠ 正文超过上限已被截断")
    return 0


def _collect_paths(items: list[str]) -> list[Path]:
    out: list[Path] = []
    for it in items:
        p = Path(it).expanduser()
        if p.is_dir():
            out.extend(sorted(x for x in p.rglob("*") if x.is_file()))
        else:
            out.append(p)
    return out


def cmd_add_file(kb: KnowledgeBase, a) -> int:
    paths = _collect_paths(a.paths)
    if not paths:
        print("没有找到任何文件。")
        return 2
    hr(f"导入文件（{len(paths)} 个）")
    ok = failed = 0
    for p in paths:
        if not p.exists():
            print(f"  ✘ 不存在：{p}")
            failed += 1
            continue
        ext = p.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            print(f"  ✘ 不支持的类型 {ext}（{p.name}）；支持：{' '.join(sorted(SUPPORTED_EXTENSIONS))}")
            failed += 1
            continue
        try:
            res = kb.add_file(p, tags=a.tags)
        except Exception as exc:  # noqa: BLE001
            print(f"  ✘ {p.name}：{exc}")
            failed += 1
            continue
        if res.added or res.updated:
            ok += 1
        else:
            failed += res.failed
        print_result(res, verbose=False)
    print(f"\n  完成：成功 {ok}　失败 {failed}")
    return 0 if ok else 1


def _read_text(a) -> str:
    if a.file:
        return Path(a.file).read_text(encoding="utf-8", errors="replace")
    return a.text or ""


def cmd_add_text(kb: KnowledgeBase, a) -> int:
    text = _read_text(a)
    if not text.strip():
        print("没有内容：用 --text 或 --file 提供。")
        return 2
    res = kb.add_text(a.title, text, source=a.source, tags=a.tags)
    hr("粘贴文本导入")
    print_result(res)
    return 0 if (res.added or res.updated) else 1


def cmd_add_url(kb: KnowledgeBase, a) -> int:
    hr(f"按 URL 抓取（{len(a.urls)} 个）")
    ok = 0
    for u in a.urls:
        try:
            res = kb.add_url(u, tags=a.tags)
        except WebImportError as exc:
            print(f"  ✘ {u}\n      {exc}")
            continue
        if res.added or res.updated:
            ok += 1
        print_result(res, verbose=False)
        # add_url 失败时是把原因塞进 errors 的
        if res.failed and res.errors:
            for e in res.errors:
                print(f"      {e}")
    print(f"\n  完成：成功 {ok}／{len(a.urls)}")
    return 0 if ok else 1


def cmd_search(kb: KnowledgeBase, a) -> int:
    hits = web_search(a.query, n=a.n, timeout=a.timeout)
    hr(f"联网搜索「{a.query}」→ {len(hits)} 条命中")
    if not hits:
        print("  没有结果。可能：网络不通／搜索引擎改版／被限流。稍后重试或改用 add-url 直连。")
        return 1
    for i, h in enumerate(hits, 1):
        print(f"  {i:>2}. {h.title[:70]}")
        print(f"      {h.url}")
        if h.snippet:
            print(f"      {h.snippet[:110]}")

    if not a.import_top:
        print("\n  （仅搜索，未入档。加 --import-top N 收录前 N 条；再加 --fetch 抓取网页正文）")
        return 0

    top = hits[: a.import_top]
    if a.fetch:
        print(f"\n  抓取前 {len(top)} 条网页正文…")
        _, docs = search_and_fetch(a.query, n=a.import_top, timeout=a.timeout,
                                   min_chars=200)
        for d in docs:
            status = "OK" if d.ok else "失败"
            print(f"    [{status}] {d.title[:56]}　{len(d.text)} 字"
                  + (f"　({d.error})" if not d.ok else ""))
        res = kb.add_documents(
            [d.to_document() for d in docs if d.ok], source_type="联网抓取", tags=a.tags
        )
    else:
        res = kb.add_search_results(top, n=len(top), tags=a.tags)

    hr("入档结果")
    print_result(res)
    return 0 if (res.added or res.updated or res.skipped) else 1


def cmd_builtin(kb: KnowledgeBase, a) -> int:
    res = kb.add_builtin()
    hr("导入内置真实样例语料")
    print_result(res)
    return 0 if (res.added or res.updated or res.skipped) else 1


def cmd_remove(kb: KnowledgeBase, a) -> int:
    hr("删除文档")
    missing = [d for d in a.doc_ids if kb.get(d) is None]
    n = kb.remove_many([d for d in a.doc_ids if d not in missing])
    print(f"  已删除 {n}／{len(a.doc_ids)} 篇")
    for d in missing:
        print(f"    未找到：{d}")
    return 0 if n else 1


def cmd_export(kb: KnowledgeBase, a) -> int:
    p = kb.export_json(a.out)
    hr("导出")
    print(f"  → {p}")
    return 0


def cmd_clear(kb: KnowledgeBase, a) -> int:
    if not a.yes:
        print(f"清空会删除全部 {kb.stats().get('n_docs', 0)} 篇文档。确认请加 --yes。")
        return 2
    n = kb.clear()
    hr("清空知识库")
    print(f"  已清空 {n} 篇")
    return 0


def cmd_preview(kb: KnowledgeBase, a) -> int:
    """预览每篇文档开头。注意 kb.preview() 返回的是**中文表头**的表格行（无正文），
    要看正文得先拿 doc_id 再用 get() 取完整文档，两者不能混用。"""
    entries = kb.entries()
    if not entries:
        hr("内容预览")
        print("  （空）")
        return 0
    if a.doc_id:
        entries = [e for e in entries if e.get("doc_id") in set(a.doc_id)]
    hr(f"内容预览（{len(entries)} 篇，每篇前 {a.chars} 字）")
    for e in entries:
        doc = kb.get(e["doc_id"])
        print(f"\n  ◆ {e.get('title')}　[{e.get('doc_id')}]　{e.get('chars', 0)} 字　v{e.get('version', 1)}")
        if e.get("source"):
            print(f"    来源：{str(e['source'])[:100]}")
        print(f"    {(doc.text if doc else '')[: a.chars]}")
    return 0


# ─────────────────────────── 参数 ───────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="智能座舱 RAG 知识库命令行（与工作台共用同一存储）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"支持的文件类型：{'  '.join(sorted(SUPPORTED_EXTENSIONS))}\n"
               f"单篇上限：{MAX_DOC_CHARS:,} 字（超出截断并在结果里提示）",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("stats", help="查看知识库状态").set_defaults(func=cmd_stats)
    sub.add_parser("list", help="列出全部文档").set_defaults(func=cmd_list)

    f = sub.add_parser("add-file", help="导入本地文件或目录（递归）")
    f.add_argument("paths", nargs="+", help="文件或目录路径")
    f.add_argument("--tags", nargs="*", default=None, help="标签，空格分隔")
    f.set_defaults(func=cmd_add_file)

    t = sub.add_parser("add-text", help="粘贴一段文本入档")
    t.add_argument("--title", required=True, help="文档标题")
    t.add_argument("--text", default=None, help="正文（短文本直接给）")
    t.add_argument("--file", default=None, help="从 utf-8 文本文件读正文（长文本用这个）")
    t.add_argument("--source", default="手工粘贴", help="来源标识")
    t.add_argument("--tags", nargs="*", default=None)
    t.set_defaults(func=cmd_add_text)

    u = sub.add_parser("add-url", help="按 URL 抓取网页正文")
    u.add_argument("urls", nargs="+")
    u.add_argument("--tags", nargs="*", default=None)
    u.set_defaults(func=cmd_add_url)

    s = sub.add_parser("search", help="联网搜索，可选收录进知识库")
    s.add_argument("query")
    s.add_argument("--n", type=int, default=6, help="搜索返回条数（默认 6）")
    s.add_argument("--import-top", type=int, default=0, dest="import_top",
                   help="收录前 N 条（0 = 只搜索不入档）")
    s.add_argument("--fetch", action="store_true", help="抓取网页正文而非只用标题+摘要")
    s.add_argument("--timeout", type=int, default=20, help="单次请求超时秒数（默认 20）")
    s.add_argument("--tags", nargs="*", default=None)
    s.set_defaults(func=cmd_search)

    sub.add_parser("builtin", help="导入内置真实样例语料").set_defaults(func=cmd_builtin)

    r = sub.add_parser("remove", help="按 doc_id 删除")
    r.add_argument("doc_ids", nargs="+")
    r.set_defaults(func=cmd_remove)

    e = sub.add_parser("export", help="导出为 JSON 备份")
    e.add_argument("--out", default=None, help="输出路径（默认 data/kb/export_时间戳.json）")
    e.set_defaults(func=cmd_export)

    c = sub.add_parser("clear", help="清空知识库（危险）")
    c.add_argument("--yes", action="store_true", help="确认执行")
    c.set_defaults(func=cmd_clear)

    v = sub.add_parser("preview", help="预览每篇文档开头")
    v.add_argument("--chars", type=int, default=200, help="每篇截取字数（默认 200）")
    v.add_argument("--doc-id", nargs="*", default=None, dest="doc_id",
                   help="只看指定文档（缺省看全部）")
    v.set_defaults(func=cmd_preview)

    return p


def main() -> int:
    a = build_parser().parse_args()
    kb = KnowledgeBase()
    try:
        return a.func(kb, a)
    except KeyboardInterrupt:
        print("\n已中断。")
        return 130
    except Exception as exc:  # noqa: BLE001
        print(f"执行失败：{type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
