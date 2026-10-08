"""知识库模块测试：导入 / 覆盖更新 / 内容去重 / 持久化 / 统计 / 导出。

全部使用临时目录，不污染真实 data/kb。运行：python tests/test_kb.py
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_eval.ingest import load_json_corpus  # noqa: E402
from rag_eval.kb import KBError, MAX_DOC_CHARS, KnowledgeBase  # noqa: E402


def _kb() -> tuple:
    tmp = Path(tempfile.mkdtemp(prefix="kb_test_"))
    return KnowledgeBase(root=tmp), tmp


def test_text_import_dedupe_and_update() -> None:
    kb, tmp = _kb()
    try:
        r1 = kb.add_text("安全带使用", "安全带应使肩带跨过肩部中央，腰带贴紧髋部。", source="手册A")
        assert (r1.added, r1.updated, r1.skipped) == (1, 0, 0), r1.summary()
        assert kb.stats()["n_docs"] == 1

        # 同来源、同内容 → 跳过（不产生重复文档）
        r2 = kb.add_text("安全带使用", "安全带应使肩带跨过肩部中央，腰带贴紧髋部。", source="手册A")
        assert (r2.added, r2.updated, r2.skipped) == (0, 0, 1), r2.summary()
        assert kb.stats()["n_docs"] == 1

        # 同来源、内容变了 → 覆盖并升版本
        r3 = kb.add_text("安全带使用", "安全带应使肩带跨过肩部中央，腰带贴紧髋部，不可压在腋下。", source="手册A")
        assert (r3.added, r3.updated, r3.skipped) == (0, 1, 0), r3.summary()
        assert kb.stats()["n_docs"] == 1, "覆盖更新不应新增文档"
        entry = kb.entries()[0]
        assert entry["version"] == 2, f"覆盖后版本应为 2，实际 {entry['version']}"
        assert "腋下" in kb.documents()[0].text, "应保存更新后的正文"
        print(f"  ✔ 同来源覆盖更新：v{entry['version']}，文档数保持 {kb.stats()['n_docs']}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_content_dedupe_across_sources() -> None:
    """正文完全相同但来源不同 → 也应去重，避免同一篇资料进两次。"""
    kb, tmp = _kb()
    try:
        text = "当车辆检测到前方碰撞风险时，AEB 会先发出预警，必要时主动制动。"
        kb.add_text("AEB 说明", text, source="来源A")
        r = kb.add_text("AEB 说明（转载）", text, source="来源B")
        assert r.skipped == 1 and r.added == 0, r.summary()
        assert kb.stats()["n_docs"] == 1
        print("  ✔ 跨来源内容去重：同一正文只入库一次")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_file_import_all_supported_types() -> None:
    kb, tmp = _kb()
    src = tmp / "src"
    src.mkdir()
    try:
        (src / "a.txt").write_text("胎压过低会触发报警，建议每月检查一次冷态胎压。", encoding="utf-8")
        (src / "b.md").write_text(
            "# 空调使用\n\n## 除雾\n前风挡起雾时请开启前除雾档位。\n\n## 温度分区\n"
            "双温区空调可分别设定主副驾温度。\n", encoding="utf-8")
        (src / "c.csv").write_text(
            "title,text\n座椅记忆,座椅记忆可保存三组位置\n氛围灯,氛围灯支持多色调节\n", encoding="utf-8")
        (src / "d.json").write_text(json.dumps(
            {"documents": [{"doc_id": "d1", "title": "充电", "text": "建议使用原厂充电桩为车辆补能。"}]},
            ensure_ascii=False), encoding="utf-8")
        (src / "e.html").write_text(
            "<html><head><title>车机升级</title></head><body><nav>菜单</nav>"
            "<p>车机系统支持 OTA 在线升级，升级前请确保电量充足。</p></body></html>", encoding="utf-8")

        expect = {"a.txt": 1, "b.md": 2, "c.csv": 2, "d.json": 1, "e.html": 1}
        for name, n in expect.items():
            r = kb.add_file(src / name)
            assert r.failed == 0, f"{name} 导入失败：{r.errors}"
            assert r.added == n, f"{name} 应入库 {n} 篇，实际 {r.added}（{r.summary()}）"

        types = kb.stats()["by_type"]
        assert types.get("纯文本") == 1 and types.get("Markdown") == 2 and types.get("CSV 表格") == 2
        assert any("车机升级" in e["title"] for e in kb.entries()), "HTML 标题应被抽取"
        print(f"  ✔ 五类文件全部导入成功，共 {kb.stats()['n_docs']} 篇；类型分布 {types}")

        # 不支持的类型要明确报错，而不是静默吞掉
        bad = src / "f.xyz"
        bad.write_text("whatever", encoding="utf-8")
        rb = kb.add_file(bad)
        assert rb.failed == 1 and "不支持的类型" in rb.errors[0], rb.errors
        print("  ✔ 不支持的类型给出明确报错")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_persistence_and_stats() -> None:
    kb, tmp = _kb()
    try:
        kb.add_text("备胎更换", "更换备胎前请拉紧手刹并开启危险报警闪光灯。", source="手册B")
        kb.add_text("雨刮", "雨刮片建议每半年检查一次，老化会刮不干净。", source="手册B")
        # 新实例读同一目录 → 文档仍在（真实落盘，不是内存态）
        kb2 = KnowledgeBase(root=tmp)
        assert kb2.stats()["n_docs"] == 2, "知识库应持久化到磁盘"
        assert len(kb2.documents()) == 2
        st = kb2.stats()
        assert st["total_chars"] > 20 and st["updated_at"], st
        print(f"  ✔ 持久化 + 统计：{st['n_docs']} 篇 / {st['total_chars']} 字 / 更新于 {st['updated_at']}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_atomic_manifest_and_corruption_detection() -> None:
    kb, tmp = _kb()
    try:
        kb.add_text("文档一", "这是第一篇用于验证原子清单写入的文档。", source="测试")
        kb.add_text("文档二", "这是第二篇用于验证清单备份与恢复提示的文档。", source="测试")
        assert (tmp / "manifest.json.bak").exists(), "第二次保存后应保留清单备份"
        assert not list(tmp.glob(".manifest.json.*.tmp")), "原子替换后不应残留临时文件"

        (tmp / "manifest.json").write_text("{broken", encoding="utf-8")
        try:
            KnowledgeBase(root=tmp).stats()
            raise AssertionError("损坏清单不应被静默当成空知识库")
        except KBError as exc:
            assert "清单损坏" in str(exc) and "备份" in str(exc)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_fingerprint_changes_on_update() -> None:
    """指纹必须随内容变化——这是索引缓存失效的唯一依据。"""
    kb, tmp = _kb()
    try:
        assert kb.fingerprint() == "0", "空库指纹应为 0"
        f0 = kb.fingerprint()
        kb.add_text("车道保持", "车道保持辅助会在车辆偏离车道时轻微修正方向。", source="手册C")
        f1 = kb.fingerprint()
        assert f1 != f0, "新增文档后指纹必须变化"
        kb.add_text("车道保持", "车道保持辅助会在车辆偏离车道时修正方向，并提示驾驶员握好方向盘。", source="手册C")
        f2 = kb.fingerprint()
        assert f2 != f1, "覆盖更新后指纹也必须变化（否则索引不会重建）"
        print(f"  ✔ 指纹随内容变化：{f0} → {f1[:8]} → {f2[:8]}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_remove_clear_and_export() -> None:
    kb, tmp = _kb()
    try:
        kb.add_text("远光灯", "会车时应在 150 米以外切换为近光灯。", source="手册D")
        kb.add_text("雾灯", "雾天行驶应开启前后雾灯并降低车速。", source="手册D")
        assert kb.stats()["n_docs"] == 2

        ids = [e["doc_id"] for e in kb.entries()]
        assert kb.remove(ids[0]) is True
        assert kb.remove("不存在的id") is False
        assert kb.stats()["n_docs"] == 1

        out = kb.export_json(tmp / "kb_export.json")
        docs = load_json_corpus(out)
        assert len(docs) == 1 and "雾灯" in docs[0].text
        print("  ✔ 删除 / 导出：导出文件可被 load_json_corpus 直接复用")

        n = kb.clear()
        assert n == 1 and kb.is_empty() and kb.fingerprint() == "0"
        assert not (tmp / "docs").glob("*.json") or not list((tmp / "docs").glob("*.json"))
        print("  ✔ 清空知识库：文档与正文文件一并清除")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_oversize_text_is_truncated() -> None:
    """单篇超长文本必须截断并留有标记，避免把索引撑爆。"""
    kb, tmp = _kb()
    try:
        r = kb.add_text("超长文档", "汽" * (MAX_DOC_CHARS + 5000), source="压测")
        assert r.added == 1, r.summary()
        entry = kb.entries()[0]
        assert entry["chars"] == MAX_DOC_CHARS, f"应截断到 {MAX_DOC_CHARS}，实际 {entry['chars']}"
        assert entry["truncated"] is True
        assert any("已截断" in m for m in r.messages), r.messages
        assert kb.stats()["n_truncated"] == 1
        print(f"  ✔ 超长正文截断到 {MAX_DOC_CHARS} 字并留痕")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_blank_input_rejected() -> None:
    kb, tmp = _kb()
    try:
        r = kb.add_text("空", "   ", source="手工")
        assert r.failed == 1 and kb.stats()["n_docs"] == 0, "空文本不应入库"
        r2 = kb.add_file(tmp / "not-exist.txt")
        assert r2.failed == 1 and "不存在" in r2.errors[0]
        print("  ✔ 空文本 / 不存在的文件均被拒绝且给出原因")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


TESTS = [
    ("文本导入·去重·覆盖更新", test_text_import_dedupe_and_update),
    ("跨来源内容去重", test_content_dedupe_across_sources),
    ("五类文件导入", test_file_import_all_supported_types),
    ("持久化与统计", test_persistence_and_stats),
    ("原子清单与损坏检测", test_atomic_manifest_and_corruption_detection),
    ("指纹随内容变化", test_fingerprint_changes_on_update),
    ("删除/清空/导出", test_remove_clear_and_export),
    ("超长正文截断", test_oversize_text_is_truncated),
    ("空输入拒绝", test_blank_input_rejected),
]


def main() -> int:
    print("── 知识库模块验证 ──")
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
