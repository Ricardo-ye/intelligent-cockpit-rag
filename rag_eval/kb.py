"""知识库：持久化存储 + 导入 + 版本化更新。

存放结构（默认 `data/kb/`）::

    data/kb/
      manifest.json          # 索引清单：每篇文档的元信息（标题/来源/字数/哈希/时间/版本）
      docs/<sha1>.json       # 每篇文档的正文，文件名用哈希保证跨平台安全

导入通道（全部真实落盘，重启后仍在）：
    上传文件  → add_file()        支持 .txt/.md/.json/.csv/.pdf/.docx/.html
    粘贴文本  → add_text()
    联网导入  → add_url() / add_search_results()
    内置样例  → add_builtin()

更新机制（三条规则，避免知识库越导越乱）：
    1. **同来源覆盖**：同一来源（文件名 / URL / 标题）再次导入 → 覆盖旧版本，version +1
    2. **同内容去重**：正文哈希相同 → 跳过，不产生重复文档
    3. **指纹驱动重建**：`fingerprint()` 变化后，问答/测评的索引缓存自动失效并重建
删除/清空是显式动作，不自动发生。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .config import DATA_DIR
from .ingest import (
    LOADERS, SUPPORTED_EXTENSIONS, Document, IngestError, clean_retrieval_text, corpus_hash, load_json_corpus,
)

def default_kb_dir() -> Path:
    """知识库默认位置：`data/kb`，可用环境变量 ``RAG_KB_DIR`` 覆盖。

    每次调用都重新读环境变量（而不是在模块加载时定死），这样测试可以把知识库
    指到临时目录、用户也可以把知识库放到别处，都不会污染项目内的真实数据。
    """
    env = os.environ.get("RAG_KB_DIR")
    return Path(env) if env else (DATA_DIR / "kb")


KB_DIR = default_kb_dir()
KB_DOCS_DIR = KB_DIR / "docs"
MANIFEST_NAME = "manifest.json"
MAX_DOC_CHARS = 300_000          # 单篇上限，防止超大网页把索引撑爆（超出部分截断并标注）

_SLUG_RE = re.compile(r"[^0-9A-Za-z\u4e00-\u9fff]+")
_QUALITY_NOISE_RE = re.compile(r"内容合作|商务合作|责任编辑|免责声明|联系我们|公众号|下载客户端|热门推荐|相关阅读")


class KBError(RuntimeError):
    pass


def _atomic_write_text(path: Path, content: str, *, backup: bool = False) -> None:
    """在目标目录内写临时文件后原子替换，避免进程中断留下半文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    if backup and path.exists():
        shutil.copy2(path, path.with_name(path.name + ".bak"))
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
            fh.write(content)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _slug(text: str, limit: int = 60) -> str:
    s = _SLUG_RE.sub("-", (text or "").strip()).strip("-")
    return s[:limit] or "doc"


def _short_hash(text: str, n: int = 8) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:n]


def _safe_filename(doc_id: str) -> str:
    """doc_id 里可能含 `:` `/` 等 Windows 非法字符，落盘前统一换成内容哈希。"""
    return hashlib.sha1(doc_id.encode("utf-8")).hexdigest()[:20] + ".json"


def _clip(text: str) -> tuple:
    """返回 (文本, 是否被截断)。"""
    if len(text) <= MAX_DOC_CHARS:
        return text, False
    return text[:MAX_DOC_CHARS], True


def assess_document_quality(text: str) -> Dict[str, Any]:
    """给入库文本一个可解释的保守质量提示，不自动删除用户文档。"""
    raw = (text or "").strip()
    clean = clean_retrieval_text(raw)
    warnings: List[str] = []
    if len(clean) < 80:
        warnings.append("正文过短")
    if _QUALITY_NOISE_RE.search(raw):
        warnings.append("含网页页脚或推荐位残留")
    if raw and len(clean) / len(raw) < 0.75:
        warnings.append("清洗前含较多结构噪声")
    # 问答聚合页通常混有大量无关标题；只提示人工复核，避免误删有效 FAQ。
    if len(re.findall(r"[？?]", clean)) >= 8:
        warnings.append("疑似多主题问答聚合页")
    score = max(0.0, round(1.0 - 0.18 * len(warnings), 2))
    return {"score": score, "warnings": warnings, "clean_chars": len(clean)}


@dataclass
class ImportResult:
    """一次导入的结果明细，供界面逐条展示（成功/更新/跳过/失败都要能看到）。"""
    added: int = 0
    updated: int = 0
    skipped: int = 0
    failed: int = 0
    docs: List[dict] = None          # type: ignore[assignment]
    errors: List[str] = None         # type: ignore[assignment]
    messages: List[str] = None       # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.docs = self.docs or []
        self.errors = self.errors or []
        self.messages = self.messages or []

    @property
    def total(self) -> int:
        return self.added + self.updated + self.skipped + self.failed

    def summary(self) -> str:
        return (f"新增 {self.added} 篇，更新 {self.updated} 篇，"
                f"去重跳过 {self.skipped} 篇，失败 {self.failed} 篇")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "added": self.added, "updated": self.updated, "skipped": self.skipped,
            "failed": self.failed, "docs": self.docs, "errors": self.errors,
            "messages": self.messages, "summary": self.summary(),
        }


class KnowledgeBase:
    """本地知识库。所有方法都真实读写磁盘，不使用内存假数据。"""

    def __init__(self, root: Optional[Path] = None):
        self.root = Path(root) if root else default_kb_dir()
        self.docs_dir = self.root / "docs"
        self.manifest_path = self.root / MANIFEST_NAME
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        self._manifest: Optional[dict] = None

    # ── 清单读写 ──────────────────────────────
    def _load(self) -> dict:
        if self._manifest is not None:
            return self._manifest
        if self.manifest_path.exists():
            try:
                self._manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            except Exception as exc:
                backup = self.manifest_path.with_name(self.manifest_path.name + ".bak")
                hint = f"；可尝试从备份恢复：{backup}" if backup.exists() else ""
                raise KBError(f"知识库清单损坏，已停止写入以避免数据丢失：{exc}{hint}") from exc
        else:
            self._manifest = {"version": 1, "updated_at": "", "entries": []}
        if not isinstance(self._manifest, dict) or not isinstance(self._manifest.get("entries", []), list):
            raise KBError("知识库清单结构不合法：根节点必须是对象，entries 必须是数组。")
        self._manifest.setdefault("entries", [])
        return self._manifest

    def _save(self) -> None:
        m = self._load()
        m["version"] = 1
        m["updated_at"] = _now()
        self.root.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(
            self.manifest_path,
            json.dumps(m, ensure_ascii=False, indent=2),
            backup=True,
        )

    # ── 读 ────────────────────────────────────
    def entries(self) -> List[dict]:
        return list(self._load()["entries"])

    def is_empty(self) -> bool:
        return not self._load()["entries"]

    def get(self, doc_id: str) -> Optional[Document]:
        entry = next((e for e in self._load()["entries"] if e["doc_id"] == doc_id), None)
        if not entry:
            return None
        path = self.docs_dir / entry["file"]
        if not path.exists():
            return None
        raw = json.loads(path.read_text(encoding="utf-8"))
        return Document(doc_id=raw["doc_id"], title=raw["title"], text=raw["text"],
                        source=raw.get("source", ""))

    def documents(self) -> List[Document]:
        """按清单顺序返回全部文档（顺序稳定，保证指纹稳定）。"""
        out: List[Document] = []
        for e in self._load()["entries"]:
            d = self.get(e["doc_id"])
            if d is not None:
                out.append(d)
        return out

    def stats(self) -> Dict[str, Any]:
        entries = self._load()["entries"]
        by_type: Dict[str, int] = {}
        chars_by_type: Dict[str, int] = {}
        for e in entries:
            t = e.get("source_type", "未知")
            by_type[t] = by_type.get(t, 0) + 1
            chars_by_type[t] = chars_by_type.get(t, 0) + int(e.get("chars", 0))
        return {
            "n_docs": len(entries),
            "total_chars": sum(int(e.get("chars", 0)) for e in entries),
            "by_type": by_type,
            "chars_by_type": chars_by_type,
            "updated_at": self._load().get("updated_at", ""),
            "root": str(self.root),
            "n_truncated": sum(1 for e in entries if e.get("truncated")),
            "n_versions": sum(int(e.get("version", 1)) - 1 for e in entries),
            "n_quality_warnings": sum(1 for e in entries if (e.get("quality") or {}).get("warnings")),
        }

    def fingerprint(self) -> str:
        """内容指纹：知识库一变，索引缓存即失效。空库返回 '0'。"""
        docs = self.documents()
        return corpus_hash(docs) if docs else "0"

    def preview(self, limit: int = 400) -> List[dict]:
        rows = []
        for e in self._load()["entries"]:
            rows.append({
                "doc_id": e["doc_id"],
                "标题": e.get("title", ""),
                "类型": e.get("source_type", ""),
                "字数": e.get("chars", 0),
                "版本": e.get("version", 1),
                "来源": (e.get("source") or "")[:80],
                "更新时间": e.get("updated_at", ""),
                "质量提示": "；".join((e.get("quality") or {}).get("warnings", [])) or "—",
            })
        return rows[:limit] if limit else rows

    def audit_quality(self, include_clean: bool = False) -> List[dict]:
        """审计现有文档质量；仅报告，不会自动移除任何用户数据。"""
        rows: List[dict] = []
        for entry in self._load()["entries"]:
            doc = self.get(entry["doc_id"])
            if doc is None:
                continue
            quality = assess_document_quality(doc.text)
            if include_clean or quality["warnings"]:
                rows.append({
                    "doc_id": doc.doc_id, "title": doc.title, "source": doc.source,
                    **quality,
                })
        return rows

    # ── 写 ────────────────────────────────────
    def _write_doc(self, doc: Document, source_type: str, tags: Sequence[str],
                    version: int, created_at: str,
                    quality: Optional[Dict[str, Any]] = None) -> dict:
        text, truncated = _clip(doc.text)
        quality = quality or assess_document_quality(text)
        chash = _short_hash(text, 16)
        fname = _safe_filename(doc.doc_id)
        _atomic_write_text(
            self.docs_dir / fname,
            json.dumps({"doc_id": doc.doc_id, "title": doc.title, "text": text,
                        "source": doc.source}, ensure_ascii=False),
        )
        return {
            "doc_id": doc.doc_id, "title": doc.title, "file": fname,
            "source": doc.source, "source_type": source_type,
            "tags": list(tags), "chars": len(text), "content_hash": chash,
            "truncated": truncated, "version": version,
            "quality": quality,
            "added_at": created_at, "updated_at": _now(),
        }

    def add_documents(self, docs: Iterable[Document], source_type: str = "文件",
                      tags: Optional[Sequence[str]] = None) -> ImportResult:
        """批量入库。按「同来源覆盖 / 同内容去重」规则处理，返回逐条明细。"""
        res = ImportResult()
        m = self._load()
        entries = m["entries"]
        by_id = {e["doc_id"]: e for e in entries}
        seen_hashes = {e.get("content_hash") for e in entries}

        for doc in docs:
            raw_quality = assess_document_quality(doc.text)
            cleaned = clean_retrieval_text(doc.text)
            if not cleaned:
                res.skipped += 1
                res.messages.append(f"跳过空文档：{doc.doc_id}")
                continue

            # 持久化清洗后的副本。原始来源 URL/文件名仍在 metadata 中，可回溯；
            # 这里只去除确定的网页模板噪声，不改变可用的业务段落。
            doc = Document(doc_id=doc.doc_id, title=doc.title, text=cleaned, source=doc.source)

            text, truncated = _clip(doc.text)
            chash = _short_hash(text, 16)
            old = by_id.get(doc.doc_id)

            if old is None and chash in seen_hashes:
                # 内容重复（例如同一篇网页换了 URL）→ 不重复入库
                res.skipped += 1
                res.messages.append(f"内容重复，已跳过：{doc.title[:40]}")
                continue

            if old is not None and old.get("content_hash") == chash:
                res.skipped += 1
                res.messages.append(f"内容未变，无需更新：{doc.title[:40]}")
                continue

            version = int(old.get("version", 1)) + 1 if old else 1
            created = old.get("added_at") if old else _now()
            entry = self._write_doc(
                doc, source_type, tags or [], version, created or _now(), quality=raw_quality,
            )

            if old is not None:
                if old.get("file") and old["file"] != entry["file"]:
                    (self.docs_dir / old["file"]).unlink(missing_ok=True)
                entries[entries.index(old)] = entry
                res.updated += 1
                res.messages.append(f"已更新（v{version}）：{doc.title[:40]}")
            else:
                entries.append(entry)
                res.added += 1
                res.messages.append(f"已新增：{doc.title[:40]}")

            if truncated:
                res.messages.append(f"⚠ 正文超过 {MAX_DOC_CHARS} 字已截断：{doc.title[:40]}")
            quality = entry.get("quality") or {}
            if quality.get("warnings"):
                res.messages.append(f"⚠ 建议复核资料质量：{doc.title[:40]}（{'；'.join(quality['warnings'])}）")
            by_id[doc.doc_id] = entry
            seen_hashes.add(chash)
            res.docs.append({"doc_id": doc.doc_id, "title": doc.title,
                             "chars": len(text), "version": version})

        if res.added or res.updated:
            self._save()
        return res

    def add_text(self, title: str, text: str, source: str = "手工粘贴",
                 source_type: str = "粘贴文本", tags: Optional[Sequence[str]] = None) -> ImportResult:
        text = (text or "").strip()
        if len(text) < 5:
            r = ImportResult(failed=1)
            r.errors.append("文本太短（<5 字），未入库。")
            return r
        title = (title or "").strip() or (text[:24] + ("…" if len(text) > 24 else ""))
        doc_id = f"{_slug(source)}::{_slug(title)}-{_short_hash(source + title)}"
        doc = Document(doc_id=doc_id, title=title, text=text, source=source)
        return self.add_documents([doc], source_type=source_type, tags=tags)

    def add_file(self, path: str | Path, source_type: Optional[str] = None,
                 tags: Optional[Sequence[str]] = None) -> ImportResult:
        """按扩展名加载并入库。同一文件重复导入 → 覆盖为最新版本。"""
        p = Path(path)
        if not p.exists():
            r = ImportResult(failed=1)
            r.errors.append(f"文件不存在：{p}")
            return r
        ext = p.suffix.lower()
        loader, type_name = LOADERS.get(ext, (None, None))
        if loader is None:
            r = ImportResult(failed=1)
            r.errors.append(f"不支持的类型 {ext or '(无扩展名)'}。支持：{'、'.join(SUPPORTED_EXTENSIONS)}")
            return r
        try:
            docs = loader(p)
        except (IngestError, Exception) as exc:  # noqa: BLE001
            r = ImportResult(failed=1)
            r.errors.append(f"{p.name} 解析失败：{exc}")
            return r

        # 命名空间化 doc_id：同一文件重复导入即视为"同来源"，从而走覆盖更新
        ns = _slug(p.stem, 40)
        namespaced = [
            Document(doc_id=f"{ns}::{d.doc_id}", title=d.title, text=d.text,
                     source=d.source or p.name)
            for d in docs
        ]
        return self.add_documents(namespaced, source_type=source_type or type_name, tags=tags)

    def add_url(self, url: str, tags: Optional[Sequence[str]] = None) -> ImportResult:
        """抓取单个 URL 并入库。抓取失败只记错误，不会污染知识库。"""
        from .webimport import fetch_url

        d = fetch_url(url)
        if not d.ok:
            r = ImportResult(failed=1)
            r.errors.append(f"{url} → {d.error}")
            return r
        # 统一走 WebDoc.to_document()：网页 doc_id 只能有一套生成规则。
        # 早先这里自己拼 `web::<site>::<hash>`，与 to_documents() 的 `web-<site>-<sha1>`
        # 不一致 —— 同一个 URL 从命令行导入和从界面导入会拿到两个 doc_id，
        # 于是同一篇网页被存成两份，「同来源覆盖」形同虚设。
        return self.add_documents([d.to_document()], source_type="联网网页", tags=tags)

    def add_search_results(self, hits: Sequence[Any], n: int = 3,
                           tags: Optional[Sequence[str]] = None) -> ImportResult:
        """把搜索结果前 n 条抓取后入库（逐条记录成功与失败）。"""
        from .webimport import fetch_url

        res = ImportResult()
        ok = 0
        for h in hits:
            if ok >= n:
                break
            d = fetch_url(h.url)
            if not d.ok:
                res.failed += 1
                res.errors.append(f"{h.title[:30]} → {d.error}")
                continue
            # 与 add_url / to_documents 共用同一套 doc_id（见 add_url 注释）
            doc = d.to_document()
            doc.title = d.title or h.title
            sub = self.add_documents([doc], source_type="联网网页", tags=tags)
            res.added += sub.added
            res.updated += sub.updated
            res.skipped += sub.skipped
            res.failed += sub.failed
            res.docs.extend(sub.docs)
            res.messages.extend(sub.messages)
            res.errors.extend(sub.errors)
            ok += 1
        return res

    def add_builtin(self) -> ImportResult:
        """导入内置真实样例语料（11 篇智能座舱知识），方便冷启动立刻可用。"""
        from .config import SAMPLE_CORPUS_PATH

        try:
            docs = load_json_corpus(SAMPLE_CORPUS_PATH)
        except Exception as exc:  # noqa: BLE001
            r = ImportResult(failed=1)
            r.errors.append(f"内置语料加载失败：{exc}")
            return r
        namespaced = [Document(doc_id=f"内置样例::{d.doc_id}", title=d.title,
                               text=d.text, source="data/sample_corpus.json") for d in docs]
        return self.add_documents(namespaced, source_type="内置样例")

    def remove(self, doc_id: str) -> bool:
        m = self._load()
        entry = next((e for e in m["entries"] if e["doc_id"] == doc_id), None)
        if not entry:
            return False
        (self.docs_dir / entry["file"]).unlink(missing_ok=True)
        m["entries"].remove(entry)
        self._save()
        return True

    def remove_many(self, doc_ids: Sequence[str]) -> int:
        n = sum(1 for d in list(doc_ids) if self.remove(d))
        return n

    def clear(self) -> int:
        n = len(self._load()["entries"])
        self._manifest = {"version": 1, "updated_at": _now(), "entries": []}
        self._save()
        if self.docs_dir.exists():
            shutil.rmtree(self.docs_dir, ignore_errors=True)
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        return n

    def export_json(self, path: Optional[str | Path] = None) -> Path:
        """导出为可被 `load_json_corpus` 复用的标准 JSON 语料（便于迁移/留痕）。"""
        out = Path(path) if path else (DATA_DIR / "reports" / f"kb_export_{datetime.now():%Y%m%d_%H%M%S}.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(
            {"documents": [d.to_dict() for d in self.documents()], "exported_at": _now()},
            ensure_ascii=False, indent=2), encoding="utf-8")
        return out


__all__ = [
    "KnowledgeBase", "KBError", "ImportResult", "KB_DIR", "KB_DOCS_DIR",
    "default_kb_dir", "SUPPORTED_EXTENSIONS", "MAX_DOC_CHARS",
]

