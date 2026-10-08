"""文档加载与分块。

- PDF 用 pypdf 真实抽取文字（按页保留页码，便于溯源）。
- 分块采用「边界感知的递归切分 + 滑动重叠」，优先在段落/句号处断开，
  避免把一句话从中间劈开（这是影响召回质量的关键）。
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

# 由粗到细的分隔符：先按段落，再按句子，最后才硬切
SEPARATORS: Sequence[str] = ("\n\n", "\n", "。", "！", "？", "；", ". ", "! ", "? ", "; ", "，", ", ", " ")


class IngestError(RuntimeError):
    pass


@dataclass
class Document:
    doc_id: str
    title: str
    text: str
    source: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    doc_title: str
    text: str
    page: Optional[int] = None
    start: int = 0
    end: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ─────────────────────────────────────────────
# 加载
# ─────────────────────────────────────────────
def load_pdf(path: str | Path) -> List[Document]:
    """用 pypdf 逐页抽取真实文本，每页作为一个 Document。"""
    p = Path(path)
    if not p.exists():
        raise IngestError(f"PDF 不存在：{p}")
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception as exc:
        raise IngestError(f"缺少 pypdf，无法解析 PDF：{exc}。请执行 pip install pypdf。") from exc

    try:
        reader = PdfReader(str(p))
    except Exception as exc:
        raise IngestError(f"PDF 打开失败：{exc}") from exc

    docs: List[Document] = []
    stem = p.stem
    for i, page in enumerate(reader.pages, 1):
        try:
            text = (page.extract_text() or "").strip()
        except Exception:
            text = ""
        if text:
            docs.append(
                Document(
                    doc_id=f"{stem}#p{i}",
                    title=f"{stem} 第{i}页",
                    text=text,
                    source=f"{p.name} p.{i}",
                )
            )
    if not docs:
        raise IngestError(
            "PDF 中没有抽取到任何文本。若是扫描件/图片型 PDF，需要先做 OCR。"
        )
    return docs


def load_json_corpus(path: str | Path) -> List[Document]:
    """加载 JSON 语料：支持 [{doc_id,title,text}] 或 {documents:[...]}。"""
    p = Path(path)
    if not p.exists():
        raise IngestError(f"语料文件不存在：{p}")
    raw = json.loads(p.read_text(encoding="utf-8"))
    items = raw.get("documents", raw) if isinstance(raw, dict) else raw
    if not isinstance(items, list):
        raise IngestError("语料 JSON 结构不合法，应为数组或 {documents:[...]}。")

    docs: List[Document] = []
    for i, item in enumerate(items, 1):
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or item.get("content") or "").strip()
        if not text:
            continue
        docs.append(
            Document(
                doc_id=str(item.get("doc_id") or f"doc-{i}"),
                title=str(item.get("title") or f"文档{i}"),
                text=text,
                source=str(item.get("source") or "local-json"),
            )
        )
    if not docs:
        raise IngestError("语料文件中没有可用文本。")
    return docs


def load_txt(path: str | Path) -> List[Document]:
    p = Path(path)
    if not p.exists():
        raise IngestError(f"文件不存在：{p}")
    text = p.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        raise IngestError("文本文件为空。")
    return [Document(doc_id=p.stem, title=p.stem, text=text, source=p.name)]


# ─────────────────────────────────────────────
# 知识库导入：更多文件类型（供 kb.py 统一调用）
# ─────────────────────────────────────────────
_MD_IMG_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_MD_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_MD_EMPH_RE = re.compile(r"(\*\*|__|\*|_|`)")
_MD_HEAD_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.M)


def _clean_markdown(text: str) -> str:
    """把 Markdown 还原成可检索的纯文本：去掉图片、链接只留锚文本、去掉强调符与代码围栏。"""
    text = _MD_IMG_RE.sub("", text)
    text = _MD_LINK_RE.sub(r"\1", text)
    text = re.sub(r"^\s*```.*$", "", text, flags=re.M)
    text = _MD_EMPH_RE.sub("", text)
    text = re.sub(r"^\s*>\s?", "", text, flags=re.M)
    return text.strip()


def load_markdown(path: str | Path) -> List[Document]:
    """Markdown 按标题切成小节，每节一个 Document（标题即检索时的可读来源名）。"""
    p = Path(path)
    if not p.exists():
        raise IngestError(f"文件不存在：{p}")
    raw = _clean_markdown(p.read_text(encoding="utf-8", errors="ignore"))
    if not raw:
        raise IngestError("Markdown 文件为空。")

    heads = list(_MD_HEAD_RE.finditer(raw))
    if not heads:
        return [Document(doc_id=p.stem, title=p.stem, text=raw, source=p.name)]

    docs: List[Document] = []
    for i, m in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(raw)
        section = raw[m.start():end].strip()
        if len(section) < 10:
            continue
        docs.append(Document(
            doc_id=f"{p.stem}#s{i + 1:02d}",
            title=f"{p.stem} · {m.group(2).strip()[:40]}",
            text=section,
            source=p.name,
        ))
    return docs or [Document(doc_id=p.stem, title=p.stem, text=raw, source=p.name)]


def load_csv(path: str | Path) -> List[Document]:
    """CSV 导库：有 text/content 列时每行一篇；否则首列当标题、其余列拼成正文。"""
    import csv as _csv

    p = Path(path)
    if not p.exists():
        raise IngestError(f"文件不存在：{p}")
    with p.open("r", encoding="utf-8-sig", errors="ignore", newline="") as fh:
        rows = list(_csv.DictReader(fh))
    if not rows:
        raise IngestError("CSV 中没有数据行。")

    fields = [str(k).strip() for k in rows[0].keys() if k]
    text_col = next((c for c in fields if c.lower() in ("text", "content", "正文", "内容", "answer", "回答")), None)
    title_col = next((c for c in fields if c.lower() in ("title", "标题", "name", "question", "问题")), None)

    docs: List[Document] = []
    for i, row in enumerate(rows, 1):
        if text_col:
            body = str(row.get(text_col) or "").strip()
        else:
            body = " ".join(str(v or "").strip() for k, v in row.items() if k != title_col).strip()
        if len(body) < 2:
            continue
        title = str(row.get(title_col) or "").strip() if title_col else ""
        docs.append(Document(
            doc_id=f"{p.stem}#r{i:03d}",
            title=title or f"{p.stem} 第{i}行",
            text=body,
            source=f"{p.name} 第{i}行",
        ))
    if not docs:
        raise IngestError("CSV 中没有可用文本。")
    return docs


def load_docx(path: str | Path) -> List[Document]:
    """Word 文档：按 Heading 1/2 切成小节，段落与表格文本都保留。"""
    p = Path(path)
    if not p.exists():
        raise IngestError(f"文件不存在：{p}")
    try:
        import docx  # type: ignore
    except Exception as exc:
        raise IngestError(f"缺少 python-docx，无法解析 Word：{exc}。请执行 pip install python-docx。") from exc

    try:
        doc = docx.Document(str(p))
    except Exception as exc:
        raise IngestError(f"Word 打开失败：{exc}") from exc

    blocks: List[tuple] = []
    for para in doc.paragraphs:
        txt = (para.text or "").strip()
        if not txt:
            continue
        style = (para.style.name if para.style is not None else "") or ""
        blocks.append(("head" if style.lower().startswith("heading") else "body", txt))
    for table in doc.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text and c.text.strip()]
            if cells:
                blocks.append(("body", " | ".join(cells)))

    if not blocks:
        raise IngestError("Word 中没有抽取到任何文本。")

    docs: List[Document] = []
    cur_title: str = p.stem
    cur: List[str] = []
    for kind, txt in blocks:
        if kind == "head" and cur:
            docs.append(Document(doc_id=f"{p.stem}#d{len(docs) + 1:02d}",
                                 title=f"{p.stem} · {cur_title[:40]}",
                                 text="\n".join(cur), source=p.name))
            cur_title, cur = txt, []
        elif kind == "head":
            cur_title = txt
        else:
            cur.append(txt)
    if cur:
        docs.append(Document(doc_id=f"{p.stem}#d{len(docs) + 1:02d}",
                             title=f"{p.stem} · {cur_title[:40]}",
                             text="\n".join(cur), source=p.name))
    return docs


def load_html_file(path: str | Path) -> List[Document]:
    """本地 HTML 文件：复用 webimport 的正文抽取，避免两套清洗逻辑口径不一致。"""
    p = Path(path)
    if not p.exists():
        raise IngestError(f"文件不存在：{p}")
    from .webimport import html_to_text

    title, text = html_to_text(p.read_text(encoding="utf-8", errors="ignore"))
    if len(text) < 20:
        raise IngestError("HTML 中没有抽取到有效正文。")
    return [Document(doc_id=p.stem, title=title or p.stem, text=text, source=p.name)]


# 扩展名 → (加载函数, 中文类型名)
LOADERS: Dict[str, tuple] = {
    ".txt": (load_txt, "纯文本"),
    ".md": (load_markdown, "Markdown"),
    ".markdown": (load_markdown, "Markdown"),
    ".json": (load_json_corpus, "JSON 语料"),
    ".csv": (load_csv, "CSV 表格"),
    ".pdf": (load_pdf, "PDF"),
    ".docx": (load_docx, "Word"),
    ".html": (load_html_file, "网页(本地)"),
    ".htm": (load_html_file, "网页(本地)"),
}

SUPPORTED_EXTENSIONS: List[str] = sorted(LOADERS.keys())


def load_any(path: str | Path) -> List[Document]:
    """按扩展名选择加载器。不支持的类型会明确报出支持清单。"""
    p = Path(path)
    if not p.exists():
        raise IngestError(f"文件不存在：{p}")
    ext = p.suffix.lower()
    if ext not in LOADERS:
        raise IngestError(f"不支持的文件类型：{ext or '(无扩展名)'}。支持：{'、'.join(SUPPORTED_EXTENSIONS)}")
    loader, _ = LOADERS[ext]
    return loader(p)


# ─────────────────────────────────────────────
# 分块
# ─────────────────────────────────────────────
class RecursiveChunker:
    """边界感知的递归字符分块器（带重叠）。"""

    def __init__(self, chunk_size: int = 300, chunk_overlap: int = 60, min_chunk_chars: int = 40):
        if chunk_size <= 0:
            raise IngestError("chunk_size 必须为正数。")
        if chunk_overlap >= chunk_size:
            raise IngestError("chunk_overlap 必须小于 chunk_size。")
        self.chunk_size = int(chunk_size)
        self.chunk_overlap = int(chunk_overlap)
        self.min_chunk_chars = int(min_chunk_chars)

    # —— 核心：把长文本按分隔符递归切成 <= chunk_size 的片段 ——
    def _split_text(self, text: str, seps: Sequence[str]) -> List[str]:
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        sep = ""
        rest: Sequence[str] = ()
        for i, s in enumerate(seps):
            if s in text:
                sep, rest = s, seps[i + 1:]
                break

        if not sep:
            # 无可用分隔符 -> 硬切
            return [
                text[i:i + self.chunk_size]
                for i in range(0, len(text), self.chunk_size)
                if text[i:i + self.chunk_size].strip()
            ]

        parts, out = text.split(sep), []
        for p in parts:
            piece = p + sep if p is not parts[-1] else p
            if not piece.strip():
                continue
            if len(piece) <= self.chunk_size:
                out.append(piece)
            else:
                out.extend(self._split_text(piece, rest or SEPARATORS))
        return out

    # —— 合并小片段 + 加重叠 ——
    def _merge(self, pieces: List[str]) -> List[str]:
        chunks: List[str] = []
        buf = ""
        for piece in pieces:
            if not buf:
                buf = piece
                continue
            if len(buf) + len(piece) <= self.chunk_size:
                buf += piece
            else:
                chunks.append(buf)
                tail = buf[-self.chunk_overlap:] if self.chunk_overlap else ""
                buf = tail + piece
                while len(buf) > self.chunk_size:
                    chunks.append(buf[:self.chunk_size])
                    buf = buf[self.chunk_size - self.chunk_overlap:]
        if buf.strip():
            chunks.append(buf)
        return [c.strip() for c in chunks if len(c.strip()) >= self.min_chunk_chars] or [
            c.strip() for c in chunks if c.strip()
        ]

    def chunk_document(self, doc: Document) -> List[Chunk]:
        pieces = self._split_text(doc.text, SEPARATORS)
        merged = self._merge(pieces)
        out: List[Chunk] = []
        cursor = 0
        for i, text in enumerate(merged):
            start = doc.text.find(text[:30], cursor) if len(text) >= 30 else doc.text.find(text, cursor)
            if start < 0:
                start = cursor
            end = start + len(text)
            cursor = max(cursor, start)
            page = _page_of(doc, text)
            out.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}::c{i:03d}",
                    doc_id=doc.doc_id,
                    doc_title=doc.title,
                    text=text,
                    page=page,
                    start=start,
                    end=end,
                )
            )
        return out

    def chunk_all(self, docs: Iterable[Document]) -> List[Chunk]:
        chunks: List[Chunk] = []
        for d in docs:
            chunks.extend(self.chunk_document(d))
        if not chunks:
            raise IngestError("分块结果为空，请检查语料内容或减小 chunk_size。")
        return chunks


def _page_of(doc: Document, text: str) -> Optional[int]:
    """从形如 'xxx 第3页' 的标题里解析页码（PDF 场景）。"""
    if "#p" in doc.doc_id:
        try:
            return int(doc.doc_id.rsplit("#p", 1)[1])
        except Exception:
            return None
    return None


def corpus_hash(docs: Sequence[Document]) -> str:
    """语料指纹，用于缓存有效性校验。"""
    h = hashlib.sha256()
    for d in docs:
        h.update(d.doc_id.encode("utf-8"))
        h.update(d.text.encode("utf-8"))
    return h.hexdigest()[:16]


__all__ = [
    "Document", "Chunk", "IngestError", "RecursiveChunker",
    "load_pdf", "load_json_corpus", "load_txt", "corpus_hash",
    # 知识库导入支持的扩展加载器
    "load_markdown", "load_csv", "load_docx", "load_html_file", "load_any",
    "LOADERS", "SUPPORTED_EXTENSIONS",
]
