"""联网导入：抓取网页正文 + 关键词搜索候选资料。

设计原则：
- **只用标准库 + requests + lxml**（本环境已装），不引入新依赖；lxml 缺失时退回正则抽取。
- 每个网络动作都有超时、UA、编码纠正与明确的中文错误归因；失败**绝不**污染知识库。
- 抓回来的正文会做真实清洗（去脚本/样式/导航/页脚、压缩空白），并保留来源 URL 便于溯源。

对外接口：
    fetch_url(url)              -> WebDoc          单页抓取
    web_search(query, n)        -> List[SearchHit] 关键词搜索（多引擎累积去重）
    search_and_fetch(q, n)      -> List[WebDoc]    搜索并抓取前 n 篇可用正文
    to_documents(webdocs)       -> List[Document]  转成可入库的文档
    is_direct_url(url)          -> bool            该链接能否直连抓取

引擎链（ENGINE_CHAIN）按实测可用性排序：360 → 百度 → 搜狗 → Bing → DuckDuckGo。
中文多词查询下 Bing 会退化成单字匹配，DuckDuckGo 在部分网络下不可达，故仅作兜底。
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import parse_qs, unquote, urljoin, urlparse

from .security import SecurityError, validate_public_http_url

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
DEFAULT_TIMEOUT = 20
MAX_HTML_BYTES = 4 * 1024 * 1024       # 单页最大 4MB，防止抓到大文件把内存吃满
MAX_REDIRECTS = 5

# 正文清洗时直接丢弃的标签（导航/脚本/装饰，留着只会污染检索）
_DROP_TAGS = (
    "script", "style", "noscript", "iframe", "svg", "form", "button", "input",
    "nav", "header", "footer", "aside", "template",
)
# 常见的正文容器，命中优先，命中不了就退回全页文本
_CONTENT_HINTS = (
    "article", "main", '[role="main"]', ".article-content", ".article_content",
    ".rich_media_content", "#js_content", ".content", ".post-content", ".entry-content",
)


class WebImportError(RuntimeError):
    """联网导入失败（网络不可达 / 站点拒绝 / 无正文）。"""


@dataclass
class WebDoc:
    url: str
    title: str = ""
    text: str = ""
    ok: bool = True
    error: Optional[str] = None
    site: str = ""
    status_code: Optional[int] = None
    fetched_at: str = ""

    @property
    def chars(self) -> int:
        return len(self.text or "")

    def doc_id(self) -> str:
        """稳定的文档 ID：同一 URL 永远得到同一个 ID（sha1，与进程无关）。

        早期版本用内置 `hash(url)`：Python 对 str 的 hash 默认按进程随机加盐，
        于是同一个 URL 换一次进程就变成另一篇文档 —— 「同来源覆盖」失效、
        知识库里不断堆积重复网页。sha1 是确定性的，可复现。
        """
        digest = hashlib.sha1((self.url or "").encode("utf-8")).hexdigest()[:12]
        return f"web-{(self.site or 'page')}-{digest}"

    def to_document(self):
        """转成可入库的 ingest.Document（保留 URL 作为 source，便于溯源）。"""
        from .ingest import Document

        return Document(
            doc_id=self.doc_id(),
            title=(self.title or self.url)[:120],
            text=self.text,
            source=self.url,
        )

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["chars"] = self.chars
        return d


@dataclass
class SearchHit:
    title: str
    url: str
    snippet: str = ""
    engine: str = ""

    @property
    def is_direct(self) -> bool:
        """URL 是否可直连抓取（跳转链在部分网络下会被拦，抓取前先判断能省一次无效请求）。"""
        return is_direct_url(self.url)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["is_direct"] = self.is_direct
        return d


# 搜索引擎的跳转链：百度 link?url= / 360 so.com/link / 搜狗 link?url= / 百度广告 baidu.php
# 实测这些地址在带代理的网络下常被拦（502/400/空正文），所以抓正文时要优先挑直连 URL。
_REDIRECT_HINTS = (
    "baidu.com/link?", "baidu.php", "so.com/link?", "sogou.com/link?",
    "www.google.com/url?", "bing.com/ck/a", "duckduckgo.com/l/?uddg=",
)


def is_direct_url(url: str) -> bool:
    u = (url or "").lower()
    if not u.startswith("http"):
        return False
    return not any(h in u for h in _REDIRECT_HINTS)


# 引擎自家页面（「相关消息」「反馈」「相关软件」「视频/榜单」这类小挂件），不是正文内容，必须剔除
_ENGINE_HOSTS = (
    "so.com", "baidu.com", "sogou.com", "bing.com", "duckduckgo.com",
    "google.com", "sogoucdn.com", "baidustatic.com", "360.cn", "360kan.com",
)
# 站内搜索页/聚合页：抓下来只有壳，没有正文（例：tv.360kan.com/s?q=...、www.so.com/s?q=...）
_SEARCH_PATH_RE = re.compile(r"/(s|web|search|so|sou|query)/?$", re.I)
_SEARCH_KEYS = ("q", "wd", "query", "word", "keyword", "k")


def _is_engine_internal(url: str) -> bool:
    host = (urlparse(url).netloc or "").lower().split(":")[0]
    return any(host == h or host.endswith("." + h) for h in _ENGINE_HOSTS)


def _is_search_page(url: str) -> bool:
    """站内搜索页判定。用 parse_qs 而不是写正则匹配 `q=`：
    正则里 `(^|&)` 在字符串中间永远不匹配（`^` 只在 pos 0 成立），
    早先那版把 `<host>/s?q=xxx` 这种最典型的形态全漏掉了。
    """
    p = urlparse(url)
    if not _SEARCH_PATH_RE.search(p.path or ""):
        return False
    return any(k in parse_qs(p.query or "") for k in _SEARCH_KEYS)


# 查询词里没有检索价值的碎片（疑问词/助词），参与相关性判定只会引入噪声
_QUERY_STOP = {
    "什么", "怎么", "如何", "哪些", "哪个", "为什么", "是否", "可以", "需要",
    "以及", "还有", "介绍", "说明", "一下", "我们", "你们", "这个", "那个",
}
_LATIN_WORD_RE = re.compile(r"[a-z0-9][a-z0-9\-\.]{1,}", re.I)
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def _query_tokens(query: str) -> set[str]:
    """把查询词切成可判重的片段：英文/数字词 + 中文二元组（bigram）。

    中文没有词边界，用 bigram 是最省事的近似：只要结果里出现「智能座舱」「语音交互」
    任意一个片段，就认为它跟查询有关。
    """
    q = (query or "").lower()
    tokens = set(_LATIN_WORD_RE.findall(q))
    cjk = "".join(_CJK_RE.findall(q))
    for i in range(len(cjk) - 1):
        tokens.add(cjk[i:i + 2])
    return {t for t in tokens if t not in _QUERY_STOP}


def _relevance_ok(h: SearchHit, tokens: set[str]) -> bool:
    """标题+摘要里至少要命中一个查询片段，否则判为无关结果。

    为什么需要：实测 Bing 的 HTML 端点在中文多词查询下会返回**完全跑题**的结果
    （查「智能座舱 语音交互 设计」，回回来一堆「visio 怎么画雷达图」）。
    多引擎累积时这类噪声会混进候选列表，用户还得自己筛。标题/摘要里连一个查询片段
    都没有的条目，直接丢掉更干净。命中不了任何片段也可能是词表切分问题，
    所以只在「查询本身有片段」时才启用这条过滤。
    """
    if not tokens:
        return True
    hay = f"{h.title} {h.snippet}".lower()
    return any(t in hay for t in tokens)


def _is_widget_junk(h: SearchHit) -> bool:
    """引擎自家的挂件页（相关消息/反馈/相关软件/榜单）应剔除。

    注意只剔除「可直连的引擎域名」：`baidu.com/link?url=` 这类跳转链虽然 host 也是引擎域名，
    但它是**真实的搜索结果链接**（标题+摘要仍有价值，可入档），不能一起丢掉。
    """
    if not h.is_direct:
        return False
    return _is_engine_internal(h.url) or _is_search_page(h.url)


# ─────────────────────────────────────────────
# HTML → 正文
# ─────────────────────────────────────────────
_WS_RE = re.compile(r"[ \t\u00a0\u3000]+")
_MULTI_NL_RE = re.compile(r"\n{3,}")
_TAG_RE = re.compile(r"<(script|style|noscript)[^>]*>.*?</\1>", re.S | re.I)
_BLOCK_RE = re.compile(r"</(p|div|li|tr|h[1-6]|section|article|br)>", re.I)


def _normalize(text: str) -> str:
    text = _WS_RE.sub(" ", text or "")
    lines = [ln.strip() for ln in text.split("\n")]
    text = "\n".join(ln for ln in lines)
    return _MULTI_NL_RE.sub("\n\n", text).strip()


def _text_with_lxml(html: str) -> Tuple[str, str]:
    """用 lxml 做真实 DOM 级抽取：先丢噪音标签，再优先取正文容器。"""
    from lxml import html as LH  # 局部导入：缺失时走正则兜底

    tree = LH.fromstring(html)
    for bad in tree.xpath("|".join(f"//{t}" for t in _DROP_TAGS)):
        parent = bad.getparent()
        if parent is not None:
            parent.remove(bad)

    title = ""
    for xp in ("//meta[@property='og:title']/@content", "//title/text()", "//h1//text()"):
        got = tree.xpath(xp)
        if got and str(got[0]).strip():
            title = str(got[0]).strip()
            break

    body = None
    for sel in _CONTENT_HINTS:
        try:
            nodes = tree.cssselect(sel)
        except Exception:  # cssselect 未安装时退回 xpath
            nodes = tree.xpath("//*[contains(@class,'content') or contains(@class,'article')]")
        if nodes:
            best = max(nodes, key=lambda n: len(n.text_content() or ""))
            if len(best.text_content() or "") > 200:
                body = best
                break
    body = body if body is not None else (tree.body if tree.body is not None else tree)

    blocks = body.xpath("//p|//li|//h1|//h2|//h3|//h4|//td|//div[not(*)]")
    if blocks and len(blocks) >= 3:
        seen, lines = set(), []
        for b in blocks:
            txt = _normalize(b.text_content() or "")
            if len(txt) < 8 or txt in seen:
                continue
            seen.add(txt)
            lines.append(txt)
        text = "\n".join(lines)
    else:
        text = _normalize(body.text_content() or "")
    return title, text


def _text_with_regex(html: str) -> Tuple[str, str]:
    """lxml 不可用时的兜底抽取（够用，但精度低于 DOM 抽取）。"""
    m = re.search(r"<title[^>]*>(.*?)</title>", html, re.S | re.I)
    title = re.sub(r"\s+", " ", m.group(1)).strip() if m else ""
    body = _TAG_RE.sub(" ", html)
    body = _BLOCK_RE.sub("\n", body)
    body = re.sub(r"<[^>]+>", " ", body)
    body = (
        body.replace("&nbsp;", " ").replace("&amp;", "&").replace("&lt;", "<")
        .replace("&gt;", ">").replace("&quot;", '"').replace("&#39;", "'")
    )
    return title, _normalize(body)


def html_to_text(html: str) -> Tuple[str, str]:
    """返回 (标题, 正文)。

    只要 lxml 抽取**没有抛异常**就用它的结果——早期版本拿「正文长度≥80」当成功判据，
    短页面会被误判成抽取失败而退回正则路径，于是导航/标题这类噪音文字全被带进正文。
    长度是否达标属于调用方（`fetch_url` 的 min_chars）的判断，
    在这里把它和"抽取失败"混为一谈是错的。
    """
    if not html:
        return "", ""
    try:
        title, text = _text_with_lxml(html)
        if (text or "").strip():
            return title, text
    except Exception:
        pass
    return _text_with_regex(html)


# ─────────────────────────────────────────────
# 抓取
# ─────────────────────────────────────────────
def _decode(resp, raw: Optional[bytes] = None) -> str:
    """中文站点经常不声明 charset 或被 requests 猜成 ISO-8859-1，这里按字节真实判定。"""
    raw = raw if raw is not None else (resp.content or b"")
    enc = (resp.encoding or "").lower()
    if not enc or enc in ("iso-8859-1", "ascii"):
        # 流式响应关闭后不再访问 resp.content；优先尝试 UTF-8，再覆盖常见中文编码。
        for candidate in ("utf-8", "gb18030"):
            try:
                return raw.decode(candidate)
            except UnicodeDecodeError:
                continue
        enc = "utf-8"
    try:
        return raw.decode(enc, errors="replace")
    except LookupError:
        return raw.decode("utf-8", errors="replace")


def fetch_url(url: str, timeout: int = DEFAULT_TIMEOUT, min_chars: int = 80) -> WebDoc:
    """抓取单个 URL 并抽取正文。失败返回 ok=False 的 WebDoc，不抛异常。"""
    url = (url or "").strip()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if not url:
        return WebDoc(url="", ok=False, error="URL 为空", fetched_at=now)
    if not re.match(r"^https?://", url, re.I):
        url = "https://" + url

    try:
        import requests
    except Exception as exc:  # noqa: BLE001
        return WebDoc(url=url, ok=False, error=f"缺少 requests 库：{exc}", fetched_at=now)

    site = urlparse(url).netloc
    try:
        current = url
        resp = None
        for _ in range(MAX_REDIRECTS + 1):
            validate_public_http_url(current)
            resp = requests.get(
                current, timeout=timeout, headers={
                    "User-Agent": USER_AGENT,
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
                },
                allow_redirects=False,
                stream=True,
            )
            if resp.status_code not in (301, 302, 303, 307, 308):
                break
            location = (resp.headers.get("Location") or "").strip()
            resp.close()
            if not location:
                return WebDoc(url=url, site=site, ok=False, fetched_at=now,
                              error="站点返回重定向但没有 Location")
            current = urljoin(current, location)
        else:
            return WebDoc(url=url, site=site, ok=False, fetched_at=now,
                          error=f"重定向超过 {MAX_REDIRECTS} 次")
        assert resp is not None
        site = urlparse(current).netloc
    except SecurityError as exc:
        return WebDoc(url=url, site=site, ok=False, fetched_at=now,
                      error=f"URL 安全校验失败：{exc}")
    except Exception as exc:  # noqa: BLE001 —— 网络类异常统一归因
        return WebDoc(url=url, site=site, ok=False, fetched_at=now,
                      error=f"请求失败（{type(exc).__name__}）：{exc}")

    if resp.status_code >= 400:
        resp.close()
        return WebDoc(url=url, site=site, status_code=resp.status_code, ok=False, fetched_at=now,
                      error=f"站点返回 HTTP {resp.status_code}"
                            + ("（可能被反爬拦截，可改用「粘贴文本」导入）" if resp.status_code in (401, 403, 429) else ""))
    try:
        declared = int(resp.headers.get("Content-Length") or 0)
    except ValueError:
        declared = 0
    if declared > MAX_HTML_BYTES:
        resp.close()
        return WebDoc(url=url, site=site, status_code=resp.status_code, ok=False, fetched_at=now,
                      error=f"页面超过 {MAX_HTML_BYTES // 1024 // 1024}MB，已跳过")

    ctype = (resp.headers.get("Content-Type") or "").lower()
    if "pdf" in ctype:
        resp.close()
        return WebDoc(url=url, site=site, status_code=resp.status_code, ok=False, fetched_at=now,
                      error="该地址是 PDF，请先下载到本地再用「上传文件」导入")

    raw = bytearray()
    try:
        for part in resp.iter_content(chunk_size=64 * 1024):
            if not part:
                continue
            raw.extend(part)
            if len(raw) > MAX_HTML_BYTES:
                return WebDoc(url=url, site=site, status_code=resp.status_code, ok=False,
                              fetched_at=now,
                              error=f"页面超过 {MAX_HTML_BYTES // 1024 // 1024}MB，已中止下载")
    finally:
        resp.close()

    html = _decode(resp, bytes(raw))
    title, text = html_to_text(html)
    if len(text) < min_chars:
        return WebDoc(url=url, title=title, site=site, status_code=resp.status_code, ok=False,
                      fetched_at=now, error=f"正文过短（{len(text)} 字），可能是动态渲染或登录墙")
    return WebDoc(url=url, title=title or site, text=text, site=site,
                  status_code=resp.status_code, ok=True, fetched_at=now)


# ─────────────────────────────────────────────
# 搜索
# ─────────────────────────────────────────────
def _ddg_real_url(href: str) -> str:
    """DuckDuckGo 的结果链接是跳转地址，需要解出真实的 uddg 参数。"""
    if "uddg=" in href:
        try:
            return unquote(parse_qs(urlparse(href).query)["uddg"][0])
        except Exception:
            return href
    return href


def _search_duckduckgo(query: str, n: int, timeout: int) -> List[SearchHit]:
    import requests

    resp = requests.get(
        "https://html.duckduckgo.com/html/",
        params={"q": query, "kl": "cn-zh"},
        timeout=timeout,
        headers={"User-Agent": USER_AGENT, "Accept-Language": "zh-CN,zh;q=0.9"},
    )
    resp.raise_for_status()
    html = _decode(resp)
    hits: List[SearchHit] = []
    try:
        from lxml import html as LH

        tree = LH.fromstring(html)
        for a in tree.xpath("//a[contains(@class,'result__a')]"):
            href = _ddg_real_url(a.get("href") or "")
            title = _normalize(a.text_content() or "")
            snippet = ""
            parent = a.getparent()
            for _ in range(4):
                if parent is None:
                    break
                sn = parent.xpath(".//*[contains(@class,'result__snippet')]")
                if sn:
                    snippet = _normalize(sn[0].text_content() or "")
                    break
                parent = parent.getparent()
            if href.startswith("http") and title:
                hits.append(SearchHit(title=title, url=href, snippet=snippet, engine="duckduckgo"))
            if len(hits) >= n:
                break
    except Exception:
        for m in re.finditer(r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', html, re.S):
            hits.append(SearchHit(title=re.sub(r"<[^>]+>", "", m.group(2)).strip(),
                                  url=_ddg_real_url(m.group(1)), engine="duckduckgo"))
            if len(hits) >= n:
                break
    return hits


def _search_bing(query: str, n: int, timeout: int) -> List[SearchHit]:
    import requests

    resp = requests.get(
        "https://www.bing.com/search",
        params={"q": query, "setlang": "zh-CN", "count": max(n, 10)},
        timeout=timeout,
        headers={"User-Agent": USER_AGENT, "Accept-Language": "zh-CN,zh;q=0.9"},
    )
    resp.raise_for_status()
    html = _decode(resp)
    hits: List[SearchHit] = []
    try:
        from lxml import html as LH

        tree = LH.fromstring(html)
        for li in tree.xpath("//li[contains(@class,'b_algo')]"):
            a = li.xpath(".//h2/a")
            if not a:
                continue
            url = a[0].get("href") or ""
            title = _normalize(a[0].text_content() or "")
            sn = li.xpath(".//*[contains(@class,'b_caption')]//p")
            snippet = _normalize(sn[0].text_content() or "") if sn else ""
            if url.startswith("http") and title:
                hits.append(SearchHit(title=title, url=url, snippet=snippet, engine="bing"))
            if len(hits) >= n:
                break
    except Exception:
        pass
    return hits


# 「标题其实是域名/路径」的噪声条目（百度结果里常见），这类条目没有检索价值
_BARE_URL_TITLE_RE = re.compile(r"^[a-z0-9][a-z0-9\-.]*\.[a-z]{2,}(/|$)", re.I)


def _looks_like_bare_url(title: str) -> bool:
    t = (title or "").strip()
    return bool(t) and " " not in t and bool(_BARE_URL_TITLE_RE.match(t))


def _search_360(query: str, n: int, timeout: int) -> List[SearchHit]:
    """360 搜索（so.com）。

    实测中文多词查询里相关性最好的一档，且结果 <a> 上带 `data-mdurl` 直接给出**真实地址**
    （百度/搜狗给的是 link?url= 跳转链，在带代理的网络下常抓不动），因此 360 排在引擎链首位。
    """
    import requests

    resp = requests.get(
        "https://www.so.com/s",
        params={"q": query},
        timeout=timeout,
        headers={"User-Agent": USER_AGENT, "Accept-Language": "zh-CN,zh;q=0.9"},
    )
    resp.raise_for_status()
    html = _decode(resp)
    hits: List[SearchHit] = []
    from lxml import html as LH

    tree = LH.fromstring(html)
    for h3 in tree.xpath("//h3"):
        a = h3.xpath(".//a")
        if not a:
            continue
        title = _normalize(" ".join(h3.itertext()))
        real = (a[0].get("data-mdurl") or "").strip()
        href = (a[0].get("href") or "").strip()
        url = real if real.startswith("http") else urljoin("https://www.so.com", href)
        if not url.startswith("http") or not title or _looks_like_bare_url(title):
            continue
        sn = h3.getparent().xpath(".//p") if h3.getparent() is not None else []
        snippet = _normalize(sn[0].text_content() or "") if sn else ""
        hits.append(SearchHit(title=title, url=url, snippet=snippet, engine="360"))
        if len(hits) >= n:
            break
    return hits


def _search_baidu(query: str, n: int, timeout: int) -> List[SearchHit]:
    """百度搜索。

    中文相关性好，但结果的 href 基本都是 `baidu.com/link?url=...` 跳转链（部分网络下 502），
    所以这里如实保留链接、由 `SearchHit.is_direct` 标注可否直抓；
    广告位（baidu.php）与「标题就是域名」的噪声条目会被剔除。
    """
    import requests

    resp = requests.get(
        "https://www.baidu.com/s",
        params={"wd": query},
        timeout=timeout,
        headers={"User-Agent": USER_AGENT, "Accept-Language": "zh-CN,zh;q=0.9"},
    )
    resp.raise_for_status()
    html = _decode(resp)
    hits: List[SearchHit] = []
    from lxml import html as LH

    tree = LH.fromstring(html)
    for h3 in tree.xpath("//h3"):
        a = h3.xpath(".//a")
        if not a:
            continue
        url = (a[0].get("href") or "").strip()
        title = _normalize(" ".join(h3.itertext()))
        if "baidu.php" in url:            # 广告位
            continue
        if not url.startswith("http") or not title or _looks_like_bare_url(title):
            continue
        snippet = ""
        box = h3.getparent()
        for _ in range(4):                # 摘要常在 h3 的同级或上层容器里
            if box is None:
                break
            got = box.xpath(".//*[contains(@class,'content-right') or contains(@class,'c-abstract') "
                            "or contains(@class,'c-span-last')]")
            if got:
                snippet = _normalize(got[0].text_content() or "")
                break
            box = box.getparent()
        hits.append(SearchHit(title=title, url=url, snippet=snippet, engine="baidu"))
        if len(hits) >= n:
            break
    return hits


def _search_sogou(query: str, n: int, timeout: int) -> List[SearchHit]:
    """搜狗搜索。中文相关性不错，但同为 link?url= 跳转链。"""
    import requests

    resp = requests.get(
        "https://www.sogou.com/web",
        params={"query": query},
        timeout=timeout,
        headers={"User-Agent": USER_AGENT, "Accept-Language": "zh-CN,zh;q=0.9"},
    )
    resp.raise_for_status()
    html = _decode(resp)
    hits: List[SearchHit] = []
    from lxml import html as LH

    tree = LH.fromstring(html)
    for h3 in tree.xpath("//h3"):
        a = h3.xpath(".//a")
        if not a:
            continue
        href = (a[0].get("href") or "").strip()
        title = _normalize(" ".join(h3.itertext()))
        url = href if href.startswith("http") else urljoin("https://www.sogou.com", href)
        if not url.startswith("http") or not title or _looks_like_bare_url(title):
            continue
        hits.append(SearchHit(title=title, url=url, engine="sogou"))
        if len(hits) >= n:
            break
    return hits


# 引擎顺序=实测可用性排序：360（相关 + 真实地址）→ 百度（相关）→ 搜狗 → Bing → DuckDuckGo
# Bing 对中文多词查询会退化成单字匹配（实测「智能座舱 语音交互」返回一堆「智能」词条），
# DuckDuckGo 在部分网络下直接不可达，故两者只作为兜底。
ENGINE_CHAIN = (
    ("360", _search_360),
    ("baidu", _search_baidu),
    ("sogou", _search_sogou),
    ("bing", _search_bing),
    ("duckduckgo", _search_duckduckgo),
)


def web_search(query: str, n: int = 6, timeout: int = DEFAULT_TIMEOUT,
               engines: Optional[Sequence[str]] = None) -> List[SearchHit]:
    """关键词搜索。

    依次尝试多个引擎并把结果**累积去重**（按 URL），而不是「第一个有结果的引擎就收工」——
    单引擎在中文多词查询上会整体退化，多引擎互补能显著提高可用率。全部失败则抛 WebImportError。
    """
    query = (query or "").strip()
    if not query:
        raise WebImportError("搜索关键词为空。")

    errors: List[str] = []
    hits: List[SearchHit] = []
    seen: set[str] = set()
    tokens = _query_tokens(query)
    dropped = 0
    for name, fn in ENGINE_CHAIN:
        if engines and name not in engines:
            continue
        try:
            got = fn(query, n=max(n, 6), timeout=timeout)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{name}：{type(exc).__name__} {exc}")
            got = []
        if not got:
            errors.append(f"{name}：无结果")
        for h in got:
            if h.url in seen or _is_widget_junk(h):
                continue
            if not _relevance_ok(h, tokens):
                dropped += 1
                continue
            seen.add(h.url)
            hits.append(h)
        if len(hits) >= n:
            break
        time.sleep(0.6)          # 引擎之间稍作停顿，避免触发频控

    if not hits:
        raise WebImportError(
            "联网搜索失败（" + "；".join(errors) + "）。"
            "若当前环境无外网，可改用「粘贴文本」或「上传文件」导入。"
        )
    # 可直抓的排前面：用户勾选前几条就能导入成功
    hits.sort(key=lambda h: (not h.is_direct,))
    return hits[:n]


def search_and_fetch(query: str, n: int = 3, timeout: int = DEFAULT_TIMEOUT,
                     min_chars: int = 200) -> Tuple[List[SearchHit], List[WebDoc]]:
    """搜索 + 抓取前 n 篇**可用**的正文。返回 (全部命中, 抓取结果含失败项)。

    注意这里按「成功篇数」计数而不是「尝试篇数」：中文站点大量反爬/动态渲染，
    若尝试 3 篇就收工，很可能一篇都没抓下来。因此会多要点候选，直到凑够 n 篇或候选用尽。
    """
    hits = web_search(query, n=max(n * 4, 10), timeout=timeout)
    docs: List[WebDoc] = []
    ok_count = 0
    tried: set[str] = set()
    for h in hits:
        if ok_count >= n:
            break
        if not h.is_direct or h.url in tried:
            continue
        tried.add(h.url)
        d = fetch_url(h.url, timeout=timeout, min_chars=min_chars)
        d.title = d.title or h.title
        docs.append(d)
        if d.ok:
            ok_count += 1
        time.sleep(0.4)
    return hits, docs


# ─────────────────────────────────────────────
# 转成可入库的文档
# ─────────────────────────────────────────────
def to_documents(webdocs: List[WebDoc]):
    """把抓取成功的网页转成 ingest.Document（保留 URL 作为 source，便于溯源）。"""
    out = []
    for d in webdocs:
        if not d.ok or not d.text.strip():
            continue
        out.append(d.to_document())
    return out


__all__ = [
    "WebDoc", "SearchHit", "WebImportError",
    "fetch_url", "web_search", "search_and_fetch", "html_to_text", "to_documents",
    "is_direct_url", "ENGINE_CHAIN", "USER_AGENT", "DEFAULT_TIMEOUT",
]
