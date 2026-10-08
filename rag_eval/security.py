"""输入与网络边界的安全校验。仅依赖 Python 标准库。"""

from __future__ import annotations

import ipaddress
import re
import socket
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit


class SecurityError(ValueError):
    """输入可能越过本地文件或公网网络边界。"""


_WINDOWS_DEVICES = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


def safe_upload_filename(name: str) -> str:
    """把浏览器提供的名称收敛为单个跨平台安全文件名。"""
    leaf = PurePosixPath((name or "").replace("\\", "/")).name.strip()
    leaf = re.sub(r"[<>:\"/\\|?*\x00-\x1f]", "_", leaf).strip(" .")
    if not leaf:
        leaf = "upload"
    stem = Path(leaf).stem.upper()
    if stem in _WINDOWS_DEVICES:
        leaf = "_" + leaf
    return leaf[:180]


def resolve_upload_path(root: Path, original_name: str) -> Path:
    """返回位于 root 内的上传路径；任何路径逃逸都会被拒绝。"""
    base = root.resolve()
    target = (base / safe_upload_filename(original_name)).resolve()
    try:
        target.relative_to(base)
    except ValueError as exc:
        raise SecurityError("上传文件路径越过了允许目录。") from exc
    return target


def validate_public_http_url(url: str) -> str:
    """验证 URL 仅指向公网 HTTP(S) 地址，并解析全部 A/AAAA 记录。"""
    parsed = urlsplit((url or "").strip())
    if parsed.scheme.lower() not in {"http", "https"}:
        raise SecurityError("仅允许 http:// 或 https:// URL。")
    if parsed.username or parsed.password:
        raise SecurityError("URL 不允许包含用户名或密码。")
    host = (parsed.hostname or "").rstrip(".").lower()
    if not host:
        raise SecurityError("URL 缺少主机名。")
    if host == "localhost" or host.endswith(".localhost"):
        raise SecurityError("不允许访问本机地址。")
    try:
        port = parsed.port or (443 if parsed.scheme.lower() == "https" else 80)
    except ValueError as exc:
        raise SecurityError("URL 端口不合法。") from exc
    try:
        addresses = {item[4][0] for item in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)}
    except OSError as exc:
        raise SecurityError(f"域名解析失败：{host}（{exc}）") from exc
    if not addresses:
        raise SecurityError(f"域名没有可用地址：{host}")
    for raw in addresses:
        try:
            address = ipaddress.ip_address(raw.split("%", 1)[0])
        except ValueError as exc:
            raise SecurityError(f"域名解析结果不是合法 IP：{raw}") from exc
        if not address.is_global:
            raise SecurityError(f"不允许访问非公网地址：{address}")
    return parsed.geturl()


__all__ = [
    "SecurityError", "safe_upload_filename", "resolve_upload_path", "validate_public_http_url",
]
