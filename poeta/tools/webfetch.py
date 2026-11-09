"""Descarga de paginas web con defensas basicas (SSRF, tamano, tipo de contenido)."""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from html.parser import HTMLParser
from urllib.parse import urlparse

import httpx

from .base import ToolError

_SKIP = {"script", "style", "noscript", "svg", "head", "template"}
_BLOCK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "section", "article", "ul", "ol", "table"}


class _Text(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in _SKIP:
            self._skip += 1
        elif tag in _BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in _SKIP and self._skip:
            self._skip -= 1
        elif tag in _BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)


def html_to_text(html: str) -> str:
    p = _Text()
    p.feed(html)
    p.close()
    lines = [" ".join(line.split()) for line in "".join(p.parts).splitlines()]
    out, blank = [], False
    for line in lines:
        if line:
            out.append(line)
            blank = False
        elif not blank and out:
            out.append("")
            blank = True
    return "\n".join(out).strip()


def is_private_ip(ip: str) -> bool:
    a = ipaddress.ip_address(ip)
    return (a.is_private or a.is_loopback or a.is_link_local or a.is_multicast
            or a.is_reserved or a.is_unspecified)


async def assert_public_url(url: str) -> None:
    u = urlparse(url)
    if u.scheme not in ("http", "https") or not u.hostname:
        raise ToolError("solo se permiten URLs http(s) completas")
    host = u.hostname
    try:
        infos = await asyncio.to_thread(socket.getaddrinfo, host, u.port or (443 if u.scheme == "https" else 80))
    except socket.gaierror as exc:
        raise ToolError(f"no se pudo resolver {host}") from exc
    for info in infos:
        if is_private_ip(info[4][0]):
            raise ToolError("la direccion apunta a una red privada o local; no se accede (proteccion SSRF)")


async def fetch_text(url: str, max_bytes: int, user_agent: str, block_private: bool = True,
                     timeout: float = 20.0, max_redirects: int = 4) -> str:
    current = url
    async with httpx.AsyncClient(follow_redirects=False, timeout=timeout, headers={"User-Agent": user_agent}) as client:
        for _ in range(max_redirects + 1):
            if block_private:
                await assert_public_url(current)  # tambien en cada redireccion
            async with client.stream("GET", current) as resp:
                if resp.is_redirect:
                    loc = resp.headers.get("location")
                    if not loc:
                        raise ToolError("redireccion sin destino")
                    current = str(httpx.URL(current).join(loc))
                    continue
                if resp.status_code >= 400:
                    raise ToolError(f"el servidor respondio HTTP {resp.status_code}")
                ctype = resp.headers.get("content-type", "").split(";")[0].strip().lower()
                if ctype and not (ctype.startswith("text/") or ctype in ("application/json", "application/xml",
                                                                       "application/xhtml+xml")):
                    raise ToolError(f"tipo de contenido no soportado: {ctype}")
                chunks, total = [], 0
                async for chunk in resp.aiter_bytes():
                    total += len(chunk)
                    chunks.append(chunk)
                    if total >= max_bytes:
                        break
                raw = b"".join(chunks)[:max_bytes]
                text = raw.decode(resp.encoding or "utf-8", errors="replace")
                return html_to_text(text) if ctype in ("text/html", "application/xhtml+xml", "") else text
        raise ToolError("demasiadas redirecciones")
