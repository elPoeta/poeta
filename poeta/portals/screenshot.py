"""Herramienta `screenshot`: pide una captura al portal (org.freedesktop.portal.Screenshot) y devuelve donde quedo."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from poeta.tools.base import Tool, ToolError

from .dbus import PortalBus, PortalError, variant
from .shortcuts import PortalsConfig, ScreenshotSettings

__all__ = ["ScreenshotConfig", "register_screenshot"]
ScreenshotConfig = ScreenshotSettings
SHOT = "org.freedesktop.portal.Screenshot"


def png_size(path: Path) -> tuple[int, int] | None:
    try:
        with path.open("rb") as f:
            head = f.read(24)
        if head[:8] == b"\x89PNG\r\n\x1a\n" and head[12:16] == b"IHDR":
            return struct.unpack(">II", head[16:24])
    except OSError:
        pass
    return None


def uri_to_path(uri: str) -> Path:
    u = urlparse(uri)
    if u.scheme != "file" or u.netloc not in ("", "localhost"):
        raise ToolError("el portal devolvio una ubicacion que no es un archivo local")
    return Path(unquote(u.path))


def register_screenshot(registry: Any, portal: PortalBus, cfg: PortalsConfig) -> list[str]:
    s = cfg.screenshot
    if not (cfg.enabled and s.enabled):
        return []

    async def run(a: dict[str, Any]) -> str:
        interactive = bool(a.get("interactive", False))
        try:
            code, res = await portal.request(SHOT, "Screenshot", "s", [""], {"interactive": _bool(interactive)},
                                             timeout=s.timeout_s * (4 if interactive else 1))
        except PortalError as exc:
            raise ToolError(f"el portal de capturas fallo: {exc}") from exc
        if code == 1:
            raise ToolError("la captura fue cancelada")
        if code != 0 or not res.get("uri"):
            raise ToolError("el portal no pudo tomar la captura")
        path = uri_to_path(str(res["uri"]))
        if not path.is_file():
            raise ToolError("el portal dijo que guardo la captura pero el archivo no existe")
        dims = png_size(path)
        size = f" ({dims[0]}x{dims[1]} px)" if dims else ""
        return f"Captura guardada en {path}{size}. Solo se guardo en este equipo; no se envio a ningun lado."

    registry.register(Tool(
        name="screenshot",
        description=("Toma una captura de pantalla con el portal del escritorio y la guarda como archivo en el equipo. "
                     "Devuelve la ruta del archivo (no el contenido de la imagen). Con interactive=true el usuario "
                     "elige el area."),
        parameters={"type": "object", "properties": {"interactive": {"type": "boolean"}}, "required": []},
        run=run, action="screen_capture", timeout_s=s.timeout_s * 4 + 5, source="portal",
        describe=lambda a: ("Tomar una captura de " + ("un area que elegis" if a.get("interactive") else "toda la pantalla")
                            + " (se guarda en tu equipo; no se envia a ningun lado)", {"interactive": bool(a.get("interactive"))}),
    ))
    return ["screenshot"]


def _bool(v: bool) -> Any:
    return variant("b", v)
