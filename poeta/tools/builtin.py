"""Herramientas integradas. Cada una declara la accion de permisos que le corresponde."""

from __future__ import annotations

import asyncio
import configparser
import os
import tempfile
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable
from urllib.parse import urlparse

from .base import Tool, ToolError, ToolRegistry
from .webfetch import fetch_text

DAYS = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
MONTHS = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre",
          "octubre", "noviembre", "diciembre"]


class ToolsConfigError(ValueError):
    pass


@dataclass(frozen=True)
class ToolsConfig:
    enabled: bool = True
    max_tool_rounds: int = 6
    timeout_s: float = 30.0
    max_output_chars: int = 8000
    builtin: tuple[str, ...] = ()
    read_file_max_bytes: int = 200_000
    write_file_max_chars: int = 1_000_000
    fetch_max_bytes: int = 1_000_000
    fetch_block_private: bool = True
    user_agent: str = "POETA/0.1 (asistente local)"

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "ToolsConfig":
        t = cfg.get("tools") or {}
        d = cls()

        def num(key: str, default: Any, kind: type, minimum: float = 1) -> Any:
            v = t.get(key, default)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or v < minimum:
                raise ToolsConfigError(f"tools.{key} debe ser un numero >= {minimum}")
            return kind(v)

        builtin = tuple(name for name, on in (t.get("builtin") or {}).items() if on is True)
        unknown = set(builtin) - set(BUILTIN_NAMES)
        if unknown:
            raise ToolsConfigError(f"herramientas integradas desconocidas: {sorted(unknown)}")
        fetch = t.get("fetch") or {}
        return cls(
            enabled=t.get("enabled", True) is True,
            max_tool_rounds=num("max_tool_rounds", d.max_tool_rounds, int, 1),
            timeout_s=num("timeout_s", d.timeout_s, float, 1),
            max_output_chars=num("max_output_chars", d.max_output_chars, int, 200),
            builtin=builtin,
            read_file_max_bytes=num("read_file_max_bytes", d.read_file_max_bytes, int, 1000),
            write_file_max_chars=num("write_file_max_chars", d.write_file_max_chars, int, 1),
            fetch_max_bytes=int(fetch.get("max_bytes", d.fetch_max_bytes)),
            fetch_block_private=fetch.get("block_private_networks", True) is not False,
            user_agent=str(fetch.get("user_agent") or d.user_agent),
        )


BUILTIN_NAMES = ("get_datetime", "read_file", "list_dir", "write_file", "open_app", "fetch_url", "run_command")


def _fold(s: str) -> str:
    n = unicodedata.normalize("NFD", s.casefold())
    return "".join(c for c in n if unicodedata.category(c) != "Mn")


# -- fecha y hora ----------------------------------------------------------------
async def _datetime(_: dict[str, Any]) -> str:
    now = datetime.now().astimezone()
    return (f"{DAYS[now.weekday()]} {now.day} de {MONTHS[now.month - 1]} de {now.year}, "
            f"{now:%H:%M}, zona horaria {now.tzname()} (UTC{now:%z})")


# -- archivos ----------------------------------------------------------------------
def make_path_resolver(workspace: Path) -> Callable[[str], Path]:
    def resolve(raw: str) -> Path:
        if not isinstance(raw, str) or not raw.strip() or "\x00" in raw:
            raise ToolError("ruta invalida")
        p = Path(raw).expanduser()
        if not p.is_absolute():
            p = workspace / p
        return p.resolve()  # misma resolucion que usa el PermissionManager (sigue enlaces simbolicos)

    return resolve


def _read_file_tool(resolve: Callable[[str], Path], cfg: ToolsConfig) -> Tool:
    async def run(a: dict[str, Any]) -> str:
        p = resolve(a["path"])
        if not p.is_file():
            raise ToolError(f"no es un archivo: {p}")
        with p.open("rb") as f:
            raw = f.read(cfg.read_file_max_bytes + 1)
        if b"\x00" in raw[:2048]:
            raise ToolError("el archivo parece binario; solo se leen archivos de texto")
        more = len(raw) > cfg.read_file_max_bytes
        text = raw[: cfg.read_file_max_bytes].decode("utf-8", errors="replace")
        return text + ("\n[... archivo recortado]" if more else "")

    return Tool(
        "read_file", "Lee un archivo de texto del espacio de trabajo. La ruta puede ser relativa a la carpeta 'workspace'.",
        {"type": "object", "properties": {"path": {"type": "string", "description": "Ruta del archivo"}},
         "required": ["path"], "additionalProperties": False},
        run, "read_authorized_file", untrusted=True,
        describe=lambda a: (f"Leer el archivo {resolve(a['path'])}", {"path": str(resolve(a["path"]))}))


def _list_dir_tool(resolve: Callable[[str], Path]) -> Tool:
    async def run(a: dict[str, Any]) -> str:
        p = resolve(a.get("path", "."))
        if not p.is_dir():
            raise ToolError(f"no es una carpeta: {p}")
        rows = []
        for e in sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))[:200]:
            rows.append(f"{e.name}/" if e.is_dir() else f"{e.name}  ({e.stat().st_size} bytes)")
        return "\n".join(rows) or "(carpeta vacia)"

    return Tool(
        "list_dir", "Lista los archivos de una carpeta del espacio de trabajo (por defecto, 'workspace').",
        {"type": "object", "properties": {"path": {"type": "string"}}, "additionalProperties": False},
        run, "read_authorized_file", untrusted=True,
        describe=lambda a: (f"Listar la carpeta {resolve(a.get('path', '.'))}", {"path": str(resolve(a.get("path", ".")))}))


def _write_file_tool(resolve: Callable[[str], Path], cfg: ToolsConfig) -> Tool:
    async def run(a: dict[str, Any]) -> str:
        p = resolve(a["path"])
        content = a["content"]
        if len(content) > cfg.write_file_max_chars:
            raise ToolError("el contenido supera el maximo permitido")
        if p.exists() and not a.get("overwrite", False):
            raise ToolError(f"{p} ya existe; para reemplazarlo usa overwrite=true")
        p.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=p.parent, prefix=".poeta-")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(content)
            os.replace(tmp, p)  # escritura atomica
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        return f"Archivo guardado: {p} ({len(content)} caracteres)"

    def action(a: dict[str, Any]) -> str:
        return "modify_file" if resolve(a["path"]).exists() else "create_document"

    def describe(a: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        p = resolve(a["path"])
        if p.exists() and not a.get("overwrite", False):  # antes de pedir permiso: no tiene sentido confirmar esto
            raise ToolError(f"{p} ya existe; para reemplazarlo usa overwrite=true")
        verb = "Reemplazar" if p.exists() else "Crear"
        return f"{verb} el archivo {p} ({len(a['content'])} caracteres)", {"path": str(p)}

    return Tool(
        "write_file", "Crea un archivo de texto en el espacio de trabajo. No reemplaza archivos existentes salvo overwrite=true.",
        {"type": "object", "properties": {
            "path": {"type": "string"}, "content": {"type": "string"}, "overwrite": {"type": "boolean"}},
         "required": ["path", "content"], "additionalProperties": False},
        run, action, describe=describe)


# -- aplicaciones --------------------------------------------------------------------
def desktop_dirs() -> list[Path]:
    home = Path.home()
    dirs = [home / ".local/share/applications", home / ".local/share/flatpak/exports/share/applications",
            Path("/var/lib/flatpak/exports/share/applications")]
    for d in os.environ.get("XDG_DATA_DIRS", "/usr/local/share:/usr/share").split(":"):
        if d:
            dirs.append(Path(d) / "applications")
    return [d for i, d in enumerate(dirs) if d not in dirs[:i]]


def find_desktop_entries(name: str, dirs: list[Path]) -> list[tuple[str, str, Path]]:
    """[(id, nombre visible, ruta)] de las aplicaciones que coinciden; las exactas primero."""
    want = _fold(name.strip())
    exact: list[tuple[str, str, Path]] = []
    partial: list[tuple[str, str, Path]] = []
    seen: set[str] = set()
    for d in dirs:
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*.desktop")):
            if f.name in seen:
                continue
            cp = configparser.ConfigParser(interpolation=None, strict=False)
            try:
                cp.read(f, encoding="utf-8")
                e = cp["Desktop Entry"]
            except (configparser.Error, KeyError, UnicodeDecodeError, OSError):
                continue
            if e.get("Type") != "Application" or e.get("NoDisplay", "false").lower() == "true" \
                    or e.get("Hidden", "false").lower() == "true":
                continue
            seen.add(f.name)
            names = [e.get("Name", ""), e.get("Name[es]", ""), f.stem.rsplit(".", 1)[-1]]
            folded = [_fold(n) for n in names if n]
            shown = e.get("Name[es]") or e.get("Name") or f.stem
            if want in folded or _fold(f.stem) == want:
                exact.append((f.stem, shown, f))
            elif want and any(want in n for n in folded):
                partial.append((f.stem, shown, f))
    return exact or partial


async def _spawn_detached(argv: list[str]) -> None:
    proc = await asyncio.create_subprocess_exec(
        *argv, stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL, start_new_session=True)
    try:
        await asyncio.wait_for(proc.wait(), 3)  # `gio launch` vuelve enseguida; la app sigue por su cuenta
    except asyncio.TimeoutError:
        pass
    if proc.returncode not in (None, 0):
        raise ToolError(f"no se pudo abrir la aplicacion (codigo {proc.returncode})")


def _open_app_tool(dirs_fn: Callable[[], list[Path]], launcher: Callable[[list[str]], Awaitable[None]]) -> Tool:
    def pick(a: dict[str, Any]) -> tuple[str, str, Path]:
        found = find_desktop_entries(a["name"], dirs_fn())
        if not found:
            raise ToolError(f"no encontre ninguna aplicacion llamada '{a['name']}'")
        if len(found) > 1:
            opts = ", ".join(sorted({n for _, n, _ in found})[:6])
            raise ToolError(f"'{a['name']}' es ambiguo; coincide con: {opts}. Pide el nombre exacto")
        return found[0]

    async def run(a: dict[str, Any]) -> str:
        _id, shown, path = pick(a)
        await launcher(["gio", "launch", str(path)])
        return f"Abri {shown}."

    def describe(a: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        _id, shown, path = pick(a)
        return f"Abrir la aplicacion {shown}", {"app": shown, "desktop_file": str(path)}

    return Tool(
        "open_app", "Abre una aplicacion instalada por su nombre (por ejemplo 'Firefox' o 'Konsole'). No acepta argumentos.",
        {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"], "additionalProperties": False},
        run, "open_app", describe=describe)


# -- web ---------------------------------------------------------------------------------
def _fetch_url_tool(cfg: ToolsConfig) -> Tool:
    async def run(a: dict[str, Any]) -> str:
        return await fetch_text(a["url"], cfg.fetch_max_bytes, cfg.user_agent, cfg.fetch_block_private)

    def describe(a: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        host = urlparse(a["url"]).hostname or "?"
        return f"Descargar la pagina {a['url']}", {"url": a["url"], "host": host}

    return Tool(
        "fetch_url", "Descarga una pagina web (http/https) y devuelve su texto. No accede a redes locales.",
        {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"], "additionalProperties": False},
        run, "browse_web", untrusted=True, describe=describe)


# -- comandos ----------------------------------------------------------------------------------
def _run_command_tool(workspace: Path, cfg: ToolsConfig) -> Tool:
    async def run(a: dict[str, Any]) -> str:
        argv = a["argv"]
        workspace.mkdir(parents=True, exist_ok=True)
        env = {k: os.environ[k] for k in ("PATH", "HOME", "LANG", "LC_ALL") if k in os.environ}
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv, cwd=workspace, env=env, stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        except (FileNotFoundError, PermissionError) as exc:
            raise ToolError(f"no se pudo ejecutar '{argv[0]}': {exc.strerror}") from exc
        try:
            out, _ = await proc.communicate()
        except asyncio.CancelledError:
            proc.kill()
            raise
        text = out.decode("utf-8", errors="replace")
        return f"[codigo de salida {proc.returncode}]\n{text}"

    return Tool(
        "run_command", "Ejecuta un comando permitido (lista argv, sin shell) en la carpeta 'workspace'. Solo funcionan "
                      "los comandos de la lista permitida del usuario.",
        {"type": "object", "properties": {"argv": {"type": "array", "items": {"type": "string"}}},
         "required": ["argv"], "additionalProperties": False},
        run, "run_command", untrusted=True,
        describe=lambda a: ("Ejecutar el comando: " + " ".join(map(str, a["argv"])), {"argv": a["argv"]}))


def register_builtin(registry: ToolRegistry, cfg: ToolsConfig, root: Path,
                     dirs_fn: Callable[[], list[Path]] = desktop_dirs,
                     launcher: Callable[[list[str]], Awaitable[None]] = _spawn_detached) -> list[str]:
    workspace = root / "workspace"
    resolve = make_path_resolver(workspace)
    factories: dict[str, Callable[[], Tool]] = {
        "get_datetime": lambda: Tool(
            "get_datetime", "Devuelve la fecha y la hora actuales del usuario.",
            {"type": "object", "properties": {}, "additionalProperties": False}, _datetime, "system_info"),
        "read_file": lambda: _read_file_tool(resolve, cfg),
        "list_dir": lambda: _list_dir_tool(resolve),
        "write_file": lambda: _write_file_tool(resolve, cfg),
        "open_app": lambda: _open_app_tool(dirs_fn, launcher),
        "fetch_url": lambda: _fetch_url_tool(cfg),
        "run_command": lambda: _run_command_tool(workspace, cfg),
    }
    added = []
    for name in cfg.builtin:
        registry.register(factories[name]())
        added.append(name)
    return added
