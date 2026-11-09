"""Cliente MCP: conexion a servidores autorizados y adaptacion de sus herramientas.

Seguridad (secc. 10.3 del plan):
- Doble llave: un servidor solo se conecta si `enabled: true` Y su nombre esta en `authorized_servers`.
- Verificacion individual: cada servidor debe aprobarse (`python main.py mcp approve NOMBRE`) tras ver sus
  herramientas; se fija una huella de nombres, descripciones y esquemas. Si cambian (por ejemplo tras una
  actualizacion del servidor), sus herramientas NO se registran hasta aprobar de nuevo.
- Cada herramienta MCP pasa por el Permission Manager con el nivel de riesgo declarado en mcp.yaml
  (por defecto `write`: pide confirmacion). Sus resultados son datos no confiables.
- Los servidores stdio solo reciben las variables de entorno seguras por defecto del SDK mas las que se
  listen en `env`: nunca las claves de POETA.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from poeta.tools import Tool, ToolError, ToolRegistry

log = logging.getLogger("poeta.mcp")

RISK_ACTIONS = {"read": "mcp_read", "browse": "mcp_browse", "write": "mcp_write", "sensitive": "mcp_sensitive"}
TRANSPORTS = ("stdio", "streamable_http")
_SERVER_OK = re.compile(r"^[a-z0-9][a-z0-9-]{0,30}$")
MAX_TOOLS_PER_SERVER = 200


class McpConfigError(ValueError):
    pass


@dataclass(frozen=True)
class ServerConfig:
    name: str
    enabled: bool
    transport: str
    command: str | None = None
    args: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    url: str | None = None
    risk: str = "write"
    tool_risk: dict[str, str] = field(default_factory=dict)
    untrusted: bool = True
    timeout_s: float = 60.0
    connect_timeout_s: float = 30.0


@dataclass(frozen=True)
class McpConfig:
    authorized: tuple[str, ...]
    servers: dict[str, ServerConfig]

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "McpConfig":
        m = cfg.get("mcp") or {}
        authorized = m.get("authorized_servers") or []
        if not isinstance(authorized, list) or not all(isinstance(a, str) for a in authorized):
            raise McpConfigError("mcp.authorized_servers debe ser una lista de nombres")
        servers: dict[str, ServerConfig] = {}
        for name, s in (m.get("servers") or {}).items():
            if not isinstance(name, str) or not _SERVER_OK.match(name):
                raise McpConfigError(f"nombre de servidor invalido: {name!r} (minusculas, numeros y guiones)")
            s = s or {}
            transport = s.get("transport", "stdio")
            if transport not in TRANSPORTS:
                raise McpConfigError(f"servers.{name}.transport debe ser uno de {TRANSPORTS}")
            risk = s.get("risk", "write")
            tool_risk = dict(s.get("tool_risk") or {})
            for r in [risk, *tool_risk.values()]:
                if r not in RISK_ACTIONS:
                    raise McpConfigError(f"servers.{name}: riesgo invalido {r!r} (usar {list(RISK_ACTIONS)})")
            args = s.get("args") or []
            env = s.get("env") or {}
            if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
                raise McpConfigError(f"servers.{name}.args debe ser una lista de textos")
            if not isinstance(env, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in env.items()):
                raise McpConfigError(f"servers.{name}.env debe ser un mapa de textos")
            enabled = s.get("enabled", False) is True
            command, url = s.get("command"), s.get("url")
            if enabled and transport == "stdio" and not (isinstance(command, str) and command.strip()):
                raise McpConfigError(f"servers.{name}: falta 'command' (el servidor esta habilitado)")
            if enabled and transport == "streamable_http":
                if not (isinstance(url, str) and (url.startswith("https://") or url.startswith("http://localhost")
                                                  or url.startswith("http://127.0.0.1"))):
                    raise McpConfigError(f"servers.{name}.url debe ser https:// (o http://localhost)")
            for key in ("timeout_s", "connect_timeout_s"):
                v = s.get(key, 60 if key == "timeout_s" else 30)
                if isinstance(v, bool) or not isinstance(v, (int, float)) or v <= 0:
                    raise McpConfigError(f"servers.{name}.{key} debe ser un numero positivo")
            servers[name] = ServerConfig(
                name, enabled, transport, command, tuple(args), env, url, risk, tool_risk,
                s.get("untrusted", True) is not False, float(s.get("timeout_s", 60)),
                float(s.get("connect_timeout_s", 30)))
        unknown = set(authorized) - set(servers)
        if unknown:
            raise McpConfigError(f"authorized_servers menciona servidores que no existen: {sorted(unknown)}")
        return cls(tuple(authorized), servers)

    def eligible(self, name: str) -> bool:
        s = self.servers[name]
        return s.enabled and name in self.authorized


@dataclass
class ServerStatus:
    name: str
    state: str  # conectado | omitido | error | deshabilitado
    detail: str = ""
    tools: list[str] = field(default_factory=list)
    fingerprint: str = ""


# -- utilidades ----------------------------------------------------------------------------------
def _clean(text: str, limit: int) -> str:
    text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", " ", text or "")
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def fingerprint(tools: list[Any]) -> str:
    items = sorted((t.name, t.description or "", _schema(t)) for t in tools)
    return hashlib.sha256(json.dumps(items, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def _schema(tool: Any) -> dict[str, Any]:
    s = getattr(tool, "input_schema", None) or getattr(tool, "inputSchema", None) or {}
    if not isinstance(s, dict) or s.get("type") != "object":
        return {"type": "object", "properties": {}}
    return s


def tool_name(server: str, tool: str) -> str:
    base = f"mcp__{server}__{re.sub(r'[^A-Za-z0-9_-]', '_', tool)}"
    if len(base) <= 64:
        return base
    return base[:57] + "_" + hashlib.sha1(tool.encode()).hexdigest()[:6]


def result_to_text(res: Any) -> str:
    parts: list[str] = []
    for block in getattr(res, "content", None) or []:
        kind = getattr(block, "type", "")
        if kind == "text":
            parts.append(getattr(block, "text", ""))
        elif kind in ("image", "audio"):
            parts.append(f"[{kind} omitido: POETA aun no procesa contenido binario de herramientas]")
        elif kind == "resource":
            r = getattr(block, "resource", None)
            parts.append(getattr(r, "text", None) or "[recurso binario omitido]")
        else:
            parts.append(f"[contenido {kind or 'desconocido'} omitido]")
    text = "\n".join(p for p in parts if p)
    if not text:
        sc = getattr(res, "structured_content", None) or getattr(res, "structuredContent", None)
        if sc:
            text = json.dumps(sc, ensure_ascii=False)
    return text or "(sin contenido)"


def default_client_factory(scfg: ServerConfig) -> Any:
    """Contexto asincrono que entrega un Client del SDK (mcp>=2)."""
    from mcp import Client, StdioServerParameters

    if scfg.transport == "stdio":
        env = {k: v for k, v in scfg.env.items()}
        return Client(StdioServerParameters(command=scfg.command, args=list(scfg.args), env=env or None),
                      read_timeout_seconds=scfg.timeout_s)
    return Client(scfg.url, read_timeout_seconds=scfg.timeout_s)


class _Runner:
    """Mantiene abierto el cliente de un servidor dentro de SU PROPIA tarea (requisito de anyio)."""

    def __init__(self, factory: Callable[[], Any]):
        self._factory = factory
        self.client: Any = None
        self.error: BaseException | None = None
        self._ready, self._stop = asyncio.Event(), asyncio.Event()
        self._task: asyncio.Task | None = None

    async def start(self, timeout: float) -> Any:
        self._task = asyncio.create_task(self._run())
        try:
            await asyncio.wait_for(self._ready.wait(), timeout)
        except asyncio.TimeoutError:
            await self.close()
            raise TimeoutError(f"sin respuesta tras {timeout:.0f} s") from None
        if self.error is not None:
            err = self.error
            await self.close()
            raise err
        return self.client

    async def _run(self) -> None:
        try:
            async with self._factory() as client:
                self.client = client
                self._ready.set()
                await self._stop.wait()
        except BaseException as exc:  # noqa: BLE001
            if not self._ready.is_set():
                self.error = exc
                self._ready.set()
            elif not isinstance(exc, asyncio.CancelledError):
                log.warning("servidor MCP cerrado: %s", exc)
        finally:
            self.client = None

    async def close(self) -> None:
        self._stop.set()
        if self._task:
            try:
                await asyncio.wait_for(asyncio.shield(self._task), 5)
            except (asyncio.TimeoutError, asyncio.CancelledError, Exception):  # noqa: BLE001
                self._task.cancel()
                with contextlib.suppress(BaseException):
                    await self._task


class McpManager:
    def __init__(self, cfg: McpConfig, registry: ToolRegistry, root: Path, audit: Any = None,
                 client_factory: Callable[[ServerConfig], Any] = default_client_factory):
        self.cfg, self.registry, self.audit, self._factory = cfg, registry, audit, client_factory
        self.pins_path = Path(root) / "data" / "mcp_pins.json"
        self._runners: dict[str, _Runner] = {}
        self._status: dict[str, ServerStatus] = {}

    # -- pins ----------------------------------------------------------------------------------
    def _pins(self) -> dict[str, Any]:
        try:
            return json.loads(self.pins_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def approve(self, name: str, fp: str, tool_names: list[str]) -> None:
        pins = self._pins()
        pins[name] = {"fingerprint": fp, "tools": sorted(tool_names)}
        self.pins_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.pins_path.with_suffix(".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(pins, f, indent=2, ensure_ascii=False)
        os.replace(tmp, self.pins_path)
        self._audit("mcp_server_approved", server=name, fingerprint=fp, tools=sorted(tool_names))

    def _audit(self, event: str, **fields: Any) -> None:
        if self.audit is not None:
            try:
                self.audit.record(event, **fields)
            except Exception:  # noqa: BLE001
                log.exception("no se pudo auditar %s", event)

    # -- ciclo de vida --------------------------------------------------------------------------
    async def _open(self, name: str) -> tuple[Any, list[Any]]:
        scfg = self.cfg.servers[name]
        runner = _Runner(lambda: self._factory(scfg))
        client = await runner.start(scfg.connect_timeout_s)
        try:
            tools: list[Any] = []
            cursor = None
            while True:
                page = await asyncio.wait_for(client.list_tools(cursor=cursor) if cursor else client.list_tools(),
                                              scfg.connect_timeout_s)
                tools.extend(page.tools)
                cursor = getattr(page, "next_cursor", None) or getattr(page, "nextCursor", None)
                if not cursor or len(tools) >= MAX_TOOLS_PER_SERVER:
                    break
        except BaseException:
            await runner.close()
            raise
        self._runners[name] = runner
        return client, tools[:MAX_TOOLS_PER_SERVER]

    async def inspect(self, name: str) -> tuple[list[Any], str]:
        """Conecta, lista las herramientas y se desconecta. No registra nada. Para `mcp list/approve`."""
        if name not in self.cfg.servers:
            raise KeyError(name)
        _, tools = await self._open(name)
        await self._close_one(name)
        return tools, fingerprint(tools)

    async def start(self) -> list[ServerStatus]:
        for name, scfg in self.cfg.servers.items():
            if not scfg.enabled:
                self._status[name] = ServerStatus(name, "deshabilitado")
                continue
            if not self.cfg.eligible(name):
                self._status[name] = ServerStatus(name, "omitido", "habilitado pero no esta en authorized_servers")
                self._audit("mcp_server_skipped", server=name, reason="no autorizado")
                continue
            self._status[name] = await self._start_one(name)
        return list(self._status.values())

    async def _start_one(self, name: str) -> ServerStatus:
        scfg = self.cfg.servers[name]
        try:
            client, tools = await self._open(name)
        except BaseException as exc:  # noqa: BLE001
            if isinstance(exc, asyncio.CancelledError):
                raise
            msg = f"{type(exc).__name__}: {exc}"[:200]
            self._audit("mcp_server_error", server=name, error=msg)
            return ServerStatus(name, "error", msg)
        fp = fingerprint(tools)
        pin = self._pins().get(name)
        if not pin or pin.get("fingerprint") != fp:
            await self._close_one(name)
            why = "sin aprobar" if not pin else "las herramientas cambiaron desde la aprobacion"
            self._audit("mcp_server_skipped", server=name, reason=why, fingerprint=fp)
            return ServerStatus(name, "omitido",
                                f"{why}: revisa con `python main.py mcp approve {name}`", fingerprint=fp)
        registered = []
        for t in tools:
            tname = tool_name(name, t.name)
            risk = scfg.tool_risk.get(t.name, scfg.risk)
            tool = Tool(
                name=tname,
                description=f"[MCP {name}] " + _clean(t.description or t.name, 400),
                parameters=_schema(t),
                run=self._make_runner(name, t.name),
                action=RISK_ACTIONS[risk],
                untrusted=scfg.untrusted,
                describe=self._make_describer(name, t.name),
                source=f"mcp:{name}",
                timeout_s=scfg.timeout_s + 5,
            )
            try:
                self.registry.register(tool)
            except ValueError:
                log.warning("herramienta MCP duplicada, se omite: %s", tname)
                continue
            registered.append(tname)
        self._audit("mcp_server_connected", server=name, tools=registered, fingerprint=fp)
        return ServerStatus(name, "conectado", tools=registered, fingerprint=fp)

    def _make_runner(self, server: str, tool: str) -> Callable[[dict[str, Any]], Any]:
        async def run(args: dict[str, Any]) -> str:
            runner = self._runners.get(server)
            if runner is None or runner.client is None:
                raise ToolError(f"el servidor MCP '{server}' no esta conectado")
            try:
                res = await runner.client.call_tool(tool, args)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                raise ToolError(f"el servidor MCP '{server}' fallo: {type(exc).__name__}") from exc
            text = result_to_text(res)
            if getattr(res, "is_error", None) or getattr(res, "isError", None):
                raise ToolError(text)
            return text

        return run

    @staticmethod
    def _make_describer(server: str, tool: str) -> Callable[[dict[str, Any]], tuple[str, dict[str, Any]]]:
        return lambda args: (f"Usar la herramienta '{tool}' del servidor MCP '{server}'",
                             {"server": server, "mcp_tool": tool})

    async def _close_one(self, name: str) -> None:
        runner = self._runners.pop(name, None)
        if runner:
            await runner.close()

    async def stop(self) -> None:
        for name in list(self._runners):
            self.registry.unregister_source(f"mcp:{name}")
            await self._close_one(name)

    def status(self) -> list[ServerStatus]:
        return list(self._status.values())
