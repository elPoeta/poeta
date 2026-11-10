"""Herramientas de escritorio (Fedora/KDE/Wayland): notificaciones, multimedia, volumen, ventanas, servicios, procesos.

Cada una usa una utilidad externa estandar (sin shell, con argv fijo) y declara su accion de permisos. Solo se registran
las que tienen su utilidad instalada. Todo texto que viene del escritorio (titulos de ventana, metadatos, nombres de
procesos) lo controlan terceros: esas herramientas son `untrusted`.
"""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import signal
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from poeta.tools import Tool, ToolError, ToolRegistry

from .runner import CmdResult, Runner, run_cmd

DESKTOP_TOOLS = ("notify", "media_status", "media_control", "get_volume", "set_volume", "list_windows",
                 "window_action", "service_status", "service_control", "list_processes", "stop_process")
# herramienta -> utilidad externa requerida
REQUIRES = {"notify": "notify-send", "media_status": "playerctl", "media_control": "playerctl", "get_volume": "wpctl",
            "set_volume": "wpctl", "list_windows": "kdotool", "window_action": "kdotool",
            "service_status": "systemctl", "service_control": "systemctl", "list_processes": "ps",
            "stop_process": None}
_UNIT = re.compile(r"^[A-Za-z0-9@_.:-]{1,100}\.(service|timer|socket)$")
_WINID = re.compile(r"^\{?[0-9A-Fa-f-]{8,40}\}?$")
BROWSERS = ("firefox", "chromium", "chrome", "brave", "vivaldi", "opera", "edge", "zen", "librewolf", "waterfox")
_PLAYER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,59}$")
CRITICAL = {"systemd", "init", "dbus-daemon", "dbus-broker", "dbus-broker-lau", "pipewire", "wireplumber", "kwin_wayland",
            "sddm", "sddm-helper", "login", "plasmashell", "polkitd", "NetworkManager"}


class DesktopConfigError(ValueError):
    pass


@dataclass(frozen=True)
class DesktopConfig:
    enabled: bool = True
    timeout_s: float = 10.0
    tools: tuple[str, ...] = ()
    max_volume_percent: int = 100
    service_allowlist: tuple[str, ...] = ()
    max_listed: int = 25
    media_wait_s: float = 6.0

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "DesktopConfig":
        d = cfg.get("desktop") or {}
        on = tuple(n for n, v in (d.get("tools") or {}).items() if v is True)
        unknown = set(on) - set(DESKTOP_TOOLS)
        if unknown:
            raise DesktopConfigError(f"herramientas de escritorio desconocidas: {sorted(unknown)}")

        def num(sec: dict, key: str, default: Any, lo: float, hi: float, kind: type) -> Any:
            v = sec.get(key, default)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not lo <= v <= hi:
                raise DesktopConfigError(f"desktop.{key} debe ser un numero entre {lo} y {hi}")
            return kind(v)

        allow = (d.get("services") or {}).get("allowlist") or []
        if not isinstance(allow, list) or not all(isinstance(u, str) and _UNIT.match(u) for u in allow):
            raise DesktopConfigError("desktop.services.allowlist debe ser una lista de unidades (nombre.service)")
        return cls(d.get("enabled", True) is True, num(d, "timeout_s", 10, 1, 120, float), on,
                   num(d.get("audio") or {}, "max_volume_percent", 100, 1, 150, int), tuple(allow),
                   num(d.get("processes") or {}, "max_listed", 25, 1, 100, int),
                   num(d.get("media") or {}, "wait_s", 6, 0, 60, float))


def _clean(text: str, limit: int = 200) -> str:
    text = " ".join(re.sub(r"[\x00-\x1f\x7f]", " ", text or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _obj(props: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {"type": "object", "properties": props, "required": required or [], "additionalProperties": False}


def _fail(res: CmdResult, what: str) -> ToolError:
    return ToolError(f"{what} fallo: {_clean(res.err or res.out, 160) or 'codigo ' + str(res.rc)}")


def register_desktop(registry: ToolRegistry, cfg: DesktopConfig, runner: Runner = run_cmd,
                     which: Callable[[str], str | None] = shutil.which,
                     killer: Callable[[int, int], None] = os.kill, proc_dir: Path = Path("/proc")) -> tuple[list[str], dict[str, str]]:
    """Devuelve (registradas, omitidas {herramienta: motivo})."""
    t = cfg.timeout_s

    async def sh(argv: list[str]) -> CmdResult:
        return await runner(argv, t)

    # -- notificaciones -----------------------------------------------------------------------
    async def notify(a: dict[str, Any]) -> str:
        argv = ["notify-send", "-a", "POETA", "--", _clean(a["title"], 80)]
        if a.get("body"):
            argv.append(_clean(a["body"], 300))
        res = await sh(argv)
        if res.rc:
            raise _fail(res, "la notificacion")
        return "Notificacion enviada."

    # -- multimedia (MPRIS via playerctl) ---------------------------------------------------------
    async def media_status(a: dict[str, Any]) -> str:
        res = await sh(["playerctl", "-a", "metadata", "--format", "{{playerName}}: {{status}} | {{artist}} - {{title}}"])
        if res.rc or not res.out.strip():
            return "No hay reproductores multimedia activos."
        return "\n".join(_clean(line, 200) for line in res.out.splitlines()[:6])

    cmds = {"play": "play", "pause": "pause", "play_pause": "play-pause", "next": "next", "previous": "previous",
            "stop": "stop"}

    async def players() -> list[tuple[str, str]]:
        res = await sh(["playerctl", "-l"])
        names = [n.strip() for n in res.out.splitlines() if _PLAYER.match(n.strip())] if not res.rc else []

        async def status(n: str) -> str:
            r = await sh(["playerctl", "-p", n, "status"])
            return r.out.strip() if not r.rc else "?"

        return list(zip(names, await asyncio.gather(*(status(n) for n in names))))

    def is_browser(name: str) -> bool:
        return name.lower().split(".")[0] in BROWSERS

    def choose(found: list[tuple[str, str]], wanted: str | None) -> str:
        if wanted:
            hits = [n for n, _ in found if wanted.lower() in n.lower()]
            if not hits:
                raise ToolError(f"no hay un reproductor '{wanted}'. Activos: {', '.join(n for n, _ in found) or 'ninguno'}")
            return hits[0]
        playing = [n for n, st in found if st == "Playing"]
        if playing:
            return playing[0]
        # Nadie suena: los navegadores casi siempre tienen una pestana en pausa; preferir una app de musica/video.
        return next((n for n, _ in found if not is_browser(n)), found[0][0])

    async def media_control(a: dict[str, Any]) -> str:
        if a.get("player") and not _PLAYER.match(a["player"]):
            raise ToolError("nombre de reproductor invalido")
        found = await players()
        waited = 0.0
        # Acaba de abrirse la app (ej. Spotify tarda unos segundos en aparecer): esperar a que se registre.
        while not found and a["command"] in ("play", "play_pause") and waited < cfg.media_wait_s:
            await asyncio.sleep(1.0)
            waited += 1.0
            found = await players()
        if not found:
            raise ToolError("no hay ningun reproductor multimedia activo")
        target = choose(found, a.get("player"))
        res = await sh(["playerctl", "-p", target, cmds[a["command"]]])
        if res.rc:
            raise _fail(res, f"enviar la orden a {target}")
        return f"Orden '{a['command']}' enviada a {target.split('.')[0]}."

    # -- volumen (PipeWire via wpctl) ----------------------------------------------------------------
    async def get_volume(a: dict[str, Any]) -> str:
        res = await sh(["wpctl", "get-volume", "@DEFAULT_AUDIO_SINK@"])
        m = re.search(r"Volume:\s*([0-9.]+)", res.out)
        if res.rc or not m:
            raise _fail(res, "leer el volumen")
        return f"Volumen: {round(float(m.group(1)) * 100)} %" + (" (silenciado)" if "MUTED" in res.out else "")

    async def set_volume(a: dict[str, Any]) -> str:
        if "percent" not in a and "muted" not in a:
            raise ToolError("indica 'percent' y/o 'muted'")
        done = []
        if "percent" in a:
            p = a["percent"]
            if p < 0 or p > cfg.max_volume_percent:
                raise ToolError(f"el volumen debe estar entre 0 y {cfg.max_volume_percent} %")
            res = await sh(["wpctl", "set-volume", "@DEFAULT_AUDIO_SINK@", f"{p / 100:.2f}"])
            if res.rc:
                raise _fail(res, "cambiar el volumen")
            done.append(f"volumen {p} %")
        if "muted" in a:
            res = await sh(["wpctl", "set-mute", "@DEFAULT_AUDIO_SINK@", "1" if a["muted"] else "0"])
            if res.rc:
                raise _fail(res, "cambiar el silencio")
            done.append("silenciado" if a["muted"] else "sonido activado")
        return "Listo: " + ", ".join(done) + "."

    # -- ventanas (KWin via kdotool; Wayland) ----------------------------------------------------------
    async def win_title(wid: str) -> str:
        res = await sh(["kdotool", "getwindowname", wid])
        return res.out.strip() if not res.rc else ""

    cache: dict[str, Any] = {"at": -1e9, "rows": []}

    async def windows(fresh: bool = False) -> list[tuple[str, str]]:
        """[(id, titulo)] de las ventanas. Los titulos se buscan de a uno (cada kdotool carga su script en KWin)."""
        if not fresh and time.monotonic() - cache["at"] < 5:
            return cache["rows"]
        res = await sh(["kdotool", "search", "--name", "."])  # patron constante: nunca texto del modelo (se inserta en JS)
        if res.rc:
            raise _fail(res, "listar ventanas")
        ids = [i.strip() for i in res.out.splitlines() if _WINID.match(i.strip())][:30]
        rows = []
        for i in ids:
            title = await win_title(i)
            if title:
                rows.append((i, title))
        if ids and not rows:
            raise ToolError("kdotool devolvio ventanas pero ningun titulo (python scripts/kdotool_probe.py)")
        cache.update(at=time.monotonic(), rows=rows)
        return rows

    async def list_windows(a: dict[str, Any]) -> str:
        rows = await windows(fresh=True)
        return "\n".join(f"{i}  {_clean(ti, 120)}" for i, ti in rows) or "No hay ventanas."

    plan: dict[str, tuple[str, str]] = {}  # clave del pedido -> (id, titulo REAL mostrado al pedir confirmacion)
    verbs = {"activate": "windowactivate", "minimize": "windowminimize", "close": "windowclose"}

    def plan_key(a: dict[str, Any]) -> str:
        return f"{a.get('id') or ''}|{(a.get('match') or '').casefold()}"

    async def window_describe(a: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        if bool(a.get("id")) == bool(a.get("match")):
            raise ToolError("indica exactamente uno: 'id' (de list_windows) o 'match' (parte del titulo)")
        if a.get("id"):
            if not _WINID.match(a["id"]):
                raise ToolError("id de ventana invalido (usa los de list_windows)")
            wid, real = a["id"], await win_title(a["id"])  # el titulo viene de KWin, no del modelo
            if not real:
                raise ToolError("esa ventana no existe (llama a list_windows de nuevo)")
        else:
            hits = [(i, ti) for i, ti in await windows(fresh=True) if a["match"].casefold() in ti.casefold()]
            if not hits:
                raise ToolError(f"ninguna ventana contiene '{_clean(a['match'], 40)}' (usa list_windows)")
            if len(hits) > 1:
                raise ToolError(f"{len(hits)} ventanas coinciden con '{_clean(a['match'], 40)}': "
                                "usa list_windows y pasa el 'id' exacto")
            wid, real = hits[0]
        plan[plan_key(a)] = (wid, _clean(real, 80))
        verb = {"activate": "Traer al frente", "minimize": "Minimizar", "close": "Cerrar"}[a["action"]]
        return f"{verb} la ventana '{_clean(real, 80)}'", {"window": wid, "title": _clean(real, 80), "action": a["action"]}

    async def window_action(a: dict[str, Any]) -> str:
        chosen = plan.pop(plan_key(a), None)
        if chosen is None:
            raise ToolError("no hay una ventana autorizada para esta orden")
        wid, shown = chosen
        real = await win_title(wid)
        if not real:
            raise ToolError("esa ventana ya no existe")
        if _clean(real, 80) != shown:
            raise ToolError("la ventana cambio entre la confirmacion y la accion: no hice nada")
        res = await sh(["kdotool", verbs[a["action"]], wid])
        if res.rc:
            raise _fail(res, "la accion sobre la ventana")
        return f"Hecho: {a['action']} '{_clean(real, 60)}'."

    # -- servicios de usuario (systemd --user) -------------------------------------------------------------
    def unit_of(a: dict[str, Any]) -> str:
        if not _UNIT.match(a["unit"]):
            raise ToolError("nombre de unidad invalido (ejemplo: llama-server.service)")
        return a["unit"]

    async def service_status(a: dict[str, Any]) -> str:
        res = await sh(["systemctl", "--user", "show", unit_of(a), "--no-pager",
                        "--property=Description,LoadState,ActiveState,SubState,UnitFileState"])
        if res.rc:
            raise _fail(res, "consultar el servicio")
        return _clean(res.out.replace("\n", "; "), 400)

    def service_describe(a: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        unit = unit_of(a)
        if unit not in cfg.service_allowlist:
            raise ToolError(f"'{unit}' no esta en desktop.services.allowlist: no puedo controlarlo")
        return f"{a['action']} el servicio de usuario {unit}", {"unit": unit, "action": a["action"]}

    async def service_control(a: dict[str, Any]) -> str:
        unit = unit_of(a)
        res = await sh(["systemctl", "--user", a["action"], unit])
        if res.rc:
            raise _fail(res, f"{a['action']} {unit}")
        return f"Hecho: {a['action']} {unit}."

    # -- procesos -------------------------------------------------------------------------------------------
    async def list_processes(a: dict[str, Any]) -> str:
        key = "-pcpu" if a.get("sort", "cpu") == "cpu" else "-pmem"
        n = min(a.get("limit", 10), cfg.max_listed)
        res = await sh(["ps", "-u", str(os.getuid()), "-o", "pid=,pcpu=,pmem=,comm=", f"--sort={key}"])
        if res.rc:
            raise _fail(res, "listar procesos")
        rows = [" ".join(line.split(None, 3)) for line in res.out.splitlines()[:n]]
        return "PID %CPU %MEM NOMBRE\n" + "\n".join(_clean(r, 80) for r in rows)

    def proc_info(pid: int) -> str:
        if pid <= 1 or pid in (os.getpid(), os.getppid()):
            raise ToolError("ese proceso esta protegido")
        try:
            if (proc_dir / str(pid)).stat().st_uid != os.getuid():
                raise ToolError("ese proceso no es tuyo: no puedo detenerlo")
            name = (proc_dir / str(pid) / "comm").read_text().strip()
        except (FileNotFoundError, ProcessLookupError):
            raise ToolError("ese proceso no existe") from None
        except PermissionError:
            raise ToolError("sin permiso sobre ese proceso") from None
        if name in CRITICAL:
            raise ToolError(f"'{name}' es un proceso critico del sistema: no lo detengo")
        return name

    def stop_describe(a: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        name = proc_info(a["pid"])
        return f"Detener (SIGTERM) el proceso {name} (PID {a['pid']})", {"pid": a["pid"], "process": name}

    async def stop_process(a: dict[str, Any]) -> str:
        name = proc_info(a["pid"])
        if name != a["name"]:
            raise ToolError(f"el PID {a['pid']} ya no es '{a['name']}' (ahora es '{name}'): no hice nada")
        try:
            killer(a["pid"], signal.SIGTERM)  # solo SIGTERM: el programa puede cerrar ordenadamente
        except ProcessLookupError:
            raise ToolError("el proceso ya termino") from None
        except PermissionError:
            raise ToolError("sin permiso para detener ese proceso") from None
        return f"Le pedi a {name} (PID {a['pid']}) que termine."

    S, I, B = {"type": "string"}, {"type": "integer"}, {"type": "boolean"}
    factories: dict[str, Callable[[], Tool]] = {
        "notify": lambda: Tool("notify", "Muestra una notificacion de escritorio al usuario.",
                               _obj({"title": S, "body": S}, ["title"]), notify, "desktop_notify"),
        "media_status": lambda: Tool("media_status", "Dice que se esta reproduciendo en los reproductores multimedia.",
                                     _obj({}), media_status, "desktop_read", untrusted=True),
        "media_control": lambda: Tool(
            "media_control", "Controla la reproduccion multimedia. Sin 'player' elige el que esta sonando o, si ninguno "
            "suena, una app de musica antes que el navegador. 'player' = nombre del reproductor (ej. spotify, vlc). "
            "Para empezar a sonar usa 'play'; 'play_pause' alterna.",
            _obj({"command": {"type": "string", "enum": list(cmds)}, "player": S}, ["command"]),
            media_control, "desktop_control", timeout_s=t * 3 + cfg.media_wait_s + 5,
            describe=lambda a: (f"Multimedia: {a['command']}" + (f" en {a['player']}" if a.get("player") else ""),
                                {"command": a["command"], "player": a.get("player")})),
        "get_volume": lambda: Tool("get_volume", "Lee el volumen de salida actual.", _obj({}), get_volume, "desktop_read"),
        "set_volume": lambda: Tool(
            "set_volume", f"Cambia el volumen (0-{cfg.max_volume_percent} %) y/o el silencio.",
            _obj({"percent": I, "muted": B}), set_volume, "desktop_control",
            describe=lambda a: ("Cambiar el volumen/silencio", {k: a[k] for k in ("percent", "muted") if k in a})),
        "list_windows": lambda: Tool("list_windows", "Lista las ventanas abiertas (id y titulo).", _obj({}),
                                     list_windows, "desktop_read", untrusted=True, timeout_s=t * 8),
        "window_action": lambda: Tool(
            "window_action", "Trae al frente, minimiza o cierra una ventana. Indica 'match' (parte del titulo, ej. "
            "'Konsole'; debe coincidir con UNA sola ventana) o el 'id' exacto de list_windows.",
            _obj({"id": S, "match": S, "action": {"type": "string", "enum": ["activate", "minimize", "close"]}},
                 ["action"]), window_action, "window_control", describe=window_describe, timeout_s=t * 8),
        "service_status": lambda: Tool("service_status", "Estado de un servicio systemd de usuario.",
                                       _obj({"unit": S}, ["unit"]), service_status, "service_read"),
        "service_control": lambda: Tool(
            "service_control", "Inicia, detiene o reinicia un servicio systemd de usuario autorizado.",
            _obj({"unit": S, "action": {"type": "string", "enum": ["start", "stop", "restart"]}}, ["unit", "action"]),
            service_control, "service_control", describe=service_describe),
        "list_processes": lambda: Tool(
            "list_processes", "Lista tus procesos que mas CPU o memoria usan.",
            _obj({"sort": {"type": "string", "enum": ["cpu", "mem"]}, "limit": I}), list_processes, "process_read",
            untrusted=True),
        "stop_process": lambda: Tool(
            "stop_process", "Pide a un proceso tuyo que termine (SIGTERM). Usa PID y nombre de list_processes.",
            _obj({"pid": I, "name": S}, ["pid", "name"]), stop_process, "process_stop", describe=stop_describe),
    }
    registered, skipped = [], {}
    for name in cfg.tools:
        need = REQUIRES[name]
        if need and not which(need):
            skipped[name] = f"falta '{need}'"
            continue
        registry.register(factories[name]())
        registered.append(name)
    for tool in registry.tools():
        if tool.name in registered:
            tool.source = "desktop"
    return registered, skipped
