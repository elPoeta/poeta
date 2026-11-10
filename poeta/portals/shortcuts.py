"""Atajos globales (org.freedesktop.portal.GlobalShortcuts): una tecla del sistema para hablar con POETA o callarlo.

KDE mostrara su propio dialogo la primera vez; el atajo se puede reasignar despues en Preferencias del sistema -> Atajos.
"""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from .dbus import PortalBus, PortalError, variant

log = logging.getLogger("poeta.portals")

GS = "org.freedesktop.portal.GlobalShortcuts"
HOST_REGISTRY = "org.freedesktop.host.portal.Registry"
SESSION_IFACE = "org.freedesktop.portal.Session"
_APP_ID = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_-]*)+$")
_TRIGGER = re.compile(r"^[A-Za-z0-9_+ ]{1,60}$")
KNOWN_SHORTCUTS = ("talk", "stop")


class PortalsConfigError(ValueError):
    pass


@dataclass(frozen=True)
class ScreenshotSettings:
    enabled: bool = True
    timeout_s: float = 30.0


@dataclass(frozen=True)
class PortalsConfig:
    enabled: bool = False
    app_id: str = "org.poeta.Assistant"
    shortcuts_enabled: bool = True
    shortcuts: dict[str, tuple[str, str]] = field(default_factory=lambda: {
        "talk": ("Hablar con POETA (o interrumpirlo)", "CTRL+ALT+SPACE"),
        "stop": ("Callar a POETA", "")})
    screenshot: ScreenshotSettings = ScreenshotSettings()

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "PortalsConfig":
        raw = cfg.get("portals") or {}
        for key in ("enabled",):
            if not isinstance(raw.get(key, False), bool):
                raise PortalsConfigError(f"portals.{key} debe ser true o false")
        app_id = raw.get("app_id", cls.app_id)
        if not isinstance(app_id, str) or not _APP_ID.match(app_id):
            raise PortalsConfigError("portals.app_id debe tener forma de dominio inverso (ej. org.poeta.Assistant)")
        sc = raw.get("shortcuts") or {}
        if not isinstance(sc.get("enabled", True), bool):
            raise PortalsConfigError("portals.shortcuts.enabled debe ser true o false")
        shortcuts = dict(cls().shortcuts)
        for name in KNOWN_SHORTCUTS:
            item = sc.get(name)
            if item is None:
                continue
            if not isinstance(item, dict):
                raise PortalsConfigError(f"portals.shortcuts.{name} debe ser un mapa con trigger/description")
            trig = item.get("trigger", shortcuts[name][1])
            desc = item.get("description", shortcuts[name][0])
            if not isinstance(trig, str) or (trig and not _TRIGGER.match(trig)):
                raise PortalsConfigError(f"portals.shortcuts.{name}.trigger invalido (ej. CTRL+ALT+SPACE)")
            if not isinstance(desc, str) or not desc.strip() or len(desc) > 100:
                raise PortalsConfigError(f"portals.shortcuts.{name}.description debe ser un texto corto")
            shortcuts[name] = (desc.strip(), trig.strip())
        sh = raw.get("screenshot") or {}
        if not isinstance(sh.get("enabled", True), bool):
            raise PortalsConfigError("portals.screenshot.enabled debe ser true o false")
        t = sh.get("timeout_s", 30)
        if isinstance(t, bool) or not isinstance(t, (int, float)) or not 5 <= t <= 300:
            raise PortalsConfigError("portals.screenshot.timeout_s debe estar entre 5 y 300")
        return cls(raw.get("enabled", False), app_id, sc.get("enabled", True), shortcuts,
                   ScreenshotSettings(sh.get("enabled", True), float(t)))


class ShortcutService:
    """Registra los atajos en el portal y despacha `Activated` a las funciones indicadas."""

    def __init__(self, portal: PortalBus, config: PortalsConfig, handlers: dict[str, Callable[[], None]],
                 out: Callable[..., None] = print):
        self.portal, self.config, self.handlers, self.out = portal, config, handlers, out
        self.session: str | None = None
        self.bound: list[str] = []
        self.error: str | None = None

    async def start(self) -> bool:
        names = [n for n in self.handlers if n in self.config.shortcuts]
        try:
            try:  # portales >= 1.19: un programa fuera de un paquete se identifica con un app_id
                await self.portal.call(HOST_REGISTRY, "Register", "sa{sv}", [self.config.app_id, {}])
            except PortalError as exc:
                log.info("Registry.Register no disponible (%s); se sigue sin registrar el app_id", exc)
            code, res = await self.portal.request(
                GS, "CreateSession", "", [], {"session_handle_token": "poeta" + uuid.uuid4().hex[:10]})
            if code != 0 or "session_handle" not in res:
                raise PortalError("el portal no creo la sesion de atajos")
            self.session = str(res["session_handle"])
            await self.portal.on_signal(GS, "Activated", self._activated)
            shortcuts = []
            for n in names:
                desc, trig = self.config.shortcuts[n]
                props: dict[str, Any] = {"description": variant("s", desc)}
                if trig:
                    props["preferred_trigger"] = variant("s", trig)
                shortcuts.append([n, props])
            code, res = await self.portal.request(GS, "BindShortcuts", "oa(sa{sv})s", [self.session, shortcuts, ""],
                                                  timeout=120)
            if code != 0:
                raise PortalError("no se autorizaron los atajos" if code == 1 else "el portal rechazo los atajos")
            self.bound = names
            return True
        except PortalError as exc:
            self.error = str(exc)
            return False

    def _activated(self, session: str, shortcut_id: str, *_rest: Any) -> None:
        if session != self.session:
            return
        fn = self.handlers.get(shortcut_id)
        if fn is not None:
            fn()

    async def stop(self) -> None:
        if self.session:
            try:
                await self.portal.call(SESSION_IFACE, "Close", "", [], path=self.session)
            except PortalError:
                pass
            self.session = None
