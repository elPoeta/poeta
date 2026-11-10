"""Une los gestos con la conversacion: la palma interrumpe a POETA y los pulgares responden confirmaciones.

Politica (secc. 9.3: un gesto nunca autoriza solo una accion sensible):
- La palma sostenida detiene la respuesta en curso. Con una confirmacion pendiente NO interrumpe: esa misma palma,
  sostenida 1.5 s, es la que activa la interaccion para poder usar los pulgares.
- Pulgar arriba/abajo solo responden confirmaciones de nivel 3 (policy `confirm`), jamas las explicitas (nivel 4,
  contenido con imagenes o audio), cloud_transfer, ni las que siguen a datos no confiables (web, ventanas, MCP...).
  Esas siempre se contestan con el teclado. El motor de gestos ademas exige la interaccion activada.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Callable

log = logging.getLogger("poeta.gestures")

NEVER_BY_GESTURE = {"cloud_transfer"}


@dataclass(frozen=True)
class GesturePolicy:
    confirm_with_gestures: bool = True
    interrupt_with_palm: bool = True

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "GesturePolicy":
        g = (cfg.get("vision") or {}).get("gestures") or {}
        for k in ("confirm_with_gestures", "interrupt_with_palm"):
            if not isinstance(g.get(k, True), bool):
                from poeta.vision import VisionConfigError
                raise VisionConfigError(f"vision.gestures.{k} debe ser true o false")
        return cls(g.get("confirm_with_gestures", True), g.get("interrupt_with_palm", True))


class GestureHub:
    def __init__(self, policy: GesturePolicy, bus: Any = None, audit: Any = None, out: Callable[..., None] = print):
        self.policy, self.bus, self.audit, self.out = policy, bus, audit, out
        self.active = False  # el servicio de gestos esta corriendo
        self.on_interrupt: Callable[[], None] | None = None
        self._pending: asyncio.Future[tuple[bool, str]] | None = None

    # -- eventos del motor (llamar desde el bucle de eventos) -------------------------------------------------
    def on_event(self, ev: Any) -> None:
        if self.bus is not None:
            self.bus.publish("gesture", name=ev.name)
        pending = self._pending
        if pending is not None and not pending.done():
            if ev.name == "thumb_up":
                pending.set_result((True, ev.name))
            elif ev.name == "thumb_down":
                pending.set_result((False, ev.name))
            return  # con una confirmacion pendiente, la palma sirve para activar: no interrumpe
        if ev.name == "palm" and self.policy.interrupt_with_palm and self.on_interrupt is not None:
            self.on_interrupt()

    # -- confirmaciones ------------------------------------------------------------------------------------------
    def eligible(self, req: Any, explicit: bool) -> bool:
        return (self.active and self.policy.confirm_with_gestures and not explicit and not getattr(req, "tainted", False)
                and getattr(req, "kind", "") not in NEVER_BY_GESTURE)

    def begin(self) -> "asyncio.Future[tuple[bool, str]]":
        self._pending = asyncio.get_running_loop().create_future()
        return self._pending

    def end(self) -> None:
        self._pending = None

    @property
    def waiting(self) -> bool:
        return self._pending is not None and not self._pending.done()


def make_gesture_confirmer(base: Callable[..., Any], hub: GestureHub) -> Callable[..., Any]:
    """Envuelve el confirmador del teclado: gana la primera respuesta, del teclado o de un gesto."""

    async def confirm(req: Any, explicit: bool) -> bool:
        if not hub.eligible(req, explicit):
            return await base(req, explicit)
        fut = hub.begin()
        hub.out("  (con gestos: mano abierta 1.5 s para activar; luego pulgar arriba = autorizar, abajo = cancelar)")
        if hub.bus is not None:
            hub.bus.set_state("confirming")
        key = asyncio.ensure_future(base(req, explicit))
        try:
            done, _ = await asyncio.wait({key, fut}, return_when=asyncio.FIRST_COMPLETED)
            if fut in done:
                allowed, gesture = fut.result()
                key.cancel()
                await asyncio.gather(key, return_exceptions=True)
                hub.out(f"\n(decidido con gesto: {'autorizado' if allowed else 'cancelado'})")
                if hub.audit is not None:
                    try:
                        hub.audit.record("gesture_confirmation", kind=req.kind, tool=(req.details or {}).get("tool"),
                                         allowed=allowed, gesture=gesture)
                    except Exception:  # noqa: BLE001
                        log.exception("no se pudo auditar la confirmacion por gesto")
                return allowed
            return key.result()
        finally:
            hub.end()
            for t in (key, fut):
                if not t.done():
                    t.cancel()
            await asyncio.gather(key, fut, return_exceptions=True)

    return confirm


def make_announcing_confirmer(base: Callable[..., Any], bus: Any,
                              channels: Callable[[Any, bool], list[str]] | None = None) -> Callable[..., Any]:
    """Publica en el bus que se pidio una confirmacion, por que canales se puede responder y que se decidio.

    El evento lleva la accion (`kind`), la herramienta y una descripcion corta para que el panel y el HUD muestren
    QUE se esta pidiendo; el contenido completo (`payload`) nunca sale por aca.
    """

    async def confirm(req: Any, explicit: bool) -> bool:
        try:
            via = list(channels(req, explicit)) if channels else []
        except Exception:  # noqa: BLE001
            via = []
        info = {"kind": req.kind, "tool": (req.details or {}).get("tool"), "explicit": bool(explicit),
                "tainted": bool(getattr(req, "tainted", False)), "via": via}
        bus.publish("permission", phase="request", description=" ".join(str(req.description).split())[:120], **info)
        allowed = False
        try:
            allowed = await base(req, explicit)
            return allowed
        finally:
            bus.publish("permission", phase="decided", allowed=bool(allowed), **info)
            bus.set_state("thinking")

    return confirm
