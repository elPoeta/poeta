"""Cliente minimo del bus de sesion D-Bus para los portales (dbus-fast, asyncio).

Los portales responden en dos tiempos: el metodo devuelve el camino de un objeto `Request` y, mas tarde, ese objeto emite
la senal `Response(code, results)`. Se suscribe ANTES de llamar para no perder una respuesta inmediata.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any, Callable

log = logging.getLogger("poeta.portals")

PORTAL_NAME = "org.freedesktop.portal.Desktop"
PORTAL_PATH = "/org/freedesktop/portal/desktop"
REQUEST_IFACE = "org.freedesktop.portal.Request"


class PortalError(RuntimeError):
    """El portal no esta disponible, rechazo la llamada o el usuario cancelo."""


def _plain(value: Any) -> Any:
    """Variant de dbus-fast -> valor de Python."""
    if hasattr(value, "signature") and hasattr(value, "value"):
        return _plain(value.value)
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def variant(signature: str, value: Any) -> Any:
    try:
        from dbus_fast import Variant
    except ImportError as exc:
        raise PortalError("falta dbus-fast: uv pip install dbus-fast") from exc
    return Variant(signature, value)


class PortalBus:
    def __init__(self, bus_address: str | None = None):
        self._address = bus_address
        self._bus: Any = None
        self._waiters: dict[str, asyncio.Future] = {}
        self._signals: list[tuple[str, str, Callable[..., None]]] = []
        self._matches: set[str] = set()
        self._connect_lock = asyncio.Lock()

    @property
    def connected(self) -> bool:
        return self._bus is not None and self._bus.connected

    async def connect(self) -> None:
        async with self._connect_lock:
            if self.connected:
                return
            try:
                from dbus_fast import BusType
                from dbus_fast.aio import MessageBus
            except ImportError as exc:
                raise PortalError("falta dbus-fast: uv pip install dbus-fast") from exc
            try:
                kw = {"bus_address": self._address} if self._address else {"bus_type": BusType.SESSION}
                self._bus = await MessageBus(**kw).connect()
            except Exception as exc:  # noqa: BLE001
                raise PortalError(f"no se pudo conectar al bus de sesion D-Bus: {exc}") from exc
            self._bus.add_message_handler(self._on_message)
            self._matches.clear()

    async def close(self) -> None:
        bus, self._bus = self._bus, None
        for fut in self._waiters.values():
            if not fut.done():
                fut.cancel()
        self._waiters.clear()
        if bus is not None:
            bus.disconnect()

    # -- internos -------------------------------------------------------------------------------------------
    async def _raw_call(self, destination: str, path: str, interface: str, member: str, signature: str = "",
                        body: list[Any] | None = None) -> Any:
        from dbus_fast import Message, MessageType

        reply = await self._bus.call(Message(destination=destination, path=path, interface=interface, member=member,
                                             signature=signature, body=body or []))
        if reply.message_type == MessageType.ERROR:
            raise PortalError(f"{reply.error_name}: {reply.body[0] if reply.body else ''}")
        return reply

    async def _add_match(self, interface: str, member: str) -> None:
        rule = f"type='signal',interface='{interface}',member='{member}'"
        if rule in self._matches:
            return
        await self._raw_call("org.freedesktop.DBus", "/org/freedesktop/DBus", "org.freedesktop.DBus", "AddMatch", "s", [rule])
        self._matches.add(rule)

    def _on_message(self, msg: Any) -> None:
        from dbus_fast import MessageType

        if msg.message_type != MessageType.SIGNAL:
            return
        if msg.interface == REQUEST_IFACE and msg.member == "Response":
            fut = self._waiters.pop(msg.path, None)
            if fut is not None and not fut.done():
                code, results = msg.body
                fut.set_result((int(code), _plain(results)))
            return
        for iface, member, cb in list(self._signals):
            if msg.interface == iface and msg.member == member:
                try:
                    cb(*[_plain(b) for b in msg.body])
                except Exception:  # noqa: BLE001
                    log.exception("fallo el manejador de la senal %s.%s", iface, member)

    # -- API ------------------------------------------------------------------------------------------------------
    async def on_signal(self, interface: str, member: str, callback: Callable[..., None]) -> None:
        await self.connect()
        self._signals.append((interface, member, callback))
        await self._add_match(interface, member)

    async def call(self, interface: str, method: str, signature: str = "", args: list[Any] | None = None,
                   path: str = PORTAL_PATH) -> Any:
        await self.connect()
        reply = await self._raw_call(PORTAL_NAME, path, interface, method, signature, args)
        return _plain(reply.body)

    async def request(self, interface: str, method: str, signature: str, args: list[Any],
                      options: dict[str, Any] | None = None, timeout: float = 30.0) -> tuple[int, dict[str, Any]]:
        """Llama a un metodo del portal que responde con un `Request`. Devuelve (codigo, resultados).

        codigo 0 = ok, 1 = el usuario cancelo, 2 = otro fallo.
        """
        await self.connect()
        await self._add_match(REQUEST_IFACE, "Response")
        token = "poeta" + uuid.uuid4().hex[:12]
        opts = {**{k: v if hasattr(v, "signature") else variant("s", v) for k, v in (options or {}).items()},
                "handle_token": variant("s", token)}
        sender = self._bus.unique_name[1:].replace(".", "_")
        expected = f"{PORTAL_PATH}/request/{sender}/{token}"
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._waiters[expected] = fut
        try:
            reply = await self._raw_call(PORTAL_NAME, PORTAL_PATH, interface, method, signature + "a{sv}", [*args, opts])
            actual = reply.body[0] if reply.body else expected
            if actual != expected:  # portales antiguos: el camino real lo dicta la respuesta
                self._waiters[actual] = self._waiters.pop(expected, fut)
            return await asyncio.wait_for(fut, timeout)
        except asyncio.TimeoutError as exc:
            raise PortalError(f"el portal no respondio en {timeout:.0f} s") from exc
        finally:
            for k in [k for k, v in self._waiters.items() if v is fut]:
                self._waiters.pop(k, None)
