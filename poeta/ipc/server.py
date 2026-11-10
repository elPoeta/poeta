"""Servidor local del panel: socket Unix privado (0600, solo tu usuario), eventos en vivo y comandos permitidos.

Seguridad: directorio 0700 propio; se verifica el uid del otro extremo (SO_PEERCRED); solo comandos de una lista fija
con argumentos validados; sin acceso a herramientas ni a datos del agente salvo lo que `query` expone al usuario.
Un cliente lento no frena al agente: su cola es acotada y, si se llena, se lo desconecta.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import socket
import stat
import struct
from pathlib import Path
from typing import Any, Awaitable, Callable

from .protocol import MAX_LINE, VERSION, ProtocolError, decode, encode

log = logging.getLogger("poeta.ipc")

Handler = Callable[[dict[str, Any]], Awaitable[Any]]
MAX_CLIENTS = 8
QUEUE_SIZE = 256


class IpcError(RuntimeError):
    pass


def default_socket_path() -> Path:
    runtime = os.environ.get("XDG_RUNTIME_DIR")
    base = Path(runtime) if runtime else Path(f"/tmp/poeta-{os.getuid()}")
    return base / "poeta" / "ui.sock"


def _private_dir(path: Path) -> None:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    st = path.stat()
    if st.st_uid != os.getuid() or stat.S_IMODE(st.st_mode) & 0o077:
        raise IpcError(f"{path} debe ser tuyo y privado (0700); revisalo o borralo")


def peer_uid(sock: socket.socket) -> int | None:
    try:
        _pid, uid, _gid = struct.unpack("3i", sock.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i")))
        return uid
    except (OSError, AttributeError, struct.error):
        return None


class UiServer:
    def __init__(self, bus: Any, snapshot: Callable[[], dict[str, Any]], handlers: dict[str, Handler],
                 path: Path | str | None = None):
        self.bus, self.snapshot, self.handlers = bus, snapshot, dict(handlers)
        self.path = Path(path) if path else default_socket_path()
        self._server: asyncio.AbstractServer | None = None
        self._clients: set["_Client"] = set()
        self._unsub: Callable[[], None] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        _private_dir(self.path.parent)
        if self.path.exists() or self.path.is_symlink():
            if not stat.S_ISSOCK(self.path.lstat().st_mode):
                raise IpcError(f"{self.path} existe y no es un socket: no lo toco")
            if await self._alive():
                raise IpcError("ya hay otra instancia de POETA con el panel abierto")
            self.path.unlink()  # socket viejo de una ejecucion que murio
        old = os.umask(0o177)
        try:
            self._server = await asyncio.start_unix_server(self._on_client, path=str(self.path), limit=MAX_LINE)
        finally:
            os.umask(old)
        os.chmod(self.path, 0o600)
        self._unsub = self.bus.subscribe(self._on_event)
        log.info("panel: escuchando en %s", self.path)

    async def _alive(self) -> bool:
        try:
            _, w = await asyncio.wait_for(asyncio.open_unix_connection(str(self.path)), 1.0)
        except (OSError, asyncio.TimeoutError):
            return False
        w.close()
        return True

    async def stop(self) -> None:
        if self._unsub:
            self._unsub()
        clients = list(self._clients)
        if self._server:
            self._server.close()
        for c in clients:
            c.close()
        if clients:  # dejar que los transportes terminen de cerrarse antes de soltar el servidor
            await asyncio.wait([asyncio.ensure_future(c.writer.wait_closed()) for c in clients], timeout=1.0)
        if self._server:
            await self._server.wait_closed()
        with contextlib.suppress(OSError):
            self.path.unlink()

    # -- eventos del bus (pueden llegar desde otros hilos) ------------------------------------------------------
    def _on_event(self, ev: Any) -> None:
        msg = {"t": "event", "kind": ev.kind, "data": ev.data, "ts": ev.ts}
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        try:
            asyncio.get_running_loop()
            self._fanout(msg)
        except RuntimeError:
            loop.call_soon_threadsafe(self._fanout, msg)

    def _fanout(self, msg: dict[str, Any]) -> None:
        for c in list(self._clients):
            c.send(msg)

    async def _on_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        sock = writer.get_extra_info("socket")
        uid = peer_uid(sock) if sock is not None else None
        if uid != os.getuid():
            log.warning("panel: conexion rechazada (uid %s)", uid)
            writer.close()
            return
        if len(self._clients) >= MAX_CLIENTS:
            writer.close()
            return
        client = _Client(self, reader, writer)
        self._clients.add(client)
        try:
            await client.run()
        finally:
            self._clients.discard(client)
            client.close()

    @property
    def clients(self) -> int:
        return len(self._clients)


class _Client:
    def __init__(self, server: UiServer, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        self.server, self.reader, self.writer = server, reader, writer
        self.queue: asyncio.Queue[bytes | None] = asyncio.Queue(QUEUE_SIZE)
        self._closed = False

    def send(self, msg: dict[str, Any]) -> None:
        if self._closed:
            return
        try:
            self.queue.put_nowait(encode(msg))
        except asyncio.QueueFull:
            log.warning("panel: cliente demasiado lento, desconectado")
            self.close(abort=True)
        except ProtocolError:
            log.warning("panel: mensaje demasiado grande, descartado")

    def close(self, abort: bool = False) -> None:
        if self._closed:
            return
        self._closed = True
        with contextlib.suppress(asyncio.QueueFull):
            self.queue.put_nowait(None)
        with contextlib.suppress(Exception):
            # close() espera a vaciar lo pendiente: un cliente que no lee lo tendria abierto para siempre
            self.writer.transport.abort() if abort else self.writer.close()

    async def _writer_loop(self) -> None:
        try:
            while (data := await self.queue.get()) is not None:
                self.writer.write(data)
                await self.writer.drain()
        except (ConnectionError, OSError):
            pass
        finally:
            self.close()

    async def run(self) -> None:
        self.send({"t": "hello", "v": VERSION})
        try:
            self.send({"t": "snapshot", "data": self.server.snapshot()})
        except Exception:  # noqa: BLE001
            log.exception("panel: no se pudo armar el snapshot")
        wtask = asyncio.create_task(self._writer_loop())
        try:
            while not self._closed:
                try:
                    line = await self.reader.readline()
                except (asyncio.LimitOverrunError, ValueError):
                    break
                if not line:
                    break
                await self._handle(line)
        except (ConnectionError, OSError):
            pass
        finally:
            self.close()
            wtask.cancel()
            await asyncio.gather(wtask, return_exceptions=True)

    async def _handle(self, line: bytes) -> None:
        try:
            msg = decode(line)
        except ProtocolError as exc:
            self.send({"t": "reply", "id": None, "ok": False, "error": str(exc)})
            return
        if msg["t"] != "cmd":
            self.send({"t": "reply", "id": msg.get("id"), "ok": False, "error": "tipo de mensaje no soportado"})
            return
        cid, name, args = msg.get("id"), msg.get("name"), msg.get("args") or {}
        handler = self.server.handlers.get(name) if isinstance(name, str) else None
        if handler is None or not isinstance(args, dict):
            self.send({"t": "reply", "id": cid, "ok": False, "error": f"comando desconocido: {name!r}"})
            return
        try:
            data = await asyncio.wait_for(handler(args), 10.0)
            self.send({"t": "reply", "id": cid, "ok": True, "data": data})
        except (ValueError, KeyError, LookupError) as exc:
            self.send({"t": "reply", "id": cid, "ok": False, "error": str(exc)[:200]})
        except asyncio.TimeoutError:
            self.send({"t": "reply", "id": cid, "ok": False, "error": "tiempo agotado"})
        except Exception:  # noqa: BLE001
            log.exception("panel: fallo el comando %s", name)
            self.send({"t": "reply", "id": cid, "ok": False, "error": "error interno"})
