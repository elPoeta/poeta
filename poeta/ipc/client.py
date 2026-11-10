"""Cliente asyncio del protocolo del panel (para pruebas, scripts y diagnostico)."""

from __future__ import annotations

import asyncio
import itertools
from pathlib import Path
from typing import Any, AsyncIterator

from .protocol import MAX_LINE, decode, encode
from .server import default_socket_path


class IpcClient:
    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path else default_socket_path()
        self._ids = itertools.count(1)
        self._r: asyncio.StreamReader | None = None
        self._w: asyncio.StreamWriter | None = None
        self.snapshot: dict[str, Any] | None = None

    async def connect(self, timeout: float = 3.0) -> None:
        self._r, self._w = await asyncio.wait_for(asyncio.open_unix_connection(str(self.path), limit=MAX_LINE), timeout)
        hello = await self.recv()
        if hello.get("t") != "hello":
            raise ConnectionError("el servidor no respondio con 'hello'")
        snap = await self.recv()
        self.snapshot = snap.get("data") if snap.get("t") == "snapshot" else None

    async def recv(self, timeout: float | None = 5.0) -> dict[str, Any]:
        assert self._r is not None
        line = await asyncio.wait_for(self._r.readline(), timeout)
        if not line:
            raise ConnectionError("conexion cerrada")
        return decode(line)

    async def command(self, name: str, **args: Any) -> Any:
        assert self._w is not None
        cid = next(self._ids)
        self._w.write(encode({"t": "cmd", "id": cid, "name": name, "args": args}))
        await self._w.drain()
        while True:
            msg = await self.recv()
            if msg.get("t") == "reply" and msg.get("id") == cid:
                if not msg.get("ok"):
                    raise RuntimeError(msg.get("error", "error"))
                return msg.get("data")

    async def events(self) -> AsyncIterator[dict[str, Any]]:
        while True:
            msg = await self.recv(None)
            if msg.get("t") == "event":
                yield msg

    async def close(self) -> None:
        if self._w:
            self._w.close()
            await asyncio.gather(self._w.wait_closed(), return_exceptions=True)
