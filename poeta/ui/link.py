"""Conexion del panel con el agente: QLocalSocket + JSON-lines, con reconexion automatica."""

from __future__ import annotations

import itertools
import json
from typing import Any, Callable

from PySide6.QtCore import QObject, QTimer, Signal
from PySide6.QtNetwork import QLocalSocket

MAX_BUFFER = 4 * 1024 * 1024


class AgentLink(QObject):
    connected = Signal()
    disconnected = Signal()
    snapshot = Signal(dict)
    event = Signal(str, dict)

    def __init__(self, path: str, parent: QObject | None = None, retry_ms: int = 2000):
        super().__init__(parent)
        self.path = path
        self._buf = b""
        self._ids = itertools.count(1)
        self._pending: dict[int, Callable[[bool, Any], None]] = {}
        self._up = False
        self.sock = QLocalSocket(self)
        self.sock.readyRead.connect(self._on_ready)
        self.sock.disconnected.connect(self._on_down)
        self.sock.errorOccurred.connect(lambda *_: self._on_down())
        self.timer = QTimer(self)
        self.timer.setInterval(retry_ms)
        self.timer.timeout.connect(self._try)

    @property
    def is_up(self) -> bool:
        return self._up

    def start(self) -> None:
        self._try()
        self.timer.start()

    def stop(self) -> None:
        self.timer.stop()
        self.sock.abort()

    def _try(self) -> None:
        if self.sock.state() == QLocalSocket.LocalSocketState.UnconnectedState:
            self.sock.connectToServer(self.path)

    def _on_down(self) -> None:
        was = self._up
        self._up = False
        self._buf = b""
        pend, self._pending = self._pending, {}
        for cb in pend.values():
            cb(False, "desconectado")
        if was:
            self.disconnected.emit()

    def _on_ready(self) -> None:
        self._buf += bytes(self.sock.readAll())
        if len(self._buf) > MAX_BUFFER:
            self.sock.abort()
            return
        while b"\n" in self._buf:
            line, self._buf = self._buf.split(b"\n", 1)
            try:
                msg = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if isinstance(msg, dict):
                self._dispatch(msg)

    def _dispatch(self, msg: dict[str, Any]) -> None:
        t = msg.get("t")
        if t == "hello":
            self._up = True
            self.connected.emit()
        elif t == "snapshot" and isinstance(msg.get("data"), dict):
            self.snapshot.emit(msg["data"])
        elif t == "event":
            self.event.emit(str(msg.get("kind")), msg.get("data") if isinstance(msg.get("data"), dict) else {})
        elif t == "reply":
            cb = self._pending.pop(msg.get("id"), None)
            if cb:
                cb(bool(msg.get("ok")), msg.get("data") if msg.get("ok") else msg.get("error", "error"))

    def command(self, name: str, callback: Callable[[bool, Any], None] | None = None, **args: Any) -> None:
        if not self._up:
            if callback:
                callback(False, "desconectado")
            return
        cid = next(self._ids)
        if callback:
            self._pending[cid] = callback
        line = json.dumps({"t": "cmd", "id": cid, "name": name, "args": args}, ensure_ascii=False)
        self.sock.write(line.encode("utf-8") + b"\n")
