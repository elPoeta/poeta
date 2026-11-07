"""Servidor falso compatible con OpenAI (SSE) para probar proveedores, router y agente sin red real."""

from __future__ import annotations

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest


def sse(obj) -> bytes:
    return f"data: {json.dumps(obj)}\n\n".encode()


def delta(**kw):
    return {"choices": [{"index": 0, "delta": kw, "finish_reason": None}]}


def finish(reason: str):
    return {"choices": [{"index": 0, "delta": {}, "finish_reason": reason}]}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # silenciar
        pass

    def _json(self, status: int, obj) -> None:
        data = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self.server.requests.append({"method": "GET", "path": self.path, "headers": dict(self.headers)})
        if self.path.endswith("/models"):
            self._json(200, {"data": [{"id": "fake-model"}]})
        else:
            self._json(404, {"error": "no encontrado"})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        self.server.requests.append(
            {"method": "POST", "path": self.path, "headers": dict(self.headers), "body": body}
        )
        model = body.get("model")
        if model == "boom":
            self._json(500, {"error": "fallo interno"})
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        w = self.wfile.write
        if model == "tools":
            w(sse(delta(tool_calls=[{"index": 0, "id": "call_1", "function": {"name": "buscar", "arguments": ""}}])))
            w(sse(delta(tool_calls=[{"index": 0, "function": {"arguments": '{"q": "ho'}}])))
            w(sse(delta(tool_calls=[{"index": 0, "function": {"arguments": 'la"}'}}])))
            w(sse(finish("tool_calls")))
        elif model == "thinking":
            w(sse(delta(reasoning_content="Pienso...")))
            w(sse(delta(content="Respuesta.")))
            w(sse(finish("stop")))
        else:
            for piece in ("Hola", ", ", "mundo", "!"):
                w(sse(delta(content=piece)))
            w(sse(finish("stop")))
        w(b"data: [DONE]\n\n")


class FakeLLM(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), Handler)
        self.requests: list[dict] = []

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}/v1"

    def posts(self) -> list[dict]:
        return [r for r in self.requests if r["method"] == "POST"]


def _start():
    server = FakeLLM()
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()
    return server


@pytest.fixture
def fake_llm():
    server = _start()
    yield server
    server.shutdown()
    server.server_close()


@pytest.fixture
def fake_cloud():
    server = _start()
    yield server
    server.shutdown()
    server.server_close()


@pytest.fixture
def dead_url() -> str:
    """URL de un puerto en el que no escucha nadie."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    return f"http://127.0.0.1:{port}/v1"
