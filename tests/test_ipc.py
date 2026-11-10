import asyncio
import json
import os
import shutil
import stat
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from poeta.core import EventBus
from poeta.gestures import make_announcing_confirmer
from poeta.ipc import IpcClient, IpcError, UiServer, build_handlers, decode, encode
from poeta.ipc import server as server_mod
from poeta.ipc.protocol import MAX_LINE, ProtocolError
from poeta.models.router import ProviderDisabled
from poeta.security import ActionRequest


@pytest.fixture
def sockdir():
    d = Path(tempfile.mkdtemp(prefix="pt-", dir="/tmp"))  # AF_UNIX limita la ruta a ~108 caracteres
    yield d
    shutil.rmtree(d, ignore_errors=True)


def run(coro):
    return asyncio.run(coro)


class FakeRouter:
    def __init__(self):
        self.p, self.m = "local", "bolt"

    def status(self):
        return {"provider": self.p, "model": self.m, "cloud": self.p == "nvidia", "providers": ["local", "nvidia"]}

    def select(self, provider=None, model=None):
        if provider and provider not in ("local", "nvidia"):
            raise ProviderDisabled(f"Proveedor '{provider}' no existe")
        self.p, self.m = provider or self.p, model or self.m


class FakeMemory:
    def list_conversations(self, limit=20):
        return [{"id": 1, "title": "hola"}][:limit]

    def get_messages(self, cid, limit=None):
        return [{"role": "user", "content": "texto"}]

    def list_memories(self, kind=None, limit=50):
        return [{"id": 1, "kind": kind}]

    def tool_history(self, limit=20):
        return []

    def get_preferences(self):
        return {"tono": "breve"}

    def stats(self):
        return {"messages": 3}


def make(sockdir, memory=True, voice=None):
    bus = EventBus()
    agent = NS(reasoning=NS(mode="auto"), incognito=False, memory=FakeMemory() if memory else None,
               tools=NS(tools=lambda: [NS(name="notify", source="desktop", action="desktop_notify", untrusted=False)]),
               mcp=NS(status=lambda: [NS(name="ddg", state="conectado", detail="", tools=[1, 2, 3])]))
    router = FakeRouter()
    snap, handlers = build_handlers(router, agent, bus, None, voice, NS(active=True), NS(last={"rss_mb": 5.0}))
    srv = UiServer(bus, snap, handlers, sockdir / "ui.sock")
    return bus, agent, router, srv


async def connected(srv, path):
    await srv.start()
    c = IpcClient(path)
    await c.connect()
    return c


def test_socket_is_private_and_removed_on_stop(sockdir):
    async def go():
        bus, agent, router, srv = make(sockdir)
        await srv.start()
        st = (sockdir / "ui.sock").stat()
        await srv.stop()
        return st, (sockdir / "ui.sock").exists()

    st, exists = run(go())
    assert stat.S_ISSOCK(st.st_mode) and stat.S_IMODE(st.st_mode) == 0o600 and not exists


def test_directory_must_be_private_and_ours(sockdir):
    os.chmod(sockdir, 0o755)
    bus, agent, router, srv = make(sockdir)
    with pytest.raises(IpcError, match="privado"):
        run(srv.start())


def test_stale_socket_replaced_live_one_refused_regular_file_untouched(sockdir):
    async def go():
        bus, agent, router, srv = make(sockdir)
        await srv.start()
        # otra instancia mientras la primera vive
        _, _, _, srv2 = make(sockdir)
        with pytest.raises(IpcError, match="otra instancia"):
            await srv2.start()
        # simula un proceso que murio sin limpiar: el socket queda pero nadie escucha
        srv._server.close()
        await srv._server.wait_closed()
        _, _, _, srv3 = make(sockdir)
        await srv3.start()
        await srv3.stop()
        await srv.stop()

    run(go())
    (sockdir / "ui.sock").write_text("no soy un socket")
    _, _, _, srv4 = make(sockdir)
    with pytest.raises(IpcError, match="no es un socket"):
        run(srv4.start())
    assert (sockdir / "ui.sock").read_text() == "no soy un socket"


def test_hello_snapshot_events_and_commands(sockdir):
    async def go():
        bus, agent, router, srv = make(sockdir)
        c = await connected(srv, sockdir / "ui.sock")
        snap = c.snapshot
        bus.publish("state", state="speaking")
        ev = await c.recv()
        pong = await c.command("ping")
        status = await c.command("select_model", provider="nvidia", model="nemotron")
        mode = await c.command("set_reasoning", mode="off")
        errs = []
        for name, args in (("nada", {}), ("select_model", {"provider": "xx"}), ("set_reasoning", {"mode": "tal vez"}),
                           ("query", {"kind": "drop table"}), ("query", {"kind": "messages", "conversation_id": "1"}),
                           ("select_model", {})):
            try:
                await c.command(name, **args)
            except RuntimeError as exc:
                errs.append(str(exc))
        convs = await c.command("query", kind="conversations", limit=5)
        msgs = await c.command("query", kind="messages", conversation_id=1)
        prefs = await c.command("query", kind="preferences")
        again = await c.command("get_snapshot")
        await c.close()
        await srv.stop()
        return snap, ev, pong, status, mode, errs, convs, msgs, prefs, again, agent, router

    snap, ev, pong, status, mode, errs, convs, msgs, prefs, again, agent, router = run(go())
    assert snap["provider"] == "local" and snap["tools"][0]["name"] == "notify" and snap["gestures_active"]
    assert snap["mcp"] == [{"name": "ddg", "state": "conectado", "detail": "", "tools": 3}] and snap["reasoning"] == "auto"
    assert ev["kind"] == "event" or ev["t"] == "event"
    assert ev["data"] == {"state": "speaking"} and pong == "pong"
    assert status["provider"] == "nvidia" and status["cloud"] and mode == "off" and agent.reasoning.mode == "off"
    assert len(errs) == 6 and "desconocido" in errs[0] and "no existe" in errs[1] and "mode" in errs[2]
    assert convs == [{"id": 1, "title": "hola"}] and msgs[0]["content"] == "texto" and prefs == {"tono": "breve"}
    assert again["provider"] == "nvidia" and again["reasoning"] == "off" and router.p == "nvidia"


def test_memory_disabled_and_interrupt(sockdir):
    async def go():
        voice = NS(interrupt=lambda: True)
        bus, agent, router, srv = make(sockdir, memory=False, voice=voice)
        c = await connected(srv, sockdir / "ui.sock")
        assert await c.command("interrupt") is True
        with pytest.raises(RuntimeError, match="deshabilitada"):
            await c.command("query", kind="conversations")
        assert await c.command("query", kind="audit") == []
        await c.close()
        await srv.stop()

    run(go())


def test_garbage_and_oversize_do_not_crash_the_server(sockdir):
    async def go():
        bus, agent, router, srv = make(sockdir)
        await srv.start()
        r, w = await asyncio.open_unix_connection(str(sockdir / "ui.sock"))
        await r.readline()
        await r.readline()
        w.write(b"{no es json\n")
        w.write(b'["lista"]\n')
        w.write(json.dumps({"t": "otra"}).encode() + b"\n")
        await w.drain()
        replies = [json.loads(await r.readline()) for _ in range(3)]
        w.write(b"x" * (MAX_LINE + 10) + b"\n")  # supera el limite: se corta SOLO esta conexion
        await w.drain()
        assert await asyncio.wait_for(r.read(), 3) is not None
        w.close()
        c = IpcClient(sockdir / "ui.sock")
        await c.connect()  # el servidor sigue vivo
        assert await c.command("ping") == "pong"
        await c.close()
        await srv.stop()
        return replies

    replies = run(go())
    assert all(not rp["ok"] for rp in replies)


def test_foreign_uid_is_rejected(sockdir, monkeypatch):
    async def go():
        bus, agent, router, srv = make(sockdir)
        await srv.start()
        monkeypatch.setattr(server_mod, "peer_uid", lambda s: os.getuid() + 1)
        r, w = await asyncio.open_unix_connection(str(sockdir / "ui.sock"))
        data = await asyncio.wait_for(r.read(), 2)
        w.close()
        await asyncio.gather(w.wait_closed(), return_exceptions=True)
        await srv.stop()
        return data, srv.clients

    data, clients = run(go())
    assert data == b"" and clients == 0  # cerrado sin enviar ni siquiera el snapshot


def test_peer_uid_reads_real_credentials(sockdir):
    async def go():
        got = []
        def handler(r, w):
            got.append(server_mod.peer_uid(w.get_extra_info("socket")))
            w.close()

        srv = await asyncio.start_unix_server(handler, path=str(sockdir / "x.sock"))
        _, w = await asyncio.open_unix_connection(str(sockdir / "x.sock"))
        await asyncio.sleep(0.1)
        w.close()
        await asyncio.gather(w.wait_closed(), return_exceptions=True)
        srv.close()
        await srv.wait_closed()
        return got

    assert run(go()) == [os.getuid()]


def test_slow_client_is_dropped_without_blocking_others(sockdir, monkeypatch):
    monkeypatch.setattr(server_mod, "QUEUE_SIZE", 20)

    async def go():
        bus, agent, router, srv = make(sockdir)
        await srv.start()
        slow_r, slow_w = await asyncio.open_unix_connection(str(sockdir / "ui.sock"))  # nunca lee
        good = IpcClient(sockdir / "ui.sock")
        await good.connect()
        await asyncio.sleep(0.1)
        assert srv.clients == 2
        big = "x" * 60_000
        for i in range(400):
            bus.publish("resources", pad=big, i=i)
            await asyncio.sleep(0)
            if i % 20 == 0:
                while True:  # el cliente sano va leyendo
                    try:
                        await good.recv(0.001)
                    except (asyncio.TimeoutError, TimeoutError):
                        break
        await asyncio.sleep(0.2)
        n = srv.clients
        pong = await good.command("ping")
        await good.close()
        slow_w.close()
        await asyncio.gather(slow_w.wait_closed(), return_exceptions=True)
        await srv.stop()
        return n, pong

    n, pong = run(go())
    assert n == 1 and pong == "pong"


def test_events_from_other_threads_reach_clients(sockdir):
    async def go():
        bus, agent, router, srv = make(sockdir)
        c = await connected(srv, sockdir / "ui.sock")
        threading.Thread(target=lambda: bus.publish("gesture", name="palm")).start()
        ev = await c.recv()
        await c.close()
        await srv.stop()
        return ev

    assert run(go())["data"] == {"name": "palm"}


def test_max_clients(sockdir, monkeypatch):
    monkeypatch.setattr(server_mod, "MAX_CLIENTS", 2)

    async def go():
        bus, agent, router, srv = make(sockdir)
        await srv.start()
        cs = [IpcClient(sockdir / "ui.sock") for _ in range(2)]
        for c in cs:
            await c.connect()
        r, w = await asyncio.open_unix_connection(str(sockdir / "ui.sock"))
        refused = await asyncio.wait_for(r.read(), 2)
        w.close()
        await asyncio.gather(w.wait_closed(), return_exceptions=True)
        for c in cs:
            await c.close()
        await srv.stop()
        return refused

    assert run(go()) == b""


def test_handler_timeout_and_internal_error_do_not_leak_details(sockdir):
    async def slow(args):
        await asyncio.sleep(30)

    async def boom(args):
        raise RuntimeError("clave secreta sk-123")

    async def go():
        bus, agent, router, srv = make(sockdir)
        srv.handlers.update(slow=slow, boom=boom)
        monkey = server_mod.asyncio.wait_for

        async def fast_wait(aw, timeout):  # no esperar 10 s reales
            return await monkey(aw, 0.2 if timeout == 10.0 else timeout)

        server_mod.asyncio.wait_for = fast_wait
        try:
            c = await connected(srv, sockdir / "ui.sock")
            out = []
            for n in ("slow", "boom"):
                try:
                    await c.command(n)
                except RuntimeError as exc:
                    out.append(str(exc))
            await c.close()
            await srv.stop()
        finally:
            server_mod.asyncio.wait_for = monkey
        return out

    out = run(go())
    assert out == ["tiempo agotado", "error interno"]


def test_protocol_roundtrip_and_limits():
    assert decode(encode({"t": "x", "a": "ñ"})) == {"t": "x", "a": "ñ"}
    for bad in (b"no", b"[]", b'{"sin_t": 1}', b'{"t": 5}', b"\xff\xfe"):
        with pytest.raises(ProtocolError):
            decode(bad)
    with pytest.raises(ProtocolError):
        encode({"t": "x", "p": "y" * MAX_LINE})


def test_announcing_confirmer_publishes_only_kind_and_tool_never_details():
    bus, seen = EventBus(), []
    bus.subscribe(seen.append, kinds=("permission",))

    async def base(req, explicit):
        return True

    req = ActionRequest("window_control", "Cerrar la ventana 'Banco - clave 1234'", {"tool": "window_action", "title": "Banco"})

    async def go():
        return await make_announcing_confirmer(base, bus)(req, False)

    assert run(go()) is True
    assert [e.data["phase"] for e in seen] == ["request", "decided"] and seen[1].data["allowed"] is True
    blob = json.dumps([e.data for e in seen])
    assert "Banco" not in json.dumps({k: v for k, v in seen[0].data.items() if k != "description"})  # details: nunca
    assert "window_action" in blob and '"title"' not in blob
    # la descripcion corta SI sale (para que panel y HUD digan que se pide), solo en la solicitud y acotada
    assert seen[0].data["description"].startswith("Cerrar la ventana") and "description" not in seen[1].data
    long_req = ActionRequest("window_control", "x " * 200, {"tool": "t"})
    asyncio.run(make_announcing_confirmer(base, bus)(long_req, False))
    assert len(seen[2].data["description"]) <= 120
