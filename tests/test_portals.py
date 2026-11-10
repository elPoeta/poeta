"""Portales XDG contra un bus de sesion REAL (dbus-daemon) con un portal falso que habla el protocolo Request/Response."""

import asyncio
import os
import shutil
import struct
import subprocess
import tempfile
import zlib
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

pytest.importorskip("dbus_fast")
if not shutil.which("dbus-daemon"):
    pytest.skip("dbus-daemon no disponible", allow_module_level=True)

from dbus_fast import Message, MessageType, Variant  # noqa: E402
from dbus_fast.aio import MessageBus  # noqa: E402

from poeta.portals import PortalBus, PortalError, PortalsConfig, ShortcutService, register_screenshot  # noqa: E402
from poeta.portals.dbus import PORTAL_NAME, PORTAL_PATH  # noqa: E402
from poeta.portals.shortcuts import PortalsConfigError  # noqa: E402
from poeta.tools.base import ToolError  # noqa: E402

GS = "org.freedesktop.portal.GlobalShortcuts"


def run(c):
    return asyncio.run(c)


def make_png(path: Path, w=640, h=480):
    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d))
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + chunk(b"IEND", b""))


@pytest.fixture
def session_bus():
    d = tempfile.mkdtemp(prefix="pt-bus-", dir="/tmp")
    p = subprocess.Popen(["dbus-daemon", "--session", "--nofork", "--print-address=1", f"--address=unix:path={d}/bus"],
                         stdout=subprocess.PIPE, text=True)
    addr = p.stdout.readline().strip()
    yield addr, Path(d)
    p.terminate()
    p.wait(5)
    shutil.rmtree(d, ignore_errors=True)


class FakePortal:
    """Implementa lo minimo de xdg-desktop-portal. `behavior` permite simular rechazos/cancelaciones."""

    def __init__(self, addr, tmp):
        self.addr, self.tmp = addr, tmp
        self.calls: list[tuple[str, str, list]] = []
        self.behavior: dict[str, object] = {}
        self.bus = None
        self.session_path = None

    async def start(self):
        self.bus = await MessageBus(bus_address=self.addr).connect()
        self.bus.add_message_handler(self._handle)
        await self.bus.request_name(PORTAL_NAME)

    async def stop(self):
        self.bus.disconnect()

    def _request_path(self, msg, opts):
        sender = msg.sender[1:].replace(".", "_")
        return f"{PORTAL_PATH}/request/{sender}/{opts['handle_token'].value}"

    def _respond(self, path, code, results):
        async def later():
            await asyncio.sleep(0.02)
            self.bus.send(Message.new_signal(path, "org.freedesktop.portal.Request", "Response", "ua{sv}", [code, results]))
        asyncio.get_running_loop().create_task(later())

    def _handle(self, msg):
        if msg.message_type != MessageType.METHOD_CALL or msg.destination != PORTAL_NAME:
            return None
        self.calls.append((msg.interface, msg.member, list(msg.body)))
        if msg.interface == "org.freedesktop.host.portal.Registry":
            if self.behavior.get("no_registry"):
                self.bus.send(Message.new_error(msg, "org.freedesktop.DBus.Error.UnknownMethod", "no"))
            else:
                self.bus.send(Message.new_method_return(msg))
            return True
        if msg.interface == "org.freedesktop.portal.Session" and msg.member == "Close":
            self.bus.send(Message.new_method_return(msg))
            return True
        opts = msg.body[-1]
        path = self._request_path(msg, opts)
        if self.behavior.get("error"):
            self.bus.send(Message.new_error(msg, "org.freedesktop.portal.Error.Failed", "boom"))
            return True
        self.bus.send(Message.new_method_return(msg, "o", [path]))
        if msg.member == "CreateSession":
            self.session_path = f"{PORTAL_PATH}/session/{msg.sender[1:].replace('.', '_')}/{opts['session_handle_token'].value}"
            self._respond(path, 0, {"session_handle": Variant("s", self.session_path)})
        elif msg.member == "BindShortcuts":
            self._respond(path, int(self.behavior.get("bind_code", 0)), {})
        elif msg.member == "Screenshot":
            code = int(self.behavior.get("shot_code", 0))
            self._respond(path, code, {"uri": Variant("s", self.behavior.get("uri", ""))} if code == 0 else {})
        return True

    def activate(self, shortcut, session=None):
        self.bus.send(Message.new_signal(PORTAL_PATH, GS, "Activated", "osta{sv}",
                                         [session or self.session_path, shortcut, 123, {}]))


@pytest.fixture
def env(session_bus):
    addr, tmp = session_bus
    fake = FakePortal(addr, tmp)
    return fake, addr, tmp


async def with_portal(env, body):
    fake, addr, tmp = env
    await fake.start()
    portal = PortalBus(addr)
    try:
        return await body(fake, portal, tmp)
    finally:
        await portal.close()
        await fake.stop()


def cfg(**kw):
    return PortalsConfig(enabled=True, **kw)


# ---- configuracion ---------------------------------------------------------------------------------------------

def test_config_defaults_and_validation():
    c = PortalsConfig.from_config({})
    assert c.enabled is False and c.shortcuts["talk"][1] == "CTRL+ALT+SPACE" and c.screenshot.enabled
    ok = PortalsConfig.from_config({"portals": {"enabled": True, "shortcuts": {"talk": {"trigger": "CTRL+SHIFT+H"}}}})
    assert ok.shortcuts["talk"][1] == "CTRL+SHIFT+H" and ok.shortcuts["stop"][1] == ""
    for bad in ({"enabled": "si"}, {"app_id": "sinpunto"}, {"shortcuts": {"talk": {"trigger": "ctrl;rm"}}},
                {"shortcuts": {"talk": "x"}}, {"screenshot": {"timeout_s": 1}}, {"shortcuts": {"enabled": 1}}):
        with pytest.raises(PortalsConfigError):
            PortalsConfig.from_config({"portals": bad})


# ---- atajos ----------------------------------------------------------------------------------------------------

def test_shortcuts_bind_and_dispatch_only_for_own_session(env):
    async def body(fake, portal, tmp):
        hits = []
        svc = ShortcutService(portal, cfg(), {"talk": lambda: hits.append("talk"), "stop": lambda: hits.append("stop")})
        assert await svc.start() is True and svc.bound == ["talk", "stop"]
        iface, member, args = [c for c in fake.calls if c[1] == "BindShortcuts"][0]
        assert args[0] == fake.session_path
        ids = {s[0]: s[1] for s in args[1]}
        assert ids["talk"]["preferred_trigger"].value == "CTRL+ALT+SPACE" and "preferred_trigger" not in ids["stop"]
        assert any(c[1] == "Register" and c[2][0] == "org.poeta.Assistant" for c in fake.calls)
        fake.activate("talk")
        fake.activate("stop")
        fake.activate("talk", session="/otra/sesion")  # de otra sesion: se ignora
        fake.activate("desconocido")
        await asyncio.sleep(0.2)
        assert hits == ["talk", "stop"]
        await svc.stop()
        await asyncio.sleep(0.05)
        assert any(c[1] == "Close" for c in fake.calls)

    run(with_portal(env, body))


def test_shortcuts_survive_missing_registry_and_report_denial(env):
    async def body(fake, portal, tmp):
        fake.behavior["no_registry"] = True
        svc = ShortcutService(portal, cfg(), {"talk": lambda: None})
        assert await svc.start() is True
        svc2 = ShortcutService(portal, cfg(), {"talk": lambda: None})
        fake.behavior["bind_code"] = 1
        assert await svc2.start() is False and "no se autorizaron" in svc2.error
        fake.behavior["error"] = True
        svc3 = ShortcutService(portal, cfg(), {"talk": lambda: None})
        assert await svc3.start() is False and "boom" in svc3.error

    run(with_portal(env, body))


def test_no_portal_service_gives_clear_error(session_bus):
    addr, tmp = session_bus

    async def go():
        portal = PortalBus(addr)
        svc = ShortcutService(portal, cfg(), {"talk": lambda: None})
        ok = await svc.start()
        await portal.close()
        return ok, svc.error

    ok, err = run(go())
    assert ok is False and err


def test_unreachable_bus_raises_portal_error():
    async def go():
        with pytest.raises(PortalError, match="bus de sesion"):
            await PortalBus("unix:path=/tmp/no-existe-poeta.sock").connect()

    run(go())


# ---- captura ---------------------------------------------------------------------------------------------------

class Reg:
    def __init__(self):
        self.tools = {}

    def register(self, t):
        self.tools[t.name] = t


def test_screenshot_tool_success_and_errors(env):
    async def body(fake, portal, tmp):
        reg = Reg()
        assert register_screenshot(reg, portal, cfg()) == ["screenshot"]
        t = reg.tools["screenshot"]
        assert t.action == "screen_capture" and "no se envia" in t.describe({})[0]
        png = tmp / "Captura 1.png"
        make_png(png)
        fake.behavior["uri"] = "file://" + str(png).replace(" ", "%20")
        out = await t.run({})
        assert str(png) in out and "640x480" in out and "no se envio" in out
        assert fake.calls[-1][2][-1]["interactive"].value is False
        await t.run({"interactive": True})
        assert fake.calls[-1][2][-1]["interactive"].value is True
        fake.behavior["shot_code"] = 1
        with pytest.raises(ToolError, match="cancelada"):
            await t.run({})
        fake.behavior["shot_code"] = 2
        with pytest.raises(ToolError, match="no pudo"):
            await t.run({})
        fake.behavior.update(shot_code=0, uri="file:///no/existe/x.png")
        with pytest.raises(ToolError, match="no existe"):
            await t.run({})
        fake.behavior["uri"] = "https://evil.example/x.png"
        with pytest.raises(ToolError, match="no es un archivo local"):
            await t.run({})
        fake.behavior["uri"] = "file://otrohost/etc/passwd"
        with pytest.raises(ToolError, match="no es un archivo local"):
            await t.run({})
        fake.behavior["error"] = True
        with pytest.raises(ToolError, match="fallo"):
            await t.run({})

    run(with_portal(env, body))


def test_screenshot_not_registered_when_disabled():
    reg = Reg()
    assert register_screenshot(reg, PortalBus("x"), PortalsConfig(enabled=False)) == []
    from poeta.portals.shortcuts import ScreenshotSettings
    assert register_screenshot(reg, PortalBus("x"), PortalsConfig(enabled=True, screenshot=ScreenshotSettings(False))) == []
    assert reg.tools == {}


def test_request_times_out_cleanly(env):
    async def body(fake, portal, tmp):
        fake.behavior["silent"] = True
        # el portal falso siempre responde; forzamos el plazo con un timeout diminuto
        with pytest.raises(PortalError, match="no respondio"):
            await portal.request("org.freedesktop.portal.Screenshot", "NoResponde", "s", [""], timeout=0.0)

    run(with_portal(env, body))
