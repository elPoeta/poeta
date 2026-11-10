import asyncio
from types import SimpleNamespace as NS

import main as poeta_main
from poeta.config import load_config
from poeta.core import EventBus
from poeta.gestures import GestureHub, GesturePolicy, make_announcing_confirmer
from poeta.portals import PortalsConfig
from poeta.security import ActionRequest
from poeta.voice.confirm import VoiceConfirmConfig, VoiceConfirmer, make_headless_base, make_voice_confirmer
from poeta.gestures import make_gesture_confirmer


class FakePortal:
    def __init__(self, fail=False):
        self.fail, self.signals, self.calls = fail, [], []

    async def call(self, *a, **k):
        self.calls.append(a)
        return []

    async def request(self, iface, method, sig, args, options=None, timeout=30):
        self.calls.append((iface, method))
        if self.fail:
            from poeta.portals import PortalError
            raise PortalError("sin portal")
        return 0, {"session_handle": "/s/1"}

    async def on_signal(self, iface, member, cb):
        self.cb = cb


def test_start_shortcuts_wires_talk_to_enter_and_stop_to_interrupt():
    cfg = load_config()
    cfg["portals"] = {"enabled": True}
    keys, interrupts = [], []
    portal = FakePortal()

    async def go():
        svc = await poeta_main.start_shortcuts(cfg, portal, NS(put=keys.append), NS(interrupt=lambda: interrupts.append(1)))
        portal.cb("/s/1", "talk", 1, {})
        portal.cb("/s/1", "stop", 2, {})
        portal.cb("/otra", "talk", 3, {})
        return svc

    svc = asyncio.run(go())
    assert svc.bound == ["talk", "stop"] and keys == [""] and interrupts == [1]


def test_start_shortcuts_disabled_or_failing_does_not_break_voice(capsys):
    cfg = load_config()
    assert asyncio.run(poeta_main.start_shortcuts(cfg, FakePortal(), None, None)) is None  # portals.enabled=false
    cfg["portals"] = {"enabled": True}
    assert asyncio.run(poeta_main.start_shortcuts(cfg, None, None, None)) is None
    assert asyncio.run(poeta_main.start_shortcuts(cfg, FakePortal(fail=True), NS(put=print), NS(interrupt=print))) is None
    assert "Atajo global desactivado" in capsys.readouterr().err


def test_default_config_has_portals_off_and_voice_confirm_off():
    cfg = load_config()
    assert PortalsConfig.from_config(cfg).enabled is False
    assert VoiceConfirmConfig.from_config(cfg).enabled is False


def test_headless_gesture_confirmation_is_not_denied_instantly():
    """Regresion: en modo servicio la base denegaba al instante y los gestos nunca llegaban a contestar."""
    bus = EventBus()
    hub = GestureHub(GesturePolicy(), bus, out=lambda *a, **k: None)
    hub.active = True
    r = ActionRequest(kind="window_control", description="cerrar ventana", details={"tool": "window_action"})

    async def go():
        conf = make_gesture_confirmer(make_headless_base(5, out=lambda *a: None), hub)
        t = asyncio.ensure_future(conf(r, False))
        await asyncio.sleep(0.1)
        assert not t.done() and hub.waiting
        hub.on_event(NS(name="thumb_up"))
        return await asyncio.wait_for(t, 2)

    assert asyncio.run(go()) is True
