import asyncio
import threading
import time
from types import SimpleNamespace as NS

import numpy as np
import pytest

from poeta.core import EventBus
from poeta.gestures import make_announcing_confirmer
from poeta.security import ActionRequest
from poeta.voice.confirm import (VoiceConfirmConfig, VoiceConfirmer, classify, make_headless_base,
                                 make_voice_confirmer, normalize)
from poeta.voice.tts import SpeechPipeline


def run(c):
    return asyncio.run(c)


def req(kind="desktop_control", tainted=False, desc="cerrar la ventana de Firefox"):
    return ActionRequest(kind=kind, description=desc, details={"tool": "window_action"}, tainted=tainted)


# ---- clasificacion -------------------------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("Sí.", True), ("dale", True), ("¡Autorizo!", True), ("De acuerdo", True), (" OK ", True),
    ("no", False), ("No, gracias.", False), ("cancelar", False), ("mejor no", False),
    ("sí pero no", None), ("si claro hacelo ahora mismo", None), ("no sé", None), ("", None), ("...", None),
    ("el sí es una palabra", None), ("nose", None),
])
def test_classify_requires_whole_utterance(text, expected):
    assert classify(text) is expected


def test_explicit_needs_exact_phrase_and_plain_yes_is_not_enough():
    assert classify("Confirmo la acción.", "confirmo la accion") is True
    assert classify("sí", "confirmo la accion") is None
    assert classify("confirmo", "confirmo la accion") is None
    assert classify("no", "confirmo la accion") is False
    assert normalize("  ¿Sí,   claro? ") == "si claro"


# ---- configuracion y elegibilidad -----------------------------------------------------------------------------

class FakeAudio:
    def __init__(self):
        self.holds = 0
        self.spoken = []
        self.cancelled = 0
        self.script = []  # enunciados que devolvera listen (None = silencio)
        self.listen_calls = 0
        self.held_during_listen = []
        self.pipe = None

    def hold_mute(self):
        self.holds += 1

    def release_mute(self):
        self.holds -= 1

    def speak(self, samples, rate):
        self.spoken.append(samples)

    def flush_input(self):
        pass

    def cancel_listen(self):
        self.cancelled += 1

    def listen(self, timeout=None):
        self.listen_calls += 1
        self.held_during_listen.append(self.holds)
        if not self.script:
            time.sleep(min(timeout or 0.05, 0.05))
            return None
        return self.script.pop(0)

    def interrupt(self):
        pass

    def wait(self, timeout=None):
        return True


class FakeEngine:
    sample_rate = 24000

    def __init__(self):
        self.said = []

    def synthesize(self, text):
        self.said.append(text)
        return np.zeros(10, dtype=np.float32)

    def close(self):
        pass


UTT = np.ones(1600, dtype=np.int16)


class FakeStt:
    def __init__(self, heard):
        self.heard = list(heard)

    def transcribe(self, audio):
        return NS(text=self.heard.pop(0))


def make_vc(heard=(), script=(), **cfg):
    audio, engine = FakeAudio(), FakeEngine()
    audio.script = list(script)
    pipe = SpeechPipeline(engine, audio)
    c = VoiceConfirmConfig(enabled=True, **{"timeout_s": 3, "retries": 1, **cfg})
    vc = VoiceConfirmer(c, out=lambda *a, **k: None, audit=NS(record=lambda ev, **f: vc.records.append((ev, f))))
    vc.records = []
    vc.attach(audio, FakeStt(heard), pipe)
    return vc, audio, engine


def test_from_config_defaults_off():
    assert VoiceConfirmConfig.from_config({"voice": {}}).enabled is False
    c = VoiceConfirmConfig.from_config({"voice": {"confirm": {"enabled": True, "retries": 0}}})
    assert c.enabled and c.retries == 0 and c.timeout_s == 20.0


def test_eligibility_rules():
    vc, *_ = make_vc()
    assert vc.eligible(req(), False)
    assert not vc.eligible(req("cloud_transfer"), False)  # nunca a la nube por voz
    assert not vc.eligible(req(), True)  # explicitas: no por defecto
    assert not vc.eligible(req(tainted=True), False)  # tras datos no confiables: no por defecto
    vc2, *_ = make_vc(allow_explicit=True, allow_tainted=True)
    assert vc2.eligible(req(), True) and vc2.eligible(req(tainted=True), False)
    assert not vc2.eligible(req("cloud_transfer"), True)
    off = VoiceConfirmer(VoiceConfirmConfig(enabled=False))
    off.attach(1, 2, 3)
    assert not off.eligible(req(), False)
    unattached = VoiceConfirmer(VoiceConfirmConfig(enabled=True))
    assert not unattached.eligible(req(), False)


# ---- ask ------------------------------------------------------------------------------------------------------

def test_ask_yes_speaks_the_action_and_audits_without_text():
    vc, audio, engine = make_vc(heard=["Sí."], script=[UTT])
    assert run(vc.ask(req(), False)) is True
    assert "cerrar la ventana de Firefox" in engine.said[0] and "si o no" in engine.said[0]
    assert vc.records == [("voice_confirmation", {"kind": "desktop_control", "tool": "window_action",
                                                  "allowed": True, "attempts": 1})]
    assert audio.cancelled >= 1


def test_ask_no_and_retry_on_unclear_then_yes():
    vc, audio, engine = make_vc(heard=["no", ], script=[UTT])
    assert run(vc.ask(req(), False)) is False
    vc, audio, engine = make_vc(heard=["quizas", "dale"], script=[UTT, UTT], retries=2)
    assert run(vc.ask(req(), False)) is True
    assert any("No te entendi" in t for t in engine.said)


def test_ask_gives_up_without_valid_answer_and_never_decides():
    vc, audio, engine = make_vc(heard=["uno", "dos"], script=[UTT, UTT], retries=1)
    assert run(vc.ask(req(), False)) is None
    vc, audio, engine = make_vc(heard=[], script=[], timeout_s=3)
    vc.config = VoiceConfirmConfig(enabled=True, timeout_s=3, retries=1)
    assert run(asyncio.wait_for(vc.ask(req(), False), 6)) is None  # silencio total: vence el plazo
    assert vc.records == []


def test_explicit_prompt_asks_exact_phrase():
    vc, audio, engine = make_vc(heard=["sí", "confirmo la accion"], script=[UTT, UTT], allow_explicit=True, retries=2)
    assert run(vc.ask(req(), True)) is True
    assert "confirmo la accion" in engine.said[0]


def test_mic_is_opened_only_while_listening_and_hold_is_restored():
    vc, audio, engine = make_vc(heard=["si"], script=[UTT])
    vc.pipe.begin()  # el turno de habla mantiene el micro silenciado
    assert audio.holds == 1
    assert run(vc.ask(req(), False)) is True
    assert audio.held_during_listen == [0]  # al escuchar el micro estaba abierto
    assert audio.holds == 1  # y se restituyo para el resto del turno
    vc.pipe.end()
    vc.pipe.wait(2)
    assert audio.holds == 0


def test_open_mic_does_not_rehold_if_turn_ended_meanwhile():
    audio, engine = FakeAudio(), FakeEngine()
    pipe = SpeechPipeline(engine, audio)
    pipe.begin()
    with pipe.open_mic():
        pipe.end()
        assert pipe.wait(2)
    assert audio.holds == 0


# ---- combinacion con teclado/gestos ----------------------------------------------------------------------------

def test_voice_wins_over_pending_base_and_cancels_it():
    vc, audio, _ = make_vc(heard=["si"], script=[UTT])
    cancelled = []

    async def base(r, e):
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            cancelled.append(True)
            raise

    assert run(make_voice_confirmer(base, vc)(req(), False)) is True
    assert cancelled == [True]


def test_base_wins_and_stops_listening():
    vc, audio, _ = make_vc(script=[])

    async def base(r, e):
        await asyncio.sleep(0.2)
        return True

    assert run(make_voice_confirmer(base, vc)(req(), False)) is True
    assert audio.cancelled >= 1


def test_voice_giving_up_leaves_decision_to_base():
    vc, audio, _ = make_vc(heard=["xx", "yy"], script=[UTT, UTT], retries=1)

    async def base(r, e):
        await asyncio.sleep(0.5)
        return False

    assert run(make_voice_confirmer(base, vc)(req(), False)) is False


def test_not_eligible_goes_straight_to_base():
    vc, audio, _ = make_vc()
    calls = []

    async def base(r, e):
        calls.append(e)
        return True

    conf = make_voice_confirmer(base, vc)
    assert run(conf(req("cloud_transfer"), False)) is True and calls == [False]
    assert audio.listen_calls == 0 and not audio.spoken


def test_headless_base_waits_then_denies_and_voice_can_answer_first():
    out = []
    slow = make_headless_base(0.2, out=out.append)
    assert run(slow(req(), False)) is False and "se deniega" in out[0]
    vc, audio, _ = make_vc(heard=["si"], script=[UTT])
    assert run(make_voice_confirmer(make_headless_base(5), vc)(req(), False)) is True


def test_announcing_confirmer_reports_channels_and_description():
    bus = EventBus()

    async def base(r, e):
        return True

    async def go():
        conf = make_announcing_confirmer(base, bus, lambda r, e: ["voz", "gestos"])
        return await conf(req(tainted=True), False)

    assert run(go()) is True
    first, last = [ev.data for ev in bus.recent("permission")]
    assert first["phase"] == "request" and first["via"] == ["voz", "gestos"] and first["tainted"] is True
    assert first["description"] == "cerrar la ventana de Firefox" and first["tool"] == "window_action"
    assert last["phase"] == "decided" and last["allowed"] is True
