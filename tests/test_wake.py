import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

from poeta.config import load_config
from poeta.core.agent import AgentEvent
from poeta.voice.loop import LineQueue, VoiceLoop
from poeta.voice.wake import TranscriptWake, WakeConfig, find_wake, make_chime
from tests.test_voice_loop import UTT, FakeAgent, FakeAudio, FakeStt, build
from tests.test_tts import FakeEngine

N = ["poeta", "boeta", "poetas"]


@pytest.mark.parametrize("text, rest", [
    ("Poeta, qué hora es", "qué hora es"),
    ("hola Poeta abrí el navegador", "abrí el navegador"),
    ("Poeta.", ""),
    ("POETA", ""),
    ("Boeta, dime algo", "dime algo"),
    ("Póeta: abre el correo", "abre el correo"),
])
def test_wake_found(text, rest):
    assert find_wake(text, N) == rest


@pytest.mark.parametrize("text", [
    "Hola, ¿cómo estás?", "la poesía es bonita", "poesia", "abrí el navegador por favor poeta",  # lejos del inicio
    "", "el perro come carne",
])
def test_wake_not_found(text):
    assert find_wake(text, N) is None


def test_threshold_controls_strictness():
    assert find_wake("poetaz abre", N, threshold=0.8) == "abre"
    assert find_wake("poetaz abre", N, threshold=1.0) is None


def test_config_from_project_and_validation():
    c = WakeConfig.from_config(load_config())
    assert c.phrase == "poeta" and c.engine == "transcript" and c.stt_device == "cpu" and "boeta" in c.names
    with pytest.raises(ValueError):
        WakeConfig.from_config({"voice": {"wake_word": {"engine": "openwakeword"}}})
    with pytest.raises(ValueError):
        WakeConfig.from_config({"voice": {"wake_word": {"threshold": 0.2}}})


def test_chime_is_short_and_quiet():
    c = make_chime(24000)
    assert c.dtype == np.float32 and 0.15 < len(c) / 24000 < 0.4 and np.abs(c).max() <= 0.3


# --- bucle en modo wake --------------------------------------------------------
class SpyAudio(FakeAudio):
    def __init__(self, utts):
        super().__init__(utts)
        self.chimes = 0

    def speak(self, samples, rate):
        self.chimes += 1


def wake_loop(utts, wake_texts, main_texts, replies):
    audio = SpyAudio(utts)
    pipe = __import__("poeta.voice.tts", fromlist=["SpeechPipeline"]).SpeechPipeline(FakeEngine(), audio.real)
    out = []
    wake_stt, main_stt = FakeStt(wake_texts), FakeStt(main_texts)
    cfg = WakeConfig()

    async def go(script):
        keys = LineQueue()
        agent = FakeAgent(replies)
        loop = VoiceLoop(audio=audio, stt=main_stt, pipeline=pipe, agent=agent, keys=keys, mode="wake",
                         wake=TranscriptWake(cfg, wake_stt), followup_s=0.3, chime=(make_chime(), 24000),
                         on_command=lambda l: asyncio.sleep(0, result=l == "/quit"),
                         out=lambda *a, **k: out.append("".join(map(str, a))), listen_timeout=0.2)
        t = asyncio.create_task(loop.run())
        await script(keys, audio)
        keys.put("/quit")
        await asyncio.wait_for(t, 3)
        return agent

    return go, audio, pipe, wake_stt, main_stt, out


def test_utterance_without_wake_word_is_discarded_everywhere():
    go, audio, pipe, wake_stt, main_stt, out = wake_loop([UTT], ["hola, estoy hablando con otra persona"], [], [])

    async def script(keys, audio):
        await asyncio.sleep(0.4)

    agent = asyncio.run(go(script))
    assert len(wake_stt.calls) == 1
    assert main_stt.calls == []  # ni siquiera pasa por el modelo principal
    assert agent.heard == [] and pipe.engine.texts == []
    assert not any("hablando con otra persona" in o for o in out)  # tampoco se imprime


def test_wake_with_command_in_same_utterance():
    go, audio, pipe, wake_stt, main_stt, out = wake_loop(
        [UTT], ["poeta qué hora es"], ["Poeta, ¿qué hora es?"], [["Son las diez."]])

    async def script(keys, audio):
        await asyncio.sleep(0.5)
        while audio.real.speaking:
            audio.real._on_output(np.zeros((4800, 1), np.float32), 4800, None, None)
        await asyncio.sleep(0.2)

    agent = asyncio.run(go(script))
    assert agent.heard == ["¿qué hora es?"]  # sin la palabra de activacion, con el texto del modelo principal
    assert audio.chimes == 0


def test_wake_alone_chimes_then_takes_next_utterance():
    go, audio, pipe, wake_stt, main_stt, out = wake_loop(
        [UTT, UTT], ["poeta"], ["Poeta.", "abrí el navegador"], [["Listo, abierto."]])

    async def script(keys, audio):
        await asyncio.sleep(0.6)
        while audio.real.speaking:
            audio.real._on_output(np.zeros((4800, 1), np.float32), 4800, None, None)
        await asyncio.sleep(0.2)

    agent = asyncio.run(go(script))
    assert audio.chimes == 1 and agent.heard == ["abrí el navegador"]
    assert any("te escucho" in o for o in out)


def test_wake_alone_then_silence_returns_to_waiting():
    go, audio, pipe, wake_stt, main_stt, out = wake_loop([UTT], ["poeta"], ["Poeta."], [])

    async def script(keys, audio):
        await asyncio.sleep(0.8)

    agent = asyncio.run(go(script))
    assert agent.heard == [] and audio.chimes == 1 and any("no te escuche" in o for o in out)


def test_wake_mode_requires_detector():
    async def go():
        with pytest.raises(ValueError):
            VoiceLoop(audio=None, stt=None, pipeline=None, agent=None, keys=LineQueue(), mode="wake")

    asyncio.run(go())
