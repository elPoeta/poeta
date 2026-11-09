import asyncio
import time
from types import SimpleNamespace

import numpy as np
import pytest

from poeta.core.agent import AgentEvent
from poeta.voice.audio import AudioConfig, AudioManager
from poeta.voice.loop import LineQueue, VoiceLoop, is_noise_transcript
from poeta.voice.tts import SpeechPipeline
from tests.test_tts import FakeEngine, FakeSD


class FakeAudio:
    """Microfono falso: entrega enunciados guiados; delega mute/play en un AudioManager real."""

    def __init__(self, utts):
        self.utts, self.flushed, self.cancelled = list(utts), 0, 0
        self.real = AudioManager(AudioConfig(tail_ms=20), sd=FakeSD())
        self.real.start()

    def listen(self, timeout=None):
        if self.utts:
            return self.utts.pop(0)
        time.sleep(min(timeout or 0.05, 0.05))
        return None

    def flush_input(self): self.flushed += 1
    def cancel_listen(self): self.cancelled += 1
    def __getattr__(self, n): return getattr(self.real, n)


class FakeStt:
    def __init__(self, texts):
        self.texts, self.calls = list(texts), []

    def transcribe(self, audio):
        self.calls.append(audio)
        return SimpleNamespace(text=self.texts.pop(0))


class FakeAgent:
    def __init__(self, replies, delay=0.0, fail=None):
        self.replies, self.delay, self.fail, self.heard, self.cancelled = list(replies), delay, fail, [], False

    async def respond(self, text):
        self.heard.append(text)
        if self.fail:
            raise self.fail
        try:
            yield AgentEvent("reasoning", text="pienso")
            for piece in self.replies.pop(0):
                if self.delay:
                    await asyncio.sleep(self.delay)
                yield AgentEvent("text", text=piece)
            yield AgentEvent("done")
        except asyncio.CancelledError:
            self.cancelled = True
            raise


UTT = np.ones(1600, np.int16)


def build(utts=(), texts=(), replies=(), mode="ptt", delay=0.0, fail=None, engine_delay=0.0):
    audio = FakeAudio(utts)
    pipe = SpeechPipeline(FakeEngine(delay=engine_delay), audio.real)
    out, cmds = [], []

    def say(*a, **k):
        out.append("".join(str(x) for x in a))

    async def on_command(line):
        cmds.append(line)
        return line == "/quit"

    async def mk():
        keys = LineQueue()
        agent = FakeAgent(replies, delay, fail)
        loop = VoiceLoop(audio=audio, stt=FakeStt(texts), pipeline=pipe, agent=agent, keys=keys,
                         mode=mode, on_command=on_command, out=say, listen_timeout=0.2)
        return keys, agent, loop

    return audio, pipe, out, cmds, mk


def drain(audio):
    while audio.real.speaking:
        audio.real._on_output(np.zeros((4800, 1), np.float32), 4800, None, None)


def test_noise_transcripts():
    assert is_noise_transcript("") and is_noise_transcript(" . ")
    assert is_noise_transcript("Subtítulos realizados por la comunidad de Amara.org")
    assert is_noise_transcript("¡Gracias por ver el video!")
    assert not is_noise_transcript("abrí el navegador")


def test_ptt_voice_turn_end_to_end():
    async def go():
        audio, pipe, out, cmds, mk = build(utts=[UTT], texts=["hola poeta"],
                                           replies=[["Hola, ¿cómo estás? ", "Todo bien por acá."]])
        keys, agent, loop = await mk()
        t = asyncio.create_task(loop.run())
        keys.put("")  # Enter: hablar
        await asyncio.sleep(0.6)
        drain(audio)
        await asyncio.sleep(0.2)
        keys.put("/quit")
        assert await asyncio.wait_for(t, 3) == 0
        return audio, pipe, out, agent

    audio, pipe, out, agent = asyncio.run(go())
    text = "".join(out)
    assert agent.heard == ["hola poeta"] and "Tu: hola poeta" in text and "POETA: Hola, ¿cómo estás?" in text
    assert pipe.engine.texts[0] == "Hola, ¿cómo estás?"  # reasoning no se habla
    assert "pienso" not in text and audio.flushed == 1
    assert "stt " in text and "1er audio" in text and "razonando desde" in text


def test_typed_text_and_commands_route():
    async def go():
        audio, pipe, out, cmds, mk = build(replies=[["Entendido."]])
        keys, agent, loop = await mk()
        t = asyncio.create_task(loop.run())
        keys.put("/status")
        keys.put("qué hora es")
        await asyncio.sleep(0.4)
        drain(audio)
        keys.put("/quit")
        await asyncio.wait_for(t, 3)
        return cmds, agent

    cmds, agent = asyncio.run(go())
    assert cmds == ["/status", "/quit"] and agent.heard == ["qué hora es"]


def test_noise_transcript_is_not_sent_to_agent():
    async def go():
        audio, pipe, out, cmds, mk = build(utts=[UTT], texts=["Gracias por ver el video"])
        keys, agent, loop = await mk()
        t = asyncio.create_task(loop.run())
        keys.put("")
        await asyncio.sleep(0.3)
        keys.put("/quit")
        await asyncio.wait_for(t, 3)
        return agent, out

    agent, out = asyncio.run(go())
    assert agent.heard == [] and any("ignorado" in o for o in out)


def test_no_speech_timeout_message():
    async def go():
        audio, pipe, out, cmds, mk = build()
        keys, agent, loop = await mk()
        t = asyncio.create_task(loop.run())
        keys.put("")
        await asyncio.sleep(0.5)
        keys.put("/quit")
        await asyncio.wait_for(t, 3)
        return agent, out

    agent, out = asyncio.run(go())
    assert agent.heard == [] and any("no te escuche" in o for o in out)


def test_enter_interrupts_generation_and_speech():
    async def go():
        audio, pipe, out, cmds, mk = build(replies=[["Una frase larga para decir. "] * 20], delay=0.05)
        keys, agent, loop = await mk()
        t = asyncio.create_task(loop.run())
        keys.put("cuéntame algo largo")
        await asyncio.sleep(0.4)
        keys.put("")  # Enter durante la respuesta
        await asyncio.sleep(0.3)
        keys.put("/quit")
        await asyncio.wait_for(t, 3)
        return audio, pipe, out, agent

    audio, pipe, out, agent = asyncio.run(go())
    assert agent.cancelled and any("interrumpido" in o for o in out)
    assert not audio.real.speaking
    time.sleep(0.1)
    assert not audio.real.muted  # el microfono se reabre


def test_enter_interrupts_speech_after_generation_finished():
    async def go():
        audio, pipe, out, cmds, mk = build(replies=[["Respuesta corta lista ya."]])
        keys, agent, loop = await mk()
        t = asyncio.create_task(loop.run())
        keys.put("hola")
        await asyncio.sleep(0.4)  # LLM termino; el "altavoz" no consume: sigue hablando
        assert audio.real.speaking
        keys.put("")
        await asyncio.sleep(0.3)
        assert not audio.real.speaking
        keys.put("/quit")
        await asyncio.wait_for(t, 3)
        return out

    assert any("interrumpido" in o for o in asyncio.run(go()))


def test_agent_error_is_reported_and_loop_continues():
    async def go():
        audio, pipe, out, cmds, mk = build(fail=RuntimeError("servidor caido"))
        keys, agent, loop = await mk()
        t = asyncio.create_task(loop.run())
        keys.put("hola")
        await asyncio.sleep(0.3)
        keys.put("/quit")
        assert await asyncio.wait_for(t, 3) == 0
        return out, audio

    out, audio = asyncio.run(go())
    assert any("servidor caido" in o for o in out)
    time.sleep(0.1)
    assert not audio.real.muted


def test_always_mode_listens_without_enter():
    async def go():
        audio, pipe, out, cmds, mk = build(utts=[UTT], texts=["hola poeta"], replies=[["Hola."]], mode="always")
        keys, agent, loop = await mk()
        t = asyncio.create_task(loop.run())
        await asyncio.sleep(0.5)
        drain(audio)
        await asyncio.sleep(0.2)
        keys.put("/quit")
        await asyncio.wait_for(t, 3)
        return agent

    assert asyncio.run(go()).heard == ["hola poeta"]


def test_invalid_mode():
    async def go():
        with pytest.raises(ValueError):
            VoiceLoop(audio=None, stt=None, pipeline=None, agent=None, keys=LineQueue(), mode="x")

    asyncio.run(go())


# --- confirmaciones de permisos durante el modo voz ---------------------------------
from poeta.interface import make_cli_confirmer  # noqa: E402
from poeta.security import ActionRequest  # noqa: E402


class AskingAgent(FakeAgent):
    """Antes de responder pide autorizacion (como el router con un proveedor cloud)."""

    def __init__(self, confirmer, replies):
        super().__init__(replies)
        self.confirmer, self.decision = confirmer, None

    async def respond(self, text):
        self.heard.append(text)
        req = ActionRequest("cloud_transfer", "enviar a NVIDIA", payload={"messages": [{"role": "user", "content": text}]})
        self.decision = await self.confirmer(req, False)
        if not self.decision:
            raise RuntimeError("no autorizado")
        for piece in self.replies.pop(0):
            yield AgentEvent("text", text=piece)
        yield AgentEvent("done")


def run_with_confirmation(answer, replies=(("Respuesta autorizada por ti.",),)):
    async def go():
        audio, pipe, out, cmds, mk = build()
        keys = LineQueue()
        agent = AskingAgent(make_cli_confirmer(keys.ask), [list(r) for r in replies])
        loop = VoiceLoop(audio=audio, stt=FakeStt([]), pipeline=pipe, agent=agent, keys=keys,
                         on_command=lambda l: asyncio.sleep(0, result=l == "/quit"), out=lambda *a, **k: out.append("".join(map(str, a))),
                         listen_timeout=0.2)
        t = asyncio.create_task(loop.run())
        keys.put("hola desde nvidia")
        await asyncio.sleep(0.3)
        keys.put(answer)  # respuesta a la confirmacion
        await asyncio.sleep(0.4)
        drain(audio)  # el altavoz termina de decir la respuesta
        await asyncio.sleep(0.2)
        keys.put("/quit")
        await asyncio.wait_for(t, 3)
        return agent, pipe, out

    return asyncio.run(go())


def test_confirmation_answer_does_not_interrupt_the_turn(capsys):
    agent, pipe, out = run_with_confirmation("s")
    assert agent.decision is True
    assert not any("interrumpido" in o for o in out)
    assert pipe.engine.texts == ["Respuesta autorizada por ti."]  # se hablo la respuesta


def test_denied_confirmation_ends_turn_cleanly_and_loop_continues(capsys):
    agent, pipe, out = run_with_confirmation("")  # Enter = cancelar
    assert agent.decision is False
    assert any("no autorizado" in o for o in out) and pipe.engine.texts == []
    assert not any("interrumpido" in o for o in out)


def test_answer_is_not_left_in_the_queue_for_the_next_turn(capsys):
    async def go():
        keys = LineQueue()
        t = asyncio.create_task(keys.ask("? "))
        await asyncio.sleep(0.05)
        keys.put("s")
        assert await asyncio.wait_for(t, 1) == "s"
        await asyncio.sleep(0.05)
        return keys.get_nowait()

    assert asyncio.run(go()) is None
