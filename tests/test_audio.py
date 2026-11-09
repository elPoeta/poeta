import threading
import time

import numpy as np
import pytest

from poeta.config import load_config
from poeta.voice.audio import (AudioConfig, AudioError, AudioManager, EnergyVAD,
                               Endpointer, level_db, to_float32)

RATE, FR = 16000, 480  # 30 ms
rng = np.random.default_rng(0)


def noise(db, frames=1):
    amp = 10 ** (db / 20) * 32768
    return [(rng.standard_normal(FR) * amp).astype(np.int16) for _ in range(frames)]


def feed_all(ep, frames):
    return [u for u in (ep.feed(f) for f in frames) if u is not None]


def test_level_db():
    assert abs(level_db(np.full(480, 16384, np.int16)) - (-6.02)) < 0.05
    assert level_db(np.zeros(480, np.int16)) == -120.0


def test_config_defaults_and_project():
    c = AudioConfig.from_config(load_config())
    assert c.strategy == "mute_during_playback" and c.sample_rate == 16000
    assert c.frame_samples == FR and c.end_silence_ms == 700


def test_config_validation():
    for bad in ({"strategy": "x"}, {"sample_rate": 44100}, {"frame_ms": 5},
                {"tail_ms": -1}, {"max_utterance_s": 0}):
        with pytest.raises(AudioError):
            AudioConfig(**bad)


def test_vad_adapts_and_detects():
    v = EnergyVAD()
    assert not any(v.is_speech(f) for f in noise(-60, 30))
    assert all(v.is_speech(f) for f in noise(-25, 5))
    assert not any(v.is_speech(f) for f in noise(-60, 5))  # vuelve a silencio


def test_vad_never_speech_below_floor_level():
    v = EnergyVAD(margin_db=12, min_level_db=-50)
    for f in noise(-80, 10):
        v.is_speech(f)
    assert not any(v.is_speech(f) for f in noise(-55, 5))  # +25 dB pero bajo -50


def test_endpointer_one_utterance_with_preroll():
    ep = Endpointer(AudioConfig())
    frames = noise(-60, 20) + noise(-25, 20) + noise(-60, 40)
    out = feed_all(ep, frames)
    assert len(out) == 1
    n = len(out[0]) // FR
    # preroll (<=10 frames) + 20 de voz + 24 de silencio de cierre (700 ms)
    assert 20 + 24 <= n <= 20 + 24 + 10
    assert not ep.speaking


def test_endpointer_short_noise_discarded():
    ep = Endpointer(AudioConfig())
    frames = noise(-60, 20) + noise(-25, 4) + noise(-60, 40)  # 120 ms < 250 ms
    assert feed_all(ep, frames) == []
    assert not ep.speaking


def test_endpointer_pause_inside_speech_does_not_split():
    ep = Endpointer(AudioConfig())
    frames = noise(-60, 20) + noise(-25, 15) + noise(-60, 10) + noise(-25, 15) + noise(-60, 40)
    assert len(feed_all(ep, frames)) == 1  # pausa de 300 ms < 700 ms


def test_endpointer_max_length_forces_close():
    ep = Endpointer(AudioConfig(max_utterance_s=1.0))
    out = feed_all(ep, noise(-60, 20) + noise(-25, 200))
    assert out and len(out[0]) <= 1.0 * RATE + 1


def test_to_float32():
    x = to_float32(np.array([0, 16384, -32768], np.int16))
    assert x.dtype == np.float32 and list(x) == [0.0, 0.5, -1.0]


# --- AudioManager con dispositivos falsos ---------------------------------
class FakeStream:
    def __init__(self, kind, **kw):
        self.kind, self.kw, self.cb = kind, kw, kw["callback"]
        self.started = self.closed = False

    def start(self): self.started = True
    def stop(self): self.started = False
    def close(self): self.closed = True


class FakeSD:
    def __init__(self, fail_in=False):
        self.streams, self.fail_in = [], fail_in

    def InputStream(self, **kw):
        if self.fail_in:
            raise OSError("no device")
        s = FakeStream("in", **kw); self.streams.append(s); return s

    def OutputStream(self, **kw):
        s = FakeStream("out", **kw); self.streams.append(s); return s


def push(am, frames):
    st = next(s for s in am._sd.streams if s.kind == "in")
    for f in frames:
        st.cb(f.reshape(-1, 1), FR, None, None)


def pull_out(am, frames=480):
    st = next(s for s in am._sd.streams if s.kind == "out")
    buf = np.zeros((frames, 1), np.float32)
    st.cb(buf, frames, None, None)
    return buf[:, 0]


def mgr(**kw):
    am = AudioManager(AudioConfig(**kw), sd=FakeSD())
    am.start()
    return am


def test_listen_returns_utterance():
    am = mgr()
    push(am, noise(-60, 20) + noise(-25, 20) + noise(-60, 40))
    u = am.listen(timeout=2)
    assert u is not None and u.dtype == np.int16 and len(u) > 20 * FR


def test_listen_timeout_and_cancel():
    am = mgr()
    t0 = time.monotonic()
    assert am.listen(timeout=0.2) is None and time.monotonic() - t0 < 1
    threading.Timer(0.2, am.cancel_listen).start()
    assert am.listen() is None


def test_listen_requires_start():
    with pytest.raises(AudioError):
        AudioManager(sd=FakeSD()).listen(timeout=0.1)


def test_input_open_failure_is_clear():
    with pytest.raises(AudioError, match="microfono"):
        AudioManager(sd=FakeSD(fail_in=True)).start()


def test_mic_ignored_while_speaking_and_tail():
    am = mgr(tail_ms=200)
    am.play(np.ones(4800, np.float32) * 0.1, 24000)
    assert am.speaking and am.muted
    push(am, noise(-60, 20) + noise(-25, 20) + noise(-60, 40))  # "eco"
    assert am.listen(timeout=0.3) is None
    out = pull_out(am, 4800)  # termina la reproduccion
    assert out[0] == pytest.approx(0.1) and not am.speaking
    assert am.muted  # cola de eco
    time.sleep(0.25)
    assert not am.muted
    am.flush_input()
    push(am, noise(-60, 20) + noise(-25, 20) + noise(-60, 40))
    assert am.listen(timeout=2) is not None


def test_strategy_none_never_mutes():
    am = mgr(strategy="none")
    am.play(np.ones(4800, np.float32), 24000)
    assert am.speaking and not am.muted
    push(am, noise(-60, 20) + noise(-25, 20) + noise(-60, 40))
    assert am.listen(timeout=2) is not None


def test_interrupt_stops_playback_and_silence_fills():
    am = mgr()
    am.play(np.ones(48000, np.float32), 24000)
    pull_out(am, 480)
    am.interrupt()
    assert not am.speaking
    assert not pull_out(am, 480).any()


def test_play_reuses_stream_and_rate_change():
    am = mgr()
    am.play(np.ones(100, np.float32), 24000)
    am.play(np.ones(100, np.float32), 24000)
    assert sum(s.kind == "out" for s in am._sd.streams) == 1
    with pytest.raises(AudioError):
        am.play(np.ones(10, np.float32), 22050)  # reproduciendo
    pull_out(am, 1000)
    am.play(np.ones(10, np.float32), 22050)
    assert sum(s.kind == "out" for s in am._sd.streams) == 2


def test_wait_and_speak_blocking():
    am = mgr()
    am.play(np.ones(480, np.float32), 24000)
    assert am.wait(timeout=0.05) is False
    threading.Timer(0.1, lambda: pull_out(am, 480)).start()
    assert am.wait(timeout=2) is True


def test_input_queue_overflow_counts():
    am = mgr()
    push(am, noise(-60, 250))
    assert am.overruns == 50 and am._q.qsize() == 200


def test_stop_closes_streams():
    am = mgr()
    am.play(np.ones(10, np.float32), 24000)
    am.stop()
    assert all(s.closed for s in am._sd.streams) and not am.speaking


def test_echo_captured_while_speaking_is_dropped_even_if_read_later():
    """Regresion: el silenciado debe decidirse al capturar, no al consumir."""
    am = mgr(tail_ms=100)
    am.play(np.ones(4800, np.float32) * 0.1, 24000)
    push(am, noise(-60, 20) + noise(-25, 20) + noise(-60, 40))  # eco mientras suena
    pull_out(am, 4800)  # termina la reproduccion
    time.sleep(0.2)  # pasa la cola: ya no esta silenciado cuando se lee la cola
    assert not am.muted
    assert am.listen(timeout=0.5) is None


def test_voice_after_playback_still_detected():
    am = mgr(tail_ms=50)
    am.play(np.ones(480, np.float32), 24000)
    pull_out(am, 480)
    time.sleep(0.1)
    push(am, noise(-60, 20) + noise(-25, 20) + noise(-60, 40))
    assert am.listen(timeout=2) is not None


def test_continuous_background_noise_eventually_closes_utterance():
    """Regresion: un sonido de fondo sostenido (TV) no debe mantener el enunciado abierto hasta el maximo."""
    ep = Endpointer(AudioConfig())  # ventana de 5 s
    out = []
    frames = noise(-80, 10) + noise(-40, 400)  # 12 s de "television" a -40 dBFS
    closed_at = None
    for i, f in enumerate(frames):
        u = ep.feed(f)
        if u is not None:
            out.append(u)
            closed_at = i
            break
    assert closed_at is not None and closed_at * 0.03 < 10  # cerro mucho antes de los 30 s
    # y la voz mas fuerte que el fondo se sigue detectando
    out2 = feed_all(ep, noise(-40, 40) + noise(-12, 20) + noise(-40, 40))
    assert len(out2) == 1


def test_vad_window_config():
    with pytest.raises(AudioError):
        AudioConfig(noise_window_s=0)
