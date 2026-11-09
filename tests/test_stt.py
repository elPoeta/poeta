import pytest

from poeta.config import load_config
from poeta.voice.metrics import normalize, wer
from poeta.voice.stt import SttConfig, SttError, Transcriber


class Seg:
    def __init__(self, s, e, t):
        self.start, self.end, self.text = s, e, t


class Info:
    language = "es"
    duration = 3.0


class FakeModel:
    def __init__(self):
        self.calls = []

    def transcribe(self, audio, **kw):
        self.calls.append((audio, kw))
        return iter([Seg(0, 1, " Hola "), Seg(1, 2, "mundo")]), Info()


def make(cfg=None):
    holder = {"n": 0, "m": FakeModel()}

    def factory(c):
        holder["n"] += 1
        return holder["m"]

    return Transcriber(cfg or SttConfig(device="cpu"), factory), holder


def test_defaults_match_vozboard():
    c = SttConfig()
    assert (c.model, c.device, c.beam_size, c.vad_filter, c.min_silence_duration_ms) == (
        "large-v3", "cuda", 5, True, 500)
    assert c.effective_compute_type == "float16"
    assert SttConfig(device="cpu").effective_compute_type == "int8"
    assert SttConfig(device="cpu", compute_type="int8_float16").effective_compute_type == "int8_float16"


def test_from_project_config(monkeypatch):
    monkeypatch.setenv("AI_MODELS", "/data/models")
    c = SttConfig.from_config(load_config())
    assert (c.model, c.device, c.effective_compute_type, c.language) == ("large-v3-turbo", "auto", "int8_float16", "es")
    assert c.download_root == "/data/models/whisper"


def test_invalid_config_rejected():
    with pytest.raises(SttError):
        SttConfig(device="tpu")
    with pytest.raises(SttError):
        SttConfig(beam_size=0)
    with pytest.raises(SttError):
        SttConfig.from_config({"voice": {"stt": {"engine": "otro"}}})


def test_lazy_load_once_and_params(tmp_path):
    tr, h = make()
    assert not tr.loaded and h["n"] == 0
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"x")
    r = tr.transcribe(wav)
    tr.transcribe(wav)
    assert h["n"] == 1 and tr.loaded
    assert r.text == "Hola mundo" and r.language == "es" and r.duration == 3.0
    audio, kw = h["m"].calls[0]
    assert kw["language"] == "es" and kw["beam_size"] == 5 and kw["vad_filter"] is True
    assert kw["vad_parameters"] == {"min_silence_duration_ms": 500}
    assert "initial_prompt" not in kw


def test_language_override_and_autodetect(tmp_path):
    tr, h = make()
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"x")
    tr.transcribe(wav, language=None)
    tr.transcribe(wav, language="en", initial_prompt="POETA")
    assert h["m"].calls[0][1]["language"] is None
    assert h["m"].calls[1][1]["language"] == "en"
    assert h["m"].calls[1][1]["initial_prompt"] == "POETA"


def test_array_input_accepted():
    tr, h = make()
    arr = object()
    tr.transcribe(arr)
    assert h["m"].calls[0][0] is arr


def test_missing_file():
    tr, h = make()
    with pytest.raises(FileNotFoundError):
        tr.transcribe("/no/existe.wav")
    assert h["n"] == 0


def test_load_failure_is_clear():
    def boom(c):
        raise OSError("libcublas.so.12 is not found")

    tr = Transcriber(SttConfig(), boom)
    with pytest.raises(SttError, match="libcublas"):
        tr.transcribe(object())


def test_transcribe_failure_wrapped():
    class Bad:
        def transcribe(self, *a, **k):
            raise RuntimeError("cuda oom")

    tr = Transcriber(SttConfig(device="cpu"), lambda c: Bad())
    with pytest.raises(SttError, match="cuda oom"):
        tr.transcribe(object())


def test_wer():
    assert normalize("¡Hola, Cómo estás!") == ["hola", "como", "estas"]
    assert wer("hola mundo", "Hola, mundo.") == 0.0
    assert wer("uno dos tres cuatro", "uno tres cuatro") == 0.25
    assert wer("a b", "x y") == 1.0
    assert wer("", "") == 0.0 and wer("", "algo") == 1.0


def _write_wav(path, rate, ch, samples):
    import struct
    import wave

    with wave.open(str(path), "wb") as w:
        w.setnchannels(ch)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(struct.pack("<%dh" % len(samples), *samples))


def test_wav_decoded_without_pyav(tmp_path):
    from poeta.voice.stt import load_wav

    p = tmp_path / "a.wav"
    _write_wav(p, 16000, 1, [0, 16384, -16384, 32767])
    x = load_wav(p)
    assert x.dtype.name == "float32" and len(x) == 4
    assert abs(x[1] - 0.5) < 1e-3 and abs(x[2] + 0.5) < 1e-3

    st = tmp_path / "s.wav"  # estereo 32 kHz -> mono 16 kHz
    _write_wav(st, 32000, 2, [1000, 3000] * 100)
    y = load_wav(st)
    assert len(y) == 50 and abs(y[0] - 2000 / 32768) < 1e-3

    bad = tmp_path / "b.wav"
    bad.write_bytes(b"no es wav")
    assert load_wav(bad) is None


def test_transcribe_passes_array_for_wav(tmp_path):
    tr, h = make()
    p = tmp_path / "a.wav"
    _write_wav(p, 16000, 1, [0] * 1600)
    tr.transcribe(p)
    assert not isinstance(h["m"].calls[0][0], str)
    m = tmp_path / "a.mp3"
    m.write_bytes(b"x")
    tr.transcribe(m)
    assert isinstance(h["m"].calls[1][0], str)


def test_warmup_runs_encoder_without_vad():
    tr, h = make()
    tr.warmup()
    audio, kw = h["m"].calls[0]
    assert kw["vad_filter"] is False and len(audio) == 32000


def test_resolve_device_auto_and_cpu_precision():
    from poeta.voice.stt import resolve_device

    c = SttConfig(device="auto", compute_type="int8_float16")
    assert resolve_device(c, 1) == ("cuda", "int8_float16")
    assert resolve_device(c, 0) == ("cpu", "int8")  # sin GPU: no pedir precision de GPU
    assert resolve_device(SttConfig(device="cpu", compute_type="float16"), 0) == ("cpu", "int8")
    assert resolve_device(SttConfig(device="cpu", compute_type="int8_float32"), 0) == ("cpu", "int8_float32")
    assert resolve_device(SttConfig(device="cuda", compute_type="float16"), 0) == ("cuda", "float16")
    assert resolve_device(SttConfig(device="auto"), 0) == ("cpu", "int8")
