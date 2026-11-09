import threading
import time

import numpy as np
import pytest

from poeta.config import load_config
from poeta.voice.audio import AudioConfig, AudioManager
from poeta.voice.tts import (KokoroEngine, SentenceSplitter, SpeechPipeline, TtsConfig,
                             TtsError, clean_for_speech, make_engine)


def split(chunks, **kw):
    sp = SentenceSplitter(**kw)
    out = []
    for c in chunks:
        out += sp.feed(c)
    return out + sp.flush()


def test_clean_for_speech():
    t = "## Título\n- **negrita** y `código` mira https://x.com/a?b=1 😀 [doc](http://z.io)"
    assert clean_for_speech(t) == "Título negrita y código mira un enlace doc"
    assert clean_for_speech("snake_case y _énfasis_") == "snake_case y énfasis"


def test_first_sentence_leaves_immediately_and_streaming_is_split_invariant():
    sp = SentenceSplitter()
    assert sp.feed("Listo.") == []  # aun no se sabe si sigue (falta espacio)
    assert sp.feed(" Abrí") == ["Listo."]
    text = "Hola, ¿cómo estás? Estoy muy bien, gracias por preguntar. Y vos qué tal andás hoy?"
    whole = split([text])
    chars = split(list(text))
    assert whole == chars and len(whole) == 3


def test_abbreviations_decimals_and_lists_do_not_split():
    out = split(["El Sr. Pérez pesa 2.5 kilos, etc. y vive en la Av. Mitre 123. Fin del mensaje largo aquí."])
    assert out == ["El Sr. Pérez pesa 2.5 kilos, etc. y vive en la Av. Mitre 123.", "Fin del mensaje largo aquí."]
    out = split(["1. Primer paso del proceso largo aquí.\n2. Segundo paso del proceso largo aquí."])
    assert out == ["Primer paso del proceso largo aquí.", "Segundo paso del proceso largo aquí."]


def test_short_following_sentences_are_merged_but_first_is_not():
    out = split(["Sí. Claro. Perfecto. Lo hago ahora mismo con mucho gusto."])
    assert out[0] == "Sí."
    assert out[1].startswith("Claro. Perfecto.")


def test_code_blocks_are_not_spoken_even_when_fence_is_split_across_chunks():
    chunks = ["Mira este ejemplo largo para ti: ``", "`python\nprint('hola. mundo')\n``", "`\nEso es todo el ejemplo, gracias."]
    joined = " ".join(split(chunks))
    assert "print" not in joined and "hola. mundo" not in joined
    assert "Eso es todo el ejemplo" in joined and "Mira este ejemplo" in joined


def test_unterminated_code_block_is_dropped():
    assert "secreto" not in " ".join(split(["Texto largo de introducción aquí. ```\nsecreto = 1"]))


def test_long_text_without_punctuation_is_cut_at_comma():
    text = ", ".join(["palabra%d adicional" % i for i in range(40)])
    out = split([text], max_chars=100)
    assert len(out) > 1 and all(len(s) <= 110 for s in out)
    assert " ".join(out).replace(",", "") == text.replace(",", "")


def test_symbols_only_are_ignored():
    assert split(["...", " --- ", "🙂"]) == []


# --- config / motores ---------------------------------------------------
def test_tts_config_from_project(monkeypatch):
    monkeypatch.setenv("AI_MODELS", "/m")
    c = TtsConfig.from_config(load_config())
    assert c.engine == "kokoro" and c.model_path == "/m/kokoro/kokoro-v1.0.onnx"
    assert c.voices_path == "/m/kokoro/voices-v1.0.bin" and c.voice is None


def test_unsupported_engine_and_missing_voice(tmp_path):
    with pytest.raises(TtsError):
        TtsConfig.from_config({"voice": {"tts": {"engine": "qwen3"}}})
    with pytest.raises(TtsError, match="voice"):
        KokoroEngine(TtsConfig(engine="kokoro"))
    with pytest.raises(TtsError, match="Piper"):
        make_engine(TtsConfig(engine="piper", model_path=str(tmp_path / "no.onnx")))


# --- pipeline -------------------------------------------------------------
class FakeEngine:
    sample_rate = 24000

    def __init__(self, delay=0.0, fail_on=None):
        self.texts, self.delay, self.fail_on, self.closed = [], delay, fail_on, False

    def synthesize(self, text):
        if self.delay:
            time.sleep(self.delay)
        if self.fail_on and self.fail_on in text:
            raise TtsError("boom")
        self.texts.append(text)
        return np.full(2400, 0.1, np.float32)

    def close(self):
        self.closed = True


class FakeStream:
    def __init__(self, **kw):
        self.cb = kw["callback"]

    def start(self): pass
    def stop(self): pass
    def close(self): pass


class FakeSD:
    def InputStream(self, **kw): return FakeStream(**kw)
    def OutputStream(self, **kw): return FakeStream(**kw)


def audio():
    a = AudioManager(AudioConfig(tail_ms=50), sd=FakeSD())
    a.start()
    return a


def drain(a):
    """Simula al altavoz consumiendo todo lo encolado."""
    while a.speaking:
        a._on_output(np.zeros((4800, 1), np.float32), 4800, None, None)


def test_pipeline_speaks_sentences_in_order_and_releases_mute():
    a, e = audio(), FakeEngine()
    p = SpeechPipeline(e, a)
    p.begin()
    p.feed("Hola, ¿cómo estás? Muy bien. ")
    p.feed("Gracias por preguntar hoy mismo.")
    p.end()
    time.sleep(0.3)
    assert e.texts[0] == "Hola, ¿cómo estás?" and len(e.texts) == 2
    assert a.muted  # sigue silenciado hasta que termine de sonar
    threading.Thread(target=lambda: (time.sleep(0.1), drain(a))).start()
    assert p.wait(3)
    time.sleep(0.12)
    assert not a.muted and p.first_audio_s is not None


def test_mic_stays_muted_in_gap_between_sentences():
    a, e = audio(), FakeEngine(delay=0.4)  # sintesis mas lenta que la cola de eco (50 ms)
    p = SpeechPipeline(e, a)
    p.begin()
    p.feed("Primera frase bastante larga para sonar. Segunda frase bastante larga para sonar.")
    time.sleep(0.5)
    drain(a)  # termina la primera
    time.sleep(0.2)  # hueco > tail_ms, la segunda aun se sintetiza
    assert a.muted
    p.stop()


def test_stop_interrupts_and_discards_pending():
    a, e = audio(), FakeEngine(delay=0.15)
    p = SpeechPipeline(e, a)
    p.begin()
    p.feed("Una frase larga para decir ahora. Otra frase larga para decir luego. Y otra más larga.")
    time.sleep(0.2)
    p.stop()
    assert not a.speaking and p.wait(0.1)
    n = len(e.texts)
    time.sleep(0.5)
    assert len(e.texts) <= n + 1 and not a.speaking  # a lo sumo la que estaba en curso, sin sonar
    time.sleep(0.1)
    assert not a.muted


def test_begin_cancels_previous_turn():
    a, e = audio(), FakeEngine(delay=0.1)
    p = SpeechPipeline(e, a)
    p.begin()
    p.feed("Respuesta vieja que no debe terminar de decirse. Segunda parte de la vieja respuesta.")
    p.begin()
    p.feed("Respuesta nueva. ")
    p.end()
    time.sleep(0.6)
    assert "Respuesta nueva." in e.texts
    assert not any("Segunda parte" in t for t in e.texts)


def test_synthesis_error_skips_sentence_and_continues():
    errs = []
    a, e = audio(), FakeEngine(fail_on="rota")
    p = SpeechPipeline(e, a, on_error=errs.append)
    p.begin()
    p.feed("Frase rota que falla al sintetizar. Frase buena que sigue funcionando bien.")
    p.end()
    time.sleep(0.3)
    assert p.errors == 1 and len(errs) == 1 and e.texts == ["Frase buena que sigue funcionando bien."]


def test_close_closes_engine():
    a, e = audio(), FakeEngine()
    SpeechPipeline(e, a).close()
    assert e.closed
