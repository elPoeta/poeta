"""Sintesis de voz: motores intercambiables y habla por frases.

- clean_for_speech / SentenceSplitter: convierten el texto en streaming del LLM en
  frases pronunciables (sin markdown, codigo ni URLs), cortando donde suena natural.
- KokoroEngine (kokoro-onnx) y PiperEngine (binario `piper`, audio crudo por tuberia:
  el texto nunca toca el disco).
- SpeechPipeline: sintetiza frase a frase en un hilo y las encola en el AudioManager,
  de modo que POETA empieza a hablar antes de que el LLM termine de generar.
"""

from __future__ import annotations

import contextlib
import json
import queue
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np


class TtsError(RuntimeError):
    pass


class TTSEngine(Protocol):
    sample_rate: int

    def synthesize(self, text: str) -> np.ndarray:
        """float32 mono en [-1, 1] a `sample_rate`."""

    def close(self) -> None: ...


# --------------------------------------------------------------------------
# Texto -> frases
# --------------------------------------------------------------------------
_ABBREV = {"sr", "sra", "srta", "dr", "dra", "ud", "uds", "etc", "vs", "av", "pag", "pág",
           "núm", "num", "ej", "tel", "ing", "lic", "prof", "cap", "fig", "aprox", "min", "máx"}
_URL = re.compile(r"https?://\S+|www\.\S+")
_EMOJI = re.compile("[\U0001F000-\U0001FFFF☀-➿️‍]")
_END = re.compile(r"""([.!?…]+)["')\]»”’]*(?=\s)|\n+""")


def clean_for_speech(text: str) -> str:
    """Quita lo que no se debe leer en voz alta (markdown, URLs, emojis)."""
    t = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", text)  # [texto](url) -> texto
    t = _URL.sub("un enlace", t)
    t = t.replace("`", "")
    t = re.sub(r"^[ \t]{0,3}#{1,6}[ \t]*", "", t, flags=re.M)
    t = re.sub(r"^[ \t]*[-*+•][ \t]+", "", t, flags=re.M)
    t = re.sub(r"^[ \t]*\d+[.)][ \t]+", "", t, flags=re.M)
    t = re.sub(r"(\*\*|__|\*|~~)", "", t)
    t = re.sub(r"(?<!\w)_|_(?!\w)", "", t)
    t = _EMOJI.sub("", t)
    return re.sub(r"\s+", " ", t).strip()


def _is_boundary(buf: str, m: re.Match[str]) -> bool:
    if m.group(0).startswith("\n"):
        return True
    if not m.group(1).endswith("."):
        return True  # ? ! …
    before = buf[: m.start()]
    if re.search(r"(?:^|\n)[ \t]*\d+$", before):
        return False  # marcador de lista numerada ("1. ")
    word = re.search(r"([^\W\d_]+)$", before)
    if word:
        w = word.group(1)
        if w.lower() in _ABBREV or (len(w) == 1 and w.isupper()):
            return False  # "Sr. ", "etc. ", "J. "
    return True


class SentenceSplitter:
    """feed(delta) -> frases completas; flush() al terminar el turno.

    - La primera frase sale en cuanto esta completa (latencia minima).
    - Las siguientes se unen si son muy cortas (< min_chars) para no sonar entrecortado.
    - Las frases demasiado largas se cortan en una coma/punto y coma (max_chars).
    - Los bloques de codigo (```...```) no se leen.
    """

    def __init__(self, min_chars: int = 28, max_chars: int = 220):
        self.min_chars, self.max_chars = min_chars, max_chars
        self._buf = ""
        self._in_code = False
        self._fence_tail = ""  # 1-2 acentos graves que podrian ser inicio de ```
        self._held = ""
        self._emitted = 0

    def _filter_code(self, delta: str) -> str:
        s, self._fence_tail, out, i = self._fence_tail + delta, "", [], 0
        while i < len(s):
            if s.startswith("```", i):
                self._in_code = not self._in_code
                out.append("\n")
                i += 3
            elif s[i] == "`" and "```".startswith(s[i:]) and len(s) - i < 3:
                self._fence_tail = s[i:]
                break
            else:
                if not self._in_code:
                    out.append(s[i])
                i += 1
        return "".join(out)

    def _emit(self, raw: str, out: list[str]) -> None:
        s = clean_for_speech(raw)
        if not s or not re.search(r"\w", s):
            return
        if self._held:
            s, self._held = self._held + " " + s, ""
        if self._emitted > 0 and len(s) < self.min_chars:
            self._held = s
            return
        self._emitted += 1
        out.append(s)

    def feed(self, delta: str) -> list[str]:
        self._buf += self._filter_code(delta)
        out: list[str] = []
        while True:
            for m in _END.finditer(self._buf):
                if _is_boundary(self._buf, m):
                    self._emit(self._buf[: m.end()], out)
                    self._buf = self._buf[m.end():]
                    break
            else:
                if len(self._buf) > self.max_chars:  # sin puntuacion: cortar en una coma
                    cut = max(self._buf.rfind(c, self.min_chars, self.max_chars) for c in ",;:")
                    if cut < 0:
                        cut = self._buf.rfind(" ", self.min_chars, self.max_chars)
                    if cut > 0:
                        self._emit(self._buf[: cut + 1], out)
                        self._buf = self._buf[cut + 1:]
                        continue
                return out

    def flush(self) -> list[str]:
        out = self.feed("")
        tail = self._buf + (self._fence_tail if not self._in_code else "")
        self._buf, self._fence_tail, self._in_code = "", "", False
        s = clean_for_speech(tail)
        pieces = [p for p in (self._held, s if re.search(r"\w", s) else "") if p]
        self._held = ""
        if pieces:
            out.append(" ".join(pieces))
        return out


# --------------------------------------------------------------------------
# Motores
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class TtsConfig:
    engine: str = "kokoro"
    voice: str | None = None
    speed: float = 1.0
    model_path: str | None = None
    voices_path: str | None = None  # kokoro
    binary: str = "piper"  # piper
    length_scale: float = 1.0
    lang: str = "es"

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "TtsConfig":
        t = (cfg.get("voice") or {}).get("tts") or {}
        engine = t.get("engine", "kokoro")
        if engine not in ("kokoro", "piper"):
            raise TtsError(f"tts.engine no soportado todavia: {engine!r} (usar kokoro o piper)")
        models = ((cfg.get("settings") or {}).get("paths") or {}).get("models")
        sub = t.get(engine) or {}
        if engine == "kokoro":
            base = Path(models, "kokoro").expanduser() if models else None
            return cls(engine, sub.get("voice"), float(sub.get("speed", 1.0)),
                       sub.get("model") or (str(base / "kokoro-v1.0.onnx") if base else None),
                       sub.get("voices_file") or (str(base / "voices-v1.0.bin") if base else None),
                       lang=sub.get("lang", "es"))
        return cls(engine, sub.get("voice"), 1.0, sub.get("model"),
                   binary=sub.get("binary", "piper"), length_scale=float(sub.get("length_scale", 1.0)))


class KokoroEngine:
    def __init__(self, cfg: TtsConfig):
        if not cfg.voice:
            raise TtsError("Falta tts.kokoro.voice en config/voice.yaml (elegir tras la prueba A/B).")
        for p in (cfg.model_path, cfg.voices_path):
            if not p or not Path(p).exists():
                raise TtsError(f"No existe {p}. Descarga kokoro-v1.0.onnx y voices-v1.0.bin.")
        try:
            from kokoro_onnx import Kokoro
        except (ImportError, OSError) as e:
            raise TtsError("Falta kokoro-onnx: uv pip install -e '.[voice]'") from e
        self._k, self._cfg = Kokoro(cfg.model_path, cfg.voices_path), cfg
        self.sample_rate = 24000

    def synthesize(self, text: str) -> np.ndarray:
        try:
            x, sr = self._k.create(text, voice=self._cfg.voice, speed=self._cfg.speed, lang=self._cfg.lang)
        except Exception as e:
            raise TtsError(f"Kokoro fallo: {e}") from e
        self.sample_rate = int(sr)
        return np.asarray(x, dtype=np.float32)

    def close(self) -> None:
        self._k = None


class PiperEngine:
    """Piper por linea de comandos: texto por stdin, PCM int16 crudo por stdout."""

    def __init__(self, cfg: TtsConfig):
        if not cfg.model_path or not Path(cfg.model_path).expanduser().exists():
            raise TtsError(f"No existe el modelo de Piper: {cfg.model_path} (tts.piper.model).")
        if shutil.which(cfg.binary) is None:
            raise TtsError(f"No se encontro '{cfg.binary}'. Instala piper-tts o ajusta tts.piper.binary.")
        self._cfg = cfg
        self._model = str(Path(cfg.model_path).expanduser())
        meta = Path(self._model + ".json")
        try:
            self.sample_rate = int(json.loads(meta.read_text())["audio"]["sample_rate"])
        except Exception as e:
            raise TtsError(f"No se pudo leer la frecuencia en {meta}: {e}") from e

    def synthesize(self, text: str) -> np.ndarray:
        cmd = [self._cfg.binary, "--model", self._model, "--output_raw",
               "--length_scale", str(self._cfg.length_scale)]
        try:
            r = subprocess.run(cmd, input=text.encode("utf-8"), capture_output=True, timeout=60, check=True)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as e:
            raise TtsError(f"Piper fallo: {e}") from e
        return np.frombuffer(r.stdout, dtype=np.int16).astype(np.float32) / 32768.0

    def close(self) -> None:
        pass


def make_engine(cfg: TtsConfig) -> TTSEngine:
    return KokoroEngine(cfg) if cfg.engine == "kokoro" else PiperEngine(cfg)


# --------------------------------------------------------------------------
# Habla por frases
# --------------------------------------------------------------------------
_STOP = object()


class SpeechPipeline:
    """Texto en streaming -> frases -> sintesis -> altavoz, con interrupcion.

    Uso: begin(); feed(delta)...; end(); (wait()) — o stop() para cortar de inmediato.
    El microfono permanece silenciado durante todo el turno (AudioManager.hold_mute).
    """

    def __init__(self, engine: TTSEngine, audio: Any, splitter_factory=SentenceSplitter, on_error=None):
        self.engine, self.audio, self._new_splitter = engine, audio, splitter_factory
        self.on_error = on_error
        self.errors = 0
        self.first_audio_s: float | None = None
        self._q: queue.Queue = queue.Queue()
        self._turn = 0
        self._splitter: SentenceSplitter | None = None
        self._held = False
        self._t0 = 0.0
        self._idle = threading.Event()
        self._idle.set()
        self._lock = threading.Lock()
        self._worker: threading.Thread | None = None

    # -- turno --
    def begin(self) -> None:
        self.stop()
        with self._lock:
            self._turn += 1
            self._splitter = self._new_splitter()
            self.first_audio_s = None
            self._t0 = time.monotonic()
            self._idle.clear()
            if not self._held:
                self.audio.hold_mute()
                self._held = True
        if self._worker is None or not self._worker.is_alive():
            self._worker = threading.Thread(target=self._run, daemon=True, name="tts")
            self._worker.start()

    def feed(self, delta: str) -> None:
        if self._splitter is None:
            return
        for s in self._splitter.feed(delta):
            self._q.put((self._turn, s))

    def end(self) -> None:
        if self._splitter is None:
            return
        for s in self._splitter.flush():
            self._q.put((self._turn, s))
        self._splitter = None
        self._q.put((self._turn, _STOP))

    def stop(self) -> None:
        """Corta ya: descarta lo pendiente y silencia el altavoz."""
        with self._lock:
            self._turn += 1
            self._splitter = None
            held, self._held = self._held, False
        while True:
            try:
                self._q.get_nowait()
            except queue.Empty:
                break
        self.audio.interrupt()
        if held:
            self.audio.release_mute()
        self._idle.set()

    def wait(self, timeout: float | None = None) -> bool:
        return self._idle.wait(timeout)

    @contextlib.contextmanager
    def open_mic(self):
        """Libera el silencio del micro que mantiene el turno de habla (p. ej. para oir una confirmacion) y lo restituye."""
        with self._lock:
            was, self._held = self._held, False
        if was:
            self.audio.release_mute()
        try:
            yield
        finally:
            if was:
                with self._lock:
                    again = not self._held and not self._idle.is_set()  # si el turno ya termino, no hay que volver a silenciar
                    if again:
                        self._held = True
                if again:
                    self.audio.hold_mute()

    # -- hilo --
    def _run(self) -> None:
        while True:
            turn, item = self._q.get()
            if turn != self._turn:
                continue  # turno cancelado
            if item is _STOP:
                self.audio.wait()
                with self._lock:
                    if turn == self._turn:
                        held, self._held = self._held, False
                        if held:
                            self.audio.release_mute()
                        self._idle.set()
                continue
            try:
                samples = self.engine.synthesize(item)
            except Exception as e:
                self.errors += 1
                if self.on_error:
                    self.on_error(e)
                continue
            if turn != self._turn:
                continue
            if self.first_audio_s is None:
                self.first_audio_s = time.monotonic() - self._t0
            self.audio.play(samples, self.engine.sample_rate)

    def close(self) -> None:
        self.stop()
        self.engine.close()
