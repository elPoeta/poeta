"""Palabra de activacion ("poeta") por transcripcion local.

openWakeWord no trae un modelo para "poeta" y entrenar uno exige un pipeline aparte (pendiente).
Mientras tanto: cada enunciado detectado por el VAD se transcribe con un modelo de Whisper pequeno
en CPU y solo cuenta si empieza con la palabra de activacion. Todo lo demas se descarta en el acto:
no llega al LLM, ni a la memoria, ni a ningun archivo; el audio solo existe en memoria.

La interfaz (`detect(audio) -> restante | None`) es la misma que tendria un detector de modelo
dedicado, de modo que se puede cambiar el motor sin tocar el bucle de voz.
"""

from __future__ import annotations

import difflib
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

import numpy as np

_SEP = " \t\n,.:;-–—…"  # separadores tras la palabra; no se tocan ¿ ¡ ? !


def _fold(word: str) -> str:
    w = unicodedata.normalize("NFD", word.lower())
    return "".join(c for c in w if unicodedata.category(c) != "Mn")


def find_wake(text: str, names: list[str], threshold: float = 0.8, max_words: int = 4) -> str | None:
    """Si el texto empieza con la palabra de activacion (en las primeras `max_words` palabras),
    devuelve lo que sigue; si no, None. Tolera tildes, mayusculas y pequenos errores de Whisper."""
    folded = [_fold(n) for n in names]
    for i, m in enumerate(re.finditer(r"\w+", text)):
        if i >= max_words:
            break
        w = _fold(m.group(0))
        if any(w == n or difflib.SequenceMatcher(None, w, n).ratio() >= threshold for n in folded):
            return text[m.end():].lstrip(_SEP).strip()
    return None


@dataclass(frozen=True)
class WakeConfig:
    engine: str = "transcript"
    phrase: str = "poeta"
    aliases: tuple[str, ...] = ("boeta", "poetas", "poeta")
    threshold: float = 0.8
    stt_model: str = "base"
    stt_device: str = "cpu"
    stt_compute_type: str = "int8"
    followup_s: float = 8.0
    chime: bool = True

    @property
    def names(self) -> list[str]:
        return [self.phrase, *self.aliases]

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "WakeConfig":
        w = (cfg.get("voice") or {}).get("wake_word") or {}
        engine = w.get("engine", "transcript")
        if engine != "transcript":
            raise ValueError(f"wake_word.engine no soportado todavia: {engine!r} (usar transcript)")
        d = cls()
        thr = float(w.get("threshold") or d.threshold)
        if not 0.5 <= thr <= 1.0:
            raise ValueError("wake_word.threshold debe estar entre 0.5 y 1.0")
        return cls(engine, str(w.get("phrase") or d.phrase), tuple(w.get("aliases") or d.aliases), thr,
                   str(w.get("stt_model") or d.stt_model), str(w.get("stt_device") or d.stt_device),
                   str(w.get("stt_compute_type") or d.stt_compute_type),
                   float(w.get("followup_s", d.followup_s)), bool(w.get("chime", d.chime)))


class TranscriptWake:
    """detect(audio float32) -> texto restante tras la palabra de activacion, o None."""

    def __init__(self, cfg: WakeConfig, transcriber: Any):
        self.cfg, self._tr = cfg, transcriber

    def load(self) -> None:
        self._tr.load()

    def detect(self, audio: np.ndarray) -> str | None:
        text = self._tr.transcribe(audio).text
        return find_wake(text, self.cfg.names, self.cfg.threshold)

    def strip(self, text: str) -> str | None:
        return find_wake(text, self.cfg.names, self.cfg.threshold)


def make_chime(sample_rate: int = 24000) -> np.ndarray:
    """Dos tonos cortos y suaves: 'te escucho'."""
    def tone(freq: float, ms: int) -> np.ndarray:
        t = np.arange(int(sample_rate * ms / 1000)) / sample_rate
        env = np.minimum(1.0, np.minimum(t, t[-1] - t) / 0.01)
        return (0.25 * np.sin(2 * np.pi * freq * t) * env).astype(np.float32)

    return np.concatenate([tone(660, 90), np.zeros(int(sample_rate * 0.02), np.float32), tone(880, 120)])
