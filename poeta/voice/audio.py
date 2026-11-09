"""Audio Manager: captura, deteccion de enunciados y reproduccion.

Piezas:
- EnergyVAD: decide si un frame es voz (energia sobre un piso de ruido adaptativo).
- Endpointer: convierte frames en enunciados completos (pre-roll, minimo de voz,
  cierre por silencio, maximo de duracion). Logica pura, sin dispositivos.
- AudioManager: abre el microfono y el altavoz (sounddevice sobre PipeWire),
  y aplica la estrategia anti-eco.

Privacidad: el audio vive solo en memoria; nada se escribe a disco aqui.
"""

from __future__ import annotations

import math
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

STRATEGIES = ("mute_during_playback", "none")


class AudioError(RuntimeError):
    pass


@dataclass(frozen=True)
class AudioConfig:
    input_device: str | int | None = None
    output_device: str | int | None = None
    sample_rate: int = 16000
    frame_ms: int = 30
    margin_db: float = 12.0
    min_level_db: float = -50.0
    start_ms: int = 90
    min_speech_ms: int = 250
    end_silence_ms: int = 700
    preroll_ms: int = 300
    max_utterance_s: float = 30.0
    noise_window_s: float = 5.0
    strategy: str = "mute_during_playback"
    tail_ms: int = 300

    def __post_init__(self) -> None:
        if self.strategy not in STRATEGIES:
            raise AudioError(f"echo_cancel.strategy invalido: {self.strategy!r} (usar {', '.join(STRATEGIES)})")
        if self.sample_rate not in (8000, 16000, 32000, 48000):
            raise AudioError(f"sample_rate no soportado: {self.sample_rate}")
        if not 10 <= self.frame_ms <= 100:
            raise AudioError("frame_ms debe estar entre 10 y 100")
        for name in ("start_ms", "min_speech_ms", "end_silence_ms", "preroll_ms", "tail_ms"):
            if getattr(self, name) < 0:
                raise AudioError(f"{name} no puede ser negativo")
        if self.max_utterance_s <= 0:
            raise AudioError("max_utterance_s debe ser positivo")
        if self.noise_window_s <= 0:
            raise AudioError("noise_window_s debe ser positivo")

    @property
    def frame_samples(self) -> int:
        return self.sample_rate * self.frame_ms // 1000

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "AudioConfig":
        a = cfg.get("audio") or {}
        v, e = a.get("vad") or {}, a.get("echo_cancel") or {}
        d = cls()
        return cls(
            input_device=a.get("input_device"),
            output_device=a.get("output_device"),
            sample_rate=int(a.get("sample_rate", d.sample_rate)),
            frame_ms=int(a.get("frame_ms", d.frame_ms)),
            margin_db=float(v.get("margin_db", d.margin_db)),
            min_level_db=float(v.get("min_level_db", d.min_level_db)),
            start_ms=int(v.get("start_ms", d.start_ms)),
            min_speech_ms=int(v.get("min_speech_ms", d.min_speech_ms)),
            end_silence_ms=int(v.get("end_silence_ms", d.end_silence_ms)),
            preroll_ms=int(v.get("preroll_ms", d.preroll_ms)),
            max_utterance_s=float(v.get("max_utterance_s", d.max_utterance_s)),
            noise_window_s=float(v.get("noise_window_s", d.noise_window_s)),
            strategy=e.get("strategy") or d.strategy,
            tail_ms=int(e.get("tail_ms", d.tail_ms)),
        )


def level_db(frame: np.ndarray) -> float:
    """Nivel RMS en dBFS de un frame int16 o float32 (-120 si es silencio total)."""
    x = frame.astype(np.float32)
    if frame.dtype == np.int16:
        x = x / 32768.0
    rms = float(np.sqrt(np.mean(x * x))) if x.size else 0.0
    return 20.0 * math.log10(rms) if rms > 1e-6 else -120.0


class EnergyVAD:
    """Voz = nivel por encima del piso de ruido + margen.

    El piso es el percentil 10 de los niveles de los ultimos `window_frames` frames
    (ventana deslizante). Asi sigue al ruido ambiente aunque este suene todo el tiempo
    (television, ventilador, calle) y no se queda pegado en un valor bajo.
    """

    def __init__(self, margin_db: float = 12.0, min_level_db: float = -50.0, window_frames: int = 166):
        self.margin_db = margin_db
        self.min_level_db = min_level_db
        self._levels: deque[float] = deque(maxlen=max(2, window_frames))

    @property
    def noise_db(self) -> float | None:
        return float(np.percentile(self._levels, 10)) if self._levels else None

    def is_speech(self, frame: np.ndarray) -> bool:
        lvl = level_db(frame)
        floor = self.noise_db
        self._levels.append(lvl)
        if floor is None:
            return False  # el primer frame solo calibra
        return lvl > max(floor + self.margin_db, self.min_level_db)

    def reset(self) -> None:
        self._levels.clear()


class Endpointer:
    """Frames -> enunciados. feed() devuelve el audio int16 del enunciado al cerrarse."""

    def __init__(self, cfg: AudioConfig, vad: EnergyVAD | None = None):
        self.cfg = cfg
        self.vad = vad or EnergyVAD(cfg.margin_db, cfg.min_level_db,
                                    int(cfg.noise_window_s * 1000 / cfg.frame_ms))
        f = cfg.frame_ms
        self._start_frames = max(1, math.ceil(cfg.start_ms / f))
        self._end_frames = max(1, math.ceil(cfg.end_silence_ms / f))
        self._min_frames = max(1, math.ceil(cfg.min_speech_ms / f))
        self._max_frames = int(cfg.max_utterance_s * 1000 / f)
        self._preroll: deque[np.ndarray] = deque(maxlen=max(1, math.ceil(cfg.preroll_ms / f)))
        self.reset()

    def reset(self) -> None:
        self._active = False
        self._buf: list[np.ndarray] = []
        self._speech_run = 0
        self._silence_run = 0
        self._speech_frames = 0
        self._preroll.clear()

    @property
    def speaking(self) -> bool:
        return self._active

    def feed(self, frame: np.ndarray) -> np.ndarray | None:
        speech = self.vad.is_speech(frame)
        if not self._active:
            self._preroll.append(frame)
            self._speech_run = self._speech_run + 1 if speech else 0
            if self._speech_run >= self._start_frames:
                self._active = True
                self._buf = list(self._preroll)
                self._speech_frames = self._speech_run
                self._silence_run = 0
            return None
        self._buf.append(frame)
        if speech:
            self._speech_frames += 1
            self._silence_run = 0
        else:
            self._silence_run += 1
        if self._silence_run >= self._end_frames or len(self._buf) >= self._max_frames:
            audio = np.concatenate(self._buf) if self._speech_frames >= self._min_frames else None
            self.reset()
            return audio
        return None


def to_float32(pcm16: np.ndarray) -> np.ndarray:
    """int16 -> float32 [-1, 1], el formato que espera faster-whisper."""
    return pcm16.astype(np.float32) / 32768.0


class AudioManager:
    """Microfono + altavoz con estrategia anti-eco. `sd` se inyecta en las pruebas."""

    def __init__(self, cfg: AudioConfig | None = None, sd: Any = None):
        self.cfg = cfg or AudioConfig()
        self._sd = sd
        self._q: queue.Queue[tuple[np.ndarray, bool]] = queue.Queue(maxsize=200)  # ~6 s de frames
        self._in: Any = None
        self._out: Any = None
        self._out_rate: int | None = None
        self._lock = threading.Lock()
        self._play_buf = np.zeros(0, dtype=np.float32)
        self._muted_until = 0.0
        self._cancel = threading.Event()
        self._holds = 0  # turnos de habla activos (aunque haya huecos entre frases)
        self.overruns = 0
        self.endpointer = Endpointer(self.cfg)

    # -- ciclo de vida -------------------------------------------------
    def _backend(self) -> Any:
        if self._sd is None:
            try:
                import sounddevice as sd
            except (ImportError, OSError) as e:
                raise AudioError("Falta sounddevice/PortAudio. Instala: uv pip install -e '.[voice]' "
                                 "(y portaudio en el sistema).") from e
            self._sd = sd
        return self._sd

    def start(self) -> None:
        if self._in is not None:
            return
        sd = self._backend()
        try:
            self._in = sd.InputStream(
                samplerate=self.cfg.sample_rate, channels=1, dtype="int16",
                blocksize=self.cfg.frame_samples, device=self.cfg.input_device,
                callback=self._on_input)
            self._in.start()
        except Exception as e:
            self._in = None
            raise AudioError(f"No se pudo abrir el microfono ({self.cfg.input_device or 'predeterminado'}): {e}") from e

    def stop(self) -> None:
        self.interrupt()
        for s in (self._in, self._out):
            if s is not None:
                try:
                    s.stop()
                    s.close()
                except Exception:
                    pass
        self._in = self._out = None
        self._out_rate = None

    def __enter__(self) -> "AudioManager":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    # -- captura -------------------------------------------------------
    @property
    def muted(self) -> bool:
        if self.cfg.strategy != "mute_during_playback":
            return False
        return self.speaking or self._holds > 0 or time.monotonic() < self._muted_until

    def _on_input(self, indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        frame = indata[:, 0].copy() if indata.ndim == 2 else indata.copy()
        item = (frame, self.muted)  # el silenciado se decide al CAPTURAR, no al consumir
        try:
            self._q.put_nowait(item)
        except queue.Full:
            self.overruns += 1
            try:
                self._q.get_nowait()
                self._q.put_nowait(item)
            except (queue.Empty, queue.Full):
                pass

    def hold_mute(self) -> None:
        """Mantiene el micrófono silenciado durante todo un turno de habla, incluso en los
        huecos entre frases (la síntesis puede tardar más que la cola de eco)."""
        with self._lock:
            self._holds += 1

    def release_mute(self) -> None:
        with self._lock:
            self._holds = max(0, self._holds - 1)
            if self._holds == 0:
                self._muted_until = time.monotonic() + self.cfg.tail_ms / 1000.0

    def cancel_listen(self) -> None:
        """Hace que un listen() en curso (otro hilo) vuelva con None."""
        self._cancel.set()

    def listen(self, timeout: float | None = None) -> np.ndarray | None:
        """Bloquea hasta tener un enunciado completo (int16) o vencer el timeout (None)."""
        if self._in is None:
            raise AudioError("AudioManager no esta iniciado (start()).")
        self._cancel.clear()
        deadline = None if timeout is None else time.monotonic() + timeout
        while not self._cancel.is_set():
            wait = 0.1 if deadline is None else min(0.1, deadline - time.monotonic())
            if wait <= 0:
                return None
            try:
                frame, was_muted = self._q.get(timeout=wait)
            except queue.Empty:
                continue
            if was_muted:
                self.endpointer.reset()  # lo capturado mientras habla POETA es eco
                continue
            utt = self.endpointer.feed(frame)
            if utt is not None:
                return utt
        return None

    def flush_input(self) -> None:
        while True:
            try:
                self._q.get_nowait()
            except queue.Empty:
                break
        self.endpointer.reset()

    # -- reproduccion --------------------------------------------------
    @property
    def speaking(self) -> bool:
        with self._lock:
            return self._play_buf.size > 0

    def _on_output(self, outdata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        with self._lock:
            n = min(frames, self._play_buf.size)
            outdata[:n, 0] = self._play_buf[:n]
            outdata[n:, 0] = 0.0
            self._play_buf = self._play_buf[n:]
            if n and self._play_buf.size == 0:
                self._muted_until = time.monotonic() + self.cfg.tail_ms / 1000.0

    def play(self, samples: np.ndarray, sample_rate: int) -> None:
        """Encola audio float32 mono (no bloquea). Util para reproducir por frases."""
        sd = self._backend()
        if self._out is not None and self._out_rate != sample_rate:
            if self.speaking:
                raise AudioError("No se puede cambiar la frecuencia mientras se reproduce.")
            self._out.stop()
            self._out.close()
            self._out = None
        if self._out is None:
            try:
                self._out = sd.OutputStream(samplerate=sample_rate, channels=1, dtype="float32",
                                            device=self.cfg.output_device, callback=self._on_output)
                self._out.start()
            except Exception as e:
                self._out = None
                raise AudioError(f"No se pudo abrir el altavoz: {e}") from e
            self._out_rate = sample_rate
        x = np.asarray(samples, dtype=np.float32).reshape(-1)
        with self._lock:
            self._play_buf = np.concatenate([self._play_buf, x])
        self.endpointer.reset()

    def wait(self, timeout: float | None = None) -> bool:
        """Espera a que termine lo encolado. True si termino, False si vencio el timeout."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while self.speaking:
            if deadline is not None and time.monotonic() >= deadline:
                return False
            time.sleep(0.01)
        return True

    def speak(self, samples: np.ndarray, sample_rate: int) -> None:
        self.play(samples, sample_rate)
        self.wait()

    def interrupt(self) -> None:
        """Corta la reproduccion de inmediato."""
        with self._lock:
            had = self._play_buf.size > 0
            self._play_buf = np.zeros(0, dtype=np.float32)
        if had:
            self._muted_until = time.monotonic() + self.cfg.tail_ms / 1000.0
