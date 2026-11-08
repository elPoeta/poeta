"""Reconocimiento de voz local con faster-whisper.

La configuracion por defecto replica la que ya funcionaba en vozboard:
beam_size 5, filtro VAD con 500 ms de silencio minimo, idioma fijo, pesos en
<models>/whisper. Solo local: no hay backend remoto (enviar audio a la nube
pasaria por PermissionManager y no esta implementado).

El audio puede ser una ruta o un arreglo float32 mono a 16 kHz.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

DEVICES = ("cpu", "cuda", "auto")
SAMPLE_RATE = 16000


class SttError(RuntimeError):
    pass


def load_wav(path: str | Path) -> Any:
    """Lee un WAV PCM a float32 mono 16 kHz sin depender de PyAV.

    Devuelve None si no es un WAV PCM legible (se deja a faster-whisper).
    """
    import wave

    import numpy as np

    try:
        with wave.open(str(path), "rb") as w:
            ch, width, rate, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
            raw = w.readframes(n)
    except (wave.Error, EOFError):
        return None
    dtype = {1: np.uint8, 2: np.int16, 4: np.int32}.get(width)
    if dtype is None:
        return None
    x = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    x = (x - 128.0) / 128.0 if width == 1 else x / float(2 ** (8 * width - 1))
    if ch > 1:
        x = x[: len(x) // ch * ch].reshape(-1, ch).mean(axis=1)
    if rate != SAMPLE_RATE and len(x):
        t = np.arange(int(len(x) * SAMPLE_RATE / rate)) * (rate / SAMPLE_RATE)
        x = np.interp(t, np.arange(len(x)), x).astype(np.float32)
    return x.astype(np.float32)


@dataclass(frozen=True)
class SttConfig:
    model: str = "large-v3"
    device: str = "cuda"
    compute_type: str | None = None  # None -> int8 en cpu, float16 en cuda/auto
    language: str | None = "es"
    beam_size: int = 5
    vad_filter: bool = True
    min_silence_duration_ms: int = 500
    download_root: str | None = None

    def __post_init__(self) -> None:
        if self.device not in DEVICES:
            raise SttError(f"stt.device invalido: {self.device!r} (usar {', '.join(DEVICES)})")
        if self.beam_size < 1:
            raise SttError("stt.beam_size debe ser >= 1")
        if self.min_silence_duration_ms < 0:
            raise SttError("stt.min_silence_duration_ms no puede ser negativo")

    @property
    def effective_compute_type(self) -> str:
        return self.compute_type or ("int8" if self.device == "cpu" else "float16")

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "SttConfig":
        s = dict((cfg.get("voice") or {}).get("stt") or {})
        if s.get("engine", "faster-whisper") != "faster-whisper":
            raise SttError(f"stt.engine no soportado: {s.get('engine')!r}")
        root = s.get("download_root")
        if not root:
            models = ((cfg.get("settings") or {}).get("paths") or {}).get("models")
            root = str(Path(models, "whisper").expanduser()) if models else None
        return cls(
            model=s.get("model") or cls.model,
            device=s.get("device") or cls.device,
            compute_type=s.get("compute_type") or None,
            language=s.get("language") or None,
            beam_size=int(s.get("beam_size", cls.beam_size)),
            vad_filter=bool(s.get("vad_filter", cls.vad_filter)),
            min_silence_duration_ms=int(s.get("min_silence_duration_ms", cls.min_silence_duration_ms)),
            download_root=root,
        )


@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class Transcription:
    text: str
    language: str
    duration: float
    elapsed: float
    segments: list[Segment] = field(default_factory=list)

    @property
    def speed_factor(self) -> float | None:
        return self.duration / self.elapsed if self.elapsed > 0 else None


def _default_factory(cfg: SttConfig) -> Any:
    from .cuda_env import patch_cuda_libs

    if cfg.device != "cpu":
        patch_cuda_libs()  # debe ir antes de que CTranslate2 cargue CUDA
    try:
        from faster_whisper import WhisperModel
    except ImportError as e:
        raise SttError(
            "Falta faster-whisper. Instala los extras de voz:\n"
            "  uv pip install -e '.[voice,cuda]'"
        ) from e
    kwargs: dict[str, Any] = {}
    if cfg.download_root:
        kwargs["download_root"] = cfg.download_root
    return WhisperModel(cfg.model, device=cfg.device,
                        compute_type=cfg.effective_compute_type, **kwargs)


class Transcriber:
    """Envoltorio sobre WhisperModel, cargado una sola vez (carga perezosa)."""

    def __init__(self, config: SttConfig | None = None,
                 model_factory: Callable[[SttConfig], Any] | None = None):
        self.config = config or SttConfig()
        self._factory = model_factory or _default_factory
        self._model: Any = None
        self.load_seconds: float | None = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is None:
            t0 = time.perf_counter()
            try:
                self._model = self._factory(self.config)
            except SttError:
                raise
            except Exception as e:  # CUDA ausente, pesos corruptos, sin red para bajarlos...
                raise SttError(f"No se pudo cargar {self.config.model} en "
                               f"{self.config.device}/{self.config.effective_compute_type}: {e}") from e
            self.load_seconds = time.perf_counter() - t0

    def transcribe(self, audio: Any, language: str | None | object = ...,
                   initial_prompt: str | None = None) -> Transcription:
        """`language=...` usa el configurado; `None` pide autodeteccion."""
        if isinstance(audio, (str, Path)):
            audio = str(audio)
            if not Path(audio).exists():
                raise FileNotFoundError(f"El archivo no existe: {audio}")
            if audio.lower().endswith(".wav"):
                decoded = load_wav(audio)
                if decoded is not None:
                    audio = decoded
        self.load()
        cfg = self.config
        lang = cfg.language if language is ... else language
        extra = {"initial_prompt": initial_prompt} if initial_prompt else {}
        t0 = time.perf_counter()
        try:
            raw, info = self._model.transcribe(
                audio,
                language=lang,
                beam_size=cfg.beam_size,
                vad_filter=cfg.vad_filter,
                vad_parameters={"min_silence_duration_ms": cfg.min_silence_duration_ms},
                **extra,
            )
            segments = [Segment(s.start, s.end, s.text.strip()) for s in raw]
        except Exception as e:
            hint = ""
            if "metadata_errors" in str(e):
                hint = " (PyAV desactualizado: uv pip install -U av)"
            raise SttError(f"Fallo al transcribir: {e}{hint}") from e
        elapsed = time.perf_counter() - t0
        return Transcription(
            text=" ".join(s.text for s in segments if s.text).strip(),
            language=getattr(info, "language", lang or ""),
            duration=float(getattr(info, "duration", 0.0)),
            elapsed=elapsed,
            segments=segments,
        )
