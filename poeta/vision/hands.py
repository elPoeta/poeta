"""Seguimiento de manos con MediaPipe Tasks (HandLandmarker). Los puntos se devuelven en un espacio sin
distorsion de aspecto: x y z multiplicados por ancho/alto, y en [0,1] (unidades de alto de imagen)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .camera import VisionConfigError, _num


@dataclass(frozen=True)
class Hand:
    landmarks: tuple[tuple[float, float, float], ...]  # 21 puntos, espacio corregido por aspecto
    handedness: str = "?"
    score: float = 1.0
    aspect: float = 4 / 3  # ancho/alto de la imagen

    @property
    def wrist_x(self) -> float:
        """Posicion horizontal de la muñeca como fraccion del ancho de la imagen (0..1)."""
        return self.landmarks[0][0] / self.aspect

    @property
    def wrist_y(self) -> float:
        return self.landmarks[0][1]


@dataclass(frozen=True)
class HandsConfig:
    model_path: Path
    num_hands: int = 1
    min_detection_confidence: float = 0.5
    min_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    fps: float = 10.0

    @classmethod
    def from_config(cls, cfg: dict[str, Any], root: Path) -> "HandsConfig":
        h = (cfg.get("vision") or {}).get("hands") or {}
        path = Path(str(h.get("model_path", "models/hand_landmarker.task"))).expanduser()
        if not path.is_absolute():
            path = root / path
        conf = {}
        for k in ("min_detection_confidence", "min_presence_confidence", "min_tracking_confidence"):
            v = _num(h, k, 0.5, 0, float, "hands")
            if v > 1:
                raise VisionConfigError(f"vision.hands.{k} debe estar entre 0 y 1")
            conf[k] = v
        return cls(path, _num(h, "num_hands", 1, 1, int, "hands"), fps=_num(h, "fps", 10, 1, float, "hands"), **conf)


class HandTracker:
    """`backend(image_bgr, ts_ms) -> list[(landmarks_xyz_normalizados, handedness, score)]` es inyectable."""

    def __init__(self, cfg: HandsConfig, backend: Callable[[Any, int], list] | None = None):
        self.cfg = cfg
        self._backend = backend
        self._last_ts = -1

    def load(self) -> None:
        if self._backend is None:
            self._backend = self._mediapipe_backend()

    def _mediapipe_backend(self) -> Callable[[Any, int], list]:
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mpt
            from mediapipe.tasks.python import vision as mpv
        except ImportError as exc:
            raise VisionConfigError("falta mediapipe: uv pip install -e '.[vision]'") from exc
        if not self.cfg.model_path.is_file():
            raise VisionConfigError(f"falta el modelo {self.cfg.model_path}: python scripts/fetch_models.py")
        opts = mpv.HandLandmarkerOptions(
            base_options=mpt.BaseOptions(model_asset_path=str(self.cfg.model_path)),
            running_mode=mpv.RunningMode.VIDEO, num_hands=self.cfg.num_hands,
            min_hand_detection_confidence=self.cfg.min_detection_confidence,
            min_hand_presence_confidence=self.cfg.min_presence_confidence,
            min_tracking_confidence=self.cfg.min_tracking_confidence)
        lm = mpv.HandLandmarker.create_from_options(opts)

        def detect(image_bgr: Any, ts_ms: int) -> list:
            rgb = image_bgr[:, :, ::-1].copy()  # BGR -> RGB
            res = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts_ms)
            out = []
            for pts, cats in zip(res.hand_landmarks, res.handedness):
                out.append(([(p.x, p.y, p.z) for p in pts], cats[0].category_name, cats[0].score))
            return out

        self._close = lm.close
        return detect

    def detect(self, image_bgr: Any, ts_s: float) -> list[Hand]:
        if self._backend is None:
            self.load()
        ts_ms = max(int(ts_s * 1000), self._last_ts + 1)  # MediaPipe exige marcas estrictamente crecientes
        self._last_ts = ts_ms
        h, w = image_bgr.shape[:2]
        aspect = w / h
        hands = []
        for pts, side, score in self._backend(image_bgr, ts_ms):
            if len(pts) != 21:
                continue
            hands.append(Hand(tuple((x * aspect, y, z * aspect) for x, y, z in pts), str(side), float(score), aspect))
        hands.sort(key=lambda hd: -hd.score)
        return hands

    def close(self) -> None:
        fn = getattr(self, "_close", None)
        if fn:
            fn()
