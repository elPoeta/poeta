"""Gesture Engine propio: poses a partir de landmarks + logica temporal anti-activaciones accidentales.

Poses: open_palm, fist, point, thumb_up, thumb_down, none. Eventos: palm, activate, deactivate, point, thumb_up,
thumb_down, swipe_left, swipe_right. Solo `palm` (detener) funciona siempre; el resto requiere la interaccion activada
(mano abierta sostenida) si require_arming=true. Un gesto NUNCA autoriza por si solo una accion sensible.
"""

from __future__ import annotations

import json
import math
import os
from collections import deque
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Sequence

from poeta.vision.camera import VisionConfigError, _num
from poeta.vision.hands import Hand

FINGERS = {"index": (5, 8), "middle": (9, 12), "ring": (13, 16), "pinky": (17, 20)}  # (mcp, tip)


@dataclass(frozen=True)
class Thresholds:
    finger_extended: float = 1.5
    finger_curled: float = 1.25
    thumb_extended: float = 1.45
    thumb_tucked: float = 1.2
    thumb_vertical: float = 0.6


@dataclass(frozen=True)
class GestureConfig:
    hold_s: float = 0.6
    activate_hold_s: float = 1.5
    armed_s: float = 10.0
    require_arming: bool = True
    cooldown_s: float = 1.0
    grace_s: float = 0.2
    swipe_window_s: float = 0.5
    swipe_min_dx: float = 0.30
    thresholds: Thresholds = field(default_factory=Thresholds)
    calibration_file: str = "data/gesture_calibration.json"

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "GestureConfig":
        g = (cfg.get("vision") or {}).get("gestures") or {}
        t = g.get("thresholds") or {}
        th = Thresholds(*(_num(t, k, getattr(Thresholds, k), 0.01, float, "gestures.thresholds")
                          for k in Thresholds.__dataclass_fields__))
        if th.finger_curled >= th.finger_extended:
            raise VisionConfigError("vision.gestures.thresholds: finger_curled debe ser menor que finger_extended")
        if th.thumb_tucked >= th.thumb_extended:
            raise VisionConfigError("vision.gestures.thresholds: thumb_tucked debe ser menor que thumb_extended")
        if th.thumb_vertical > 1:
            raise VisionConfigError("vision.gestures.thresholds.thumb_vertical debe ser <= 1")
        if not isinstance(g.get("require_arming", True), bool):
            raise VisionConfigError("vision.gestures.require_arming debe ser true o false")
        if g.get("sensitive_actions_require_more_than_gesture", True) is not True:
            raise VisionConfigError("vision.gestures.sensitive_actions_require_more_than_gesture debe ser true: "
                                    "un gesto nunca autoriza solo una accion sensible (secc. 9.3)")
        n = lambda k, d, m=0.0: _num(g, k, d, m, float, "gestures")  # noqa: E731
        c = cls(n("hold_s", 0.6, 0.05), n("activate_hold_s", 1.5, 0.05), n("armed_s", 10, 1),
                g.get("require_arming", True), n("cooldown_s", 1.0), n("grace_s", 0.2), n("swipe_window_s", 0.5, 0.1),
                n("swipe_min_dx", 0.3, 0.05), th, str(g.get("calibration_file", cls.calibration_file)))
        if c.activate_hold_s <= c.hold_s:
            raise VisionConfigError("vision.gestures.activate_hold_s debe ser mayor que hold_s")
        return c


@dataclass(frozen=True)
class GestureEvent:
    name: str
    ts: float


# -- clasificacion de pose ---------------------------------------------------------------------
def _d(a: Sequence[float], b: Sequence[float]) -> float:
    return math.dist(a, b)


def finger_ratio(lm: Sequence[Sequence[float]], finger: str) -> float:
    """Distancia punta-muñeca / distancia nudillo-muñeca: ~1.9 extendido, ~1.0 o menos doblado."""
    mcp, tip = FINGERS[finger]
    base = _d(lm[mcp], lm[0])
    return _d(lm[tip], lm[0]) / base if base > 1e-6 else 0.0


def thumb_ratio(lm: Sequence[Sequence[float]]) -> float:
    """Distancia punta del pulgar-nudillo del menique / ancho de la palma."""
    palm = _d(lm[5], lm[17])
    return _d(lm[4], lm[17]) / palm if palm > 1e-6 else 0.0


def _state(value: float, low: float, high: float, low_name: str = "curl") -> str:
    return "ext" if value >= high else (low_name if value <= low else "mid")


def classify(hand: Hand, th: Thresholds) -> str:
    lm = hand.landmarks
    if len(lm) != 21:
        return "none"
    f = {n: _state(finger_ratio(lm, n), th.finger_curled, th.finger_extended) for n in FINGERS}
    thumb = _state(thumb_ratio(lm), th.thumb_tucked, th.thumb_extended, "tucked")
    others_curled = all(f[n] == "curl" for n in ("middle", "ring", "pinky"))
    if all(s == "ext" for s in f.values()):
        return "open_palm"
    if f["index"] == "ext" and others_curled:
        return "point"
    if all(s == "curl" for s in f.values()):
        if thumb == "ext":
            vx, vy = lm[4][0] - lm[2][0], lm[4][1] - lm[2][1]
            n = math.hypot(vx, vy)
            if n > 1e-6:
                up = -vy / n  # el eje y de la imagen crece hacia abajo
                if up >= th.thumb_vertical:
                    return "thumb_up"
                if up <= -th.thumb_vertical:
                    return "thumb_down"
            return "none"
        if thumb == "tucked":
            return "fist"
    return "none"


# -- calibracion ---------------------------------------------------------------------------------
def mean_finger_ratio(hand: Hand) -> float:
    return sum(finger_ratio(hand.landmarks, n) for n in FINGERS) / len(FINGERS)


def compute_calibration(open_samples: Sequence[float], closed_samples: Sequence[float], base: Thresholds) -> Thresholds:
    if len(open_samples) < 5 or len(closed_samples) < 5:
        raise ValueError("muy pocas muestras: mantene cada pose a la vista de la camara unos segundos")
    srt = lambda xs: sorted(xs)[len(xs) // 2]  # noqa: E731  mediana: robusta a cuadros raros
    op, cl = srt(open_samples), srt(closed_samples)
    if op - cl < 0.4:
        raise ValueError(f"la mano abierta ({op:.2f}) y el puño ({cl:.2f}) casi no se distinguen: "
                         "revisa la iluminacion y que la mano se vea entera")
    return replace(base, finger_extended=round(cl + 0.65 * (op - cl), 3), finger_curled=round(cl + 0.35 * (op - cl), 3))


def save_calibration(path: Path, th: Thresholds) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump({"finger_extended": th.finger_extended, "finger_curled": th.finger_curled}, fh, indent=2)
    os.replace(tmp, path)


def load_calibration(path: Path, base: Thresholds) -> Thresholds:
    """Devuelve los umbrales calibrados; si el archivo no existe o es invalido, los de la configuracion."""
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        ext, cur = float(d["finger_extended"]), float(d["finger_curled"])
    except (OSError, ValueError, KeyError, TypeError):
        return base
    if not (0.2 < cur < ext < 5):
        return base
    return replace(base, finger_extended=ext, finger_curled=cur)


# -- logica temporal --------------------------------------------------------------------------------
_HOLD_EVENTS = {"point": "point", "thumb_up": "thumb_up", "thumb_down": "thumb_down"}


class GestureEngine:
    def __init__(self, cfg: GestureConfig, thresholds: Thresholds | None = None):
        self.cfg = cfg
        self.th = thresholds or cfg.thresholds
        self.pose = "none"      # pose cruda del ultimo cuadro
        self._cand = "none"     # pose estable (con tolerancia a parpadeos)
        self._since = 0.0
        self._diff_since: float | None = None
        self._fired: set[str] = set()
        self._last_event = -1e9
        self._armed_until = -1e9
        self._armed = False
        self._track: deque[tuple[float, float, float]] = deque()
        self._last_update: float | None = None

    @property
    def armed(self) -> bool:
        return self._armed

    def reset(self) -> None:
        self.__init__(self.cfg, self.th)  # type: ignore[misc]

    def update(self, hand: Hand | None, now: float) -> list[GestureEvent]:
        if self._last_update is not None and now - self._last_update > 1.0:
            self.reset()  # la camara se detuvo un rato: no arrastrar esperas viejas
        self._last_update = now
        events: list[GestureEvent] = []
        c = self.cfg

        if self._armed and now >= self._armed_until:
            self._armed = False
            events.append(GestureEvent("deactivate", now))

        self.pose = classify(hand, self.th) if hand is not None else "none"
        if self.pose == self._cand:
            self._diff_since = None
        elif self._diff_since is None:
            self._diff_since = now
        elif now - self._diff_since >= c.grace_s:
            self._cand, self._since, self._diff_since = self.pose, self._diff_since, None  # cuenta desde que aparecio
            self._fired.clear()
        held = now - self._since
        cooled = now - self._last_event >= c.cooldown_s
        usable = self._armed or not c.require_arming

        if self._cand == "open_palm":
            if held >= c.hold_s and "palm" not in self._fired:
                self._fired.add("palm")
                events.append(GestureEvent("palm", now))  # detener: seguro, sin enfriamiento ni activacion
            if held >= c.activate_hold_s and "activate" not in self._fired:
                self._fired.add("activate")
                self._armed, self._armed_until = True, now + c.armed_s
                events.append(GestureEvent("activate", now))
        elif self._cand in _HOLD_EVENTS and held >= c.hold_s and self._cand not in self._fired and usable and cooled:
            self._fired.add(self._cand)
            self._last_event = now
            self._armed_until = now + c.armed_s
            events.append(GestureEvent(_HOLD_EVENTS[self._cand], now))

        if hand is not None:
            self._track.append((now, hand.wrist_x, hand.wrist_y))
        while self._track and now - self._track[0][0] > c.swipe_window_s:
            self._track.popleft()
        if hand is None:
            self._track.clear()
        elif usable and cooled and len(self._track) >= 3:
            t0, x0, y0 = self._track[0]
            dx, dy = hand.wrist_x - x0, hand.wrist_y - y0
            if now - t0 >= 0.1 and abs(dx) >= c.swipe_min_dx and abs(dy) <= 0.6 * abs(dx):
                self._track.clear()
                self._last_event = now
                self._armed_until = now + c.armed_s if self._armed else self._armed_until
                events.append(GestureEvent("swipe_right" if dx > 0 else "swipe_left", now))
        return events
