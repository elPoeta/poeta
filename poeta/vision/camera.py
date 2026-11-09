"""Camera Manager: UNA captura compartida que se reparte entre los modulos (secc. 9.1).

La webcam solo esta abierta mientras haya al menos un suscriptor; al irse el ultimo se libera. Cada suscriptor
recibe, como maximo, `fps` cuadros por segundo y siempre el mas reciente (si va lento, se descartan los viejos).
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

log = logging.getLogger("poeta.camera")


class VisionConfigError(ValueError):
    pass


class CameraError(RuntimeError):
    pass


def _num(d: dict, key: str, default: Any, minimum: float, kind: type = float, where: str = "camera") -> Any:
    v = d.get(key, default)
    if isinstance(v, bool) or not isinstance(v, (int, float)) or v < minimum:
        raise VisionConfigError(f"vision.{where}.{key} debe ser un numero >= {minimum}")
    return kind(v)


@dataclass(frozen=True)
class CameraConfig:
    device: str | int = "/dev/video0"
    width: int = 640
    height: int = 480
    fps: float = 15.0
    fourcc: str = "MJPG"
    mirror: bool = True
    read_failures_max: int = 30

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "CameraConfig":
        c = (cfg.get("vision") or {}).get("camera") or {}
        dev = c.get("device", cls.device)
        if isinstance(dev, bool) or not isinstance(dev, (str, int)) or dev == "":
            raise VisionConfigError("vision.camera.device debe ser una ruta (/dev/video0) o un indice")
        fourcc = c.get("fourcc", cls.fourcc)
        if not isinstance(fourcc, str) or len(fourcc) != 4:
            raise VisionConfigError("vision.camera.fourcc debe tener 4 caracteres (MJPG, YUYV)")
        if not isinstance(c.get("mirror", True), bool):
            raise VisionConfigError("vision.camera.mirror debe ser true o false")
        return cls(dev, _num(c, "width", 640, 16, int), _num(c, "height", 480, 16, int), _num(c, "fps", 15, 1),
                   fourcc, c.get("mirror", True), _num(c, "read_failures_max", 30, 1, int))


@dataclass(frozen=True)
class Frame:
    image: Any  # ndarray BGR de solo lectura: no modificar
    ts: float   # time.monotonic()
    seq: int


class Subscription:
    """Cola de un solo elemento: `get` devuelve el cuadro mas reciente que no se haya entregado."""

    def __init__(self, name: str, fps: float, manager: "CameraManager"):
        self.name, self.min_interval = name, 1.0 / fps
        self._mgr = manager
        self._cond = threading.Condition()
        self._frame: Frame | None = None
        self._last_push = 0.0
        self.closed = False

    def _push(self, frame: Frame) -> None:
        if frame.ts - self._last_push < self.min_interval:
            return
        self._last_push = frame.ts
        with self._cond:
            self._frame = frame
            self._cond.notify()

    def get(self, timeout: float = 1.0) -> Frame | None:
        with self._cond:
            if self._frame is None and not self.closed:
                self._cond.wait(timeout)
            f, self._frame = self._frame, None
            return f

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            with self._cond:
                self._cond.notify_all()
            self._mgr._unsubscribe(self)

    def __enter__(self) -> "Subscription":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def open_capture(cfg: CameraConfig) -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise CameraError("falta opencv: uv pip install -e '.[vision]'") from exc
    dev = cfg.device
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2) if isinstance(dev, (str, int)) else cv2.VideoCapture(dev)
    if not cap.isOpened():
        cap.release()
        raise CameraError(f"no pude abrir {dev}: ¿esta en uso por otra app, o tu usuario no esta en el grupo 'video'? "
                          "(python scripts/camera_probe.py)")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*cfg.fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
    cap.set(cv2.CAP_PROP_FPS, cfg.fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


class CameraManager:
    def __init__(self, cfg: CameraConfig, capture_factory: Callable[[CameraConfig], Any] = open_capture):
        self.cfg, self._factory = cfg, capture_factory
        self._lock = threading.Lock()
        self._subs: list[Subscription] = []
        self._cap: Any = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._shut = threading.Lock()  # un cierre en curso termina antes de abrir de nuevo
        self.error: str | None = None
        self.frames = 0
        self._t_first = 0.0

    @property
    def active(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def subscribe(self, name: str, fps: float | None = None) -> Subscription:
        with self._shut, self._lock:
            if not self._subs:
                self._open()
            sub = Subscription(name, min(fps or self.cfg.fps, self.cfg.fps), self)
            self._subs.append(sub)
            return sub

    def _open(self) -> None:
        self.error = None
        self._cap = self._factory(self.cfg)  # puede lanzar CameraError: nada queda a medias
        self._stop = threading.Event()  # uno por apertura: un hilo viejo nunca ve el "stop" de otro
        self.frames = 0
        self._t_first = time.monotonic()
        self._thread = threading.Thread(target=self._run, args=(self._cap, self._stop), name="camera", daemon=True)
        self._thread.start()
        log.info("camara abierta", extra={"device": str(self.cfg.device)})

    def _unsubscribe(self, sub: Subscription) -> None:
        with self._lock:
            if sub in self._subs:
                self._subs.remove(sub)
            handles = None if self._subs else self._detach()
        if handles:
            with self._shut:
                self._shutdown(*handles)

    def _detach(self) -> tuple[Any, Any]:
        """Quita el hilo y la captura del manager (bajo el candado); se detienen fuera del candado."""
        self._stop.set()
        handles = (self._thread, self._cap)
        self._thread = self._cap = None
        return handles

    @staticmethod
    def _shutdown(thread: Any, cap: Any) -> None:
        if thread is not None and thread is not threading.current_thread():
            thread.join(2.0)  # el hilo necesita el candado para terminar: nunca llamar con el candado tomado
        if cap is not None:
            try:
                cap.release()
            except Exception:  # noqa: BLE001
                log.exception("fallo al liberar la camara")
        log.info("camara liberada")

    def _run(self, cap: Any, stop: threading.Event) -> None:
        fails, seq = 0, 0
        while not stop.is_set():
            try:
                ok, img = cap.read()
            except Exception as exc:  # noqa: BLE001
                ok, img = False, None
                log.warning("lectura de camara fallo: %s", exc)
            if not ok or img is None:
                fails += 1
                if fails >= self.cfg.read_failures_max:
                    self.error = f"la camara dejo de entregar cuadros ({fails} lecturas fallidas seguidas)"
                    log.error(self.error)
                    with self._lock:
                        for s in self._subs:
                            with s._cond:
                                s._cond.notify_all()
                    return
                time.sleep(0.02)
                continue
            fails = 0
            if self.cfg.mirror:
                img = img[:, ::-1]
            try:
                img = img.copy()
                img.flags.writeable = False
            except AttributeError:
                pass
            seq += 1
            self.frames = seq
            frame = Frame(img, time.monotonic(), seq)
            with self._lock:
                subs = list(self._subs)
            for s in subs:
                s._push(frame)

    def fps(self) -> float:
        dt = time.monotonic() - self._t_first
        return self.frames / dt if self.active and dt > 0 else 0.0

    def close(self) -> None:
        with self._lock:
            for s in list(self._subs):
                s.closed = True
                with s._cond:
                    s._cond.notify_all()
            self._subs.clear()
            handles = self._detach()
        with self._shut:
            self._shutdown(*handles)
