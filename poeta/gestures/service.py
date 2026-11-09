"""Servicio de gestos: suscribe a la camara compartida, detecta manos y emite eventos de gesto."""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any, Callable

from poeta.security import ActionRequest
from poeta.vision import CameraManager
from poeta.vision.hands import Hand, HandTracker

from .engine import GestureEngine, GestureEvent

log = logging.getLogger("poeta.gestures")


class GestureService:
    def __init__(self, camera: CameraManager, tracker: HandTracker, engine: GestureEngine,
                 on_event: Callable[[GestureEvent], Any], permissions: Any = None, fps: float = 10.0,
                 on_error: Callable[[str], Any] | None = None,
                 on_frame: Callable[[Any, Hand | None, str], Any] | None = None):
        self.camera, self.tracker, self.engine = camera, tracker, engine
        self.on_event, self.on_error, self.on_frame = on_event, on_error, on_frame
        self.permissions, self.fps = permissions, fps
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self.error: str | None = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    async def start(self) -> bool:
        """Pide permiso (camera_capture, auditado), abre la camara y arranca. Devuelve False si no pudo."""
        if self.running:
            return True
        if self.permissions is not None:
            d = await self.permissions.check(ActionRequest(
                "camera_capture", "Abrir la webcam para reconocer gestos (los cuadros no salen del equipo)",
                {"purpose": "gestures"}))
            if not d.allowed:
                self.error = f"permiso denegado: {d.reason}"
                return False
        self._loop = asyncio.get_running_loop()
        self._stop.clear()
        self._ready.clear()
        self.error = None
        self._thread = threading.Thread(target=self._run, name="gestures", daemon=True)
        self._thread.start()
        await asyncio.to_thread(self._ready.wait, 60)  # carga del modelo + apertura de camara
        return self.error is None

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(5)
        self._thread = None

    def _emit(self, ev: GestureEvent) -> None:
        if self._loop is not None and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self.on_event, ev)
        else:
            self.on_event(ev)

    def _fail(self, msg: str) -> None:
        self.error = msg
        log.error(msg)
        self._ready.set()
        if self.on_error is not None and self._loop is not None and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self.on_error, msg)

    def _run(self) -> None:
        sub = None
        try:
            self.tracker.load()
            sub = self.camera.subscribe("gestures", self.fps)
        except Exception as exc:  # noqa: BLE001
            self._fail(f"{type(exc).__name__}: {exc}")
            return
        self._ready.set()
        try:
            while not self._stop.is_set():
                frame = sub.get(0.5)
                if frame is None:
                    if self.camera.error:
                        self._fail(self.camera.error)
                        return
                    continue
                hands = self.tracker.detect(frame.image, frame.ts)
                hand = hands[0] if hands else None
                for ev in self.engine.update(hand, frame.ts):
                    self._emit(ev)
                if self.on_frame is not None:
                    self.on_frame(frame, hand, self.engine.pose)
        except Exception as exc:  # noqa: BLE001
            log.exception("el servicio de gestos fallo")
            self._fail(f"{type(exc).__name__}: {exc}")
        finally:
            if sub is not None:
                sub.close()
            self.tracker.close()
