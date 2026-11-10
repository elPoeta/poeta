"""Bus de eventos interno: publica el estado de POETA (escuchando, pensando, hablando...) para el panel y el HUD.

Reglas: nunca lleva texto del usuario ni del modelo (solo estados y numeros); publicar nunca bloquea ni falla; los
suscriptores corren en el hilo que publica, asi que deben ser rapidos (un HUD encola y sale).
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable

log = logging.getLogger("poeta.events")

STATES = ("idle", "listening", "transcribing", "thinking", "tool", "speaking", "confirming", "stopped")


@dataclass(frozen=True)
class Event:
    kind: str                       # state | turn | gesture | permission | mcp | error
    data: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)


class EventBus:
    def __init__(self, history: int = 200):
        self._subs: list[tuple[Callable[[Event], Any], frozenset[str] | None]] = []
        self._lock = threading.Lock()
        self._history: deque[Event] = deque(maxlen=history)
        self.state = "idle"

    def subscribe(self, callback: Callable[[Event], Any], kinds: tuple[str, ...] | None = None,
                  replay: bool = False) -> Callable[[], None]:
        """Devuelve la funcion para desuscribirse. `replay` entrega primero el historial reciente."""
        entry = (callback, frozenset(kinds) if kinds else None)
        with self._lock:
            self._subs.append(entry)
            past = list(self._history) if replay else []
        for ev in past:
            self._deliver(entry, ev)

        def unsubscribe() -> None:
            with self._lock:
                if entry in self._subs:
                    self._subs.remove(entry)

        return unsubscribe

    def _deliver(self, entry: tuple[Callable[[Event], Any], frozenset[str] | None], ev: Event) -> None:
        cb, kinds = entry
        if kinds is not None and ev.kind not in kinds:
            return
        try:
            cb(ev)
        except Exception:  # noqa: BLE001 - un suscriptor roto no debe tumbar a POETA
            log.exception("suscriptor del bus fallo (evento %s)", ev.kind)

    def publish(self, kind: str, /, **data: Any) -> Event:
        ev = Event(kind, data)
        with self._lock:
            self._history.append(ev)
            subs = list(self._subs)
        if kind == "state":
            self.state = str(data.get("state", self.state))
        for entry in subs:
            self._deliver(entry, ev)
        return ev

    def set_state(self, state: str, **data: Any) -> None:
        if state != self.state:
            self.publish("state", state=state, **data)

    def recent(self, kind: str | None = None) -> list[Event]:
        with self._lock:
            return [e for e in self._history if kind is None or e.kind == kind]
