"""HUD: circulo flotante y animado con el estado de POETA. Solo lee eventos; no puede ordenar nada al agente."""

from __future__ import annotations

import math
from typing import Any

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QAction, QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QApplication, QMenu, QWidget

from .link import AgentLink
from .panel import STATE_COLORS, permission_lines

ANIMATED = {"listening", "transcribing", "thinking", "tool", "speaking", "confirming"}
CORNERS = ("top-left", "top-right", "bottom-left", "bottom-right")
DEFAULTS = {"corner": "bottom-right", "size": 96, "margin": 24, "opacity": 0.9, "hide_when_idle": False,
            "click_through": True, "always_on_top": True}


class Hud(QWidget):
    def __init__(self, link: AgentLink, cfg: dict[str, Any] | None = None):
        super().__init__()
        self.cfg = {**DEFAULTS, **{k: v for k, v in (cfg or {}).items() if v is not None}}
        self.state = "offline"
        self.phase = 0.0
        self.perm: list[str] | None = None  # lineas del permiso pendiente (None = ninguno)
        flags = Qt.WindowType.FramelessWindowHint | Qt.WindowType.Tool | Qt.WindowType.WindowDoesNotAcceptFocus
        if self.cfg["always_on_top"]:
            flags |= Qt.WindowType.WindowStaysOnTopHint
        if self.cfg["click_through"]:
            flags |= Qt.WindowType.WindowTransparentForInput
        self.setWindowFlags(flags)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setWindowTitle("POETA HUD")
        self.size_px = int(self.cfg["size"])
        self.setFixedSize(self.size_px, self.size_px)
        self.setWindowOpacity(float(self.cfg["opacity"]))
        self._drag: QPointF | None = None
        self.timer = QTimer(self)
        self.timer.setInterval(33)
        self.timer.timeout.connect(self._tick)
        link.event.connect(self.on_event)
        link.snapshot.connect(lambda s: self.set_state(s.get("state", "idle")))
        link.disconnected.connect(lambda: self.set_state("offline"))
        self.place()

    # ------------------------------------------------------------------ estado

    def on_event(self, kind: str, data: dict[str, Any]) -> None:
        if kind == "state":
            self.set_state(str(data.get("state", "idle")))
        elif kind == "permission":
            if data.get("phase") == "request":
                self.show_permission(permission_lines(data))
            else:
                self.show_permission(None)

    def set_state(self, state: str) -> None:
        self.state = state
        animated = state in ANIMATED
        if animated and not self.timer.isActive():
            self.timer.start()
        elif not animated:
            self.timer.stop()
            self.phase = 0.0
        hide = self.cfg["hide_when_idle"] and state in ("idle", "offline", "stopped")
        if hide and self.isVisible():
            self.hide()
        elif not hide and not self.isVisible():
            self.show()
        self.update()

    PERM_W = 340

    @property
    def orb_on_right(self) -> bool:
        return str(self.cfg["corner"]).endswith("right")

    def show_permission(self, lines: list[str] | None) -> None:
        """Agranda el HUD con una tarjeta que dice que permiso se pide y como contestarlo."""
        self.perm = lines
        w = self.size_px + (self.PERM_W if lines else 0)
        self.setFixedSize(w, self.size_px)
        self.place()
        if lines and not self.isVisible():
            self.show()
        elif not lines:
            self.set_state(self.state)  # re-aplica ocultar-si-inactivo
        self.update()

    def _tick(self) -> None:
        self.phase = (self.phase + 0.05) % (2 * math.pi * 10)
        self.update()

    def place(self) -> None:
        screen = QApplication.primaryScreen()
        if screen is None:
            return
        g = screen.availableGeometry()
        m, s = int(self.cfg["margin"]), int(self.cfg["size"])
        corner = self.cfg["corner"] if self.cfg["corner"] in CORNERS else DEFAULTS["corner"]
        w = self.width()
        x = g.left() + m if corner.endswith("left") else g.right() - w - m + 1
        y = g.top() + m if corner.startswith("top") else g.bottom() - s - m + 1
        self.move(x, y)  # en Wayland el compositor puede ignorarlo: ver la regla de KWin en el README

    # ------------------------------------------------------------------ dibujo

    def paintEvent(self, _e: Any) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        c = QColor(STATE_COLORS.get(self.state, STATE_COLORS["idle"]))
        w = self.size_px
        ox = self.width() - w if self.orb_on_right else 0
        if self.perm:
            self._paint_card(p, 0 if self.orb_on_right else w, self.width() - w)
        p.translate(ox, 0)
        center = QPointF(w / 2, w / 2)
        base = w * 0.28
        ph = self.phase
        st = self.state
        if st == "offline":
            p.setPen(QPen(c, 3))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(center, base, base)
            return
        halo = c
        halo.setAlpha(70)
        if st in ("listening", "confirming"):
            speed = 3.0 if st == "confirming" else 1.5
            k = 0.5 + 0.5 * math.sin(ph * speed)
            r = base * (1.25 + 0.45 * k)
            halo.setAlpha(int(40 + 90 * (1 - k)))
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(halo)
            p.drawEllipse(center, r, r)
        elif st in ("thinking", "transcribing"):
            pen = QPen(c, w * 0.05)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            p.setPen(pen)
            rr = base * 1.5
            rect = QRectF(center.x() - rr, center.y() - rr, 2 * rr, 2 * rr)
            span = 110 if st == "thinking" else 60
            p.drawArc(rect, int(-ph * 180 / math.pi * 2 * 16), span * 16)
            p.drawArc(rect, int((-ph * 180 / math.pi * 2 + 180) * 16), span * 16)
        elif st == "tool":
            p.setPen(Qt.PenStyle.NoPen)
            for i in range(8):
                a = ph * 2 + i * math.pi / 4
                col = QColor(c)
                col.setAlpha(int(60 + 195 * ((i / 8))))
                p.setBrush(col)
                p.drawEllipse(QPointF(center.x() + math.cos(a) * base * 1.5, center.y() + math.sin(a) * base * 1.5),
                              w * 0.04, w * 0.04)
        elif st == "speaking":
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(halo)
            for i in range(3):
                k = 0.5 + 0.5 * math.sin(ph * 3 - i * 0.9)
                r = base * (1.2 + 0.18 * i + 0.25 * k)
                col = QColor(c)
                col.setAlpha(int(110 - 30 * i))
                p.setPen(QPen(col, 2))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawEllipse(center, r, r)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(c)
        p.drawEllipse(center, base, base)

    def _paint_card(self, p: QPainter, x: int, width: int) -> None:
        h = self.size_px
        rect = QRectF(x + 4, h * 0.08, width - 8, h * 0.84)
        p.setPen(QPen(QColor("#ef4444"), 2))
        p.setBrush(QColor(20, 22, 26, 235))
        p.drawRoundedRect(rect, 12, 12)
        p.setPen(QColor("#e5e7eb"))
        f = QFont(p.font())
        f.setPointSize(max(8, int(h / 10)))
        p.setFont(f)
        inner = rect.adjusted(10, 6, -10, -6)
        p.drawText(inner, int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter | Qt.TextFlag.TextWordWrap),
                   "\n".join(self.perm or []))

    # ------------------------------------------------------------------ interaccion (si no es click-through)

    def mousePressEvent(self, e: Any) -> None:
        if e.button() == Qt.MouseButton.LeftButton:
            self._drag = e.globalPosition() - QPointF(self.pos())
        elif e.button() == Qt.MouseButton.RightButton:
            menu = QMenu(self)
            quit_ = QAction("Cerrar HUD", menu)
            quit_.triggered.connect(self.close)
            menu.addAction(quit_)
            menu.exec(e.globalPosition().toPoint())

    def mouseMoveEvent(self, e: Any) -> None:
        if self._drag is not None:
            self.move((e.globalPosition() - self._drag).toPoint())

    def mouseReleaseEvent(self, _e: Any) -> None:
        self._drag = None


def run_hud(path: str | None = None, cfg: dict[str, Any] | None = None) -> int:
    import os
    import sys

    from poeta.ipc.server import default_socket_path

    app = QApplication.instance() or QApplication(sys.argv[:1])
    link = AgentLink(str(path or default_socket_path()))
    hud = Hud(link, cfg)
    link.start()
    hud.set_state("offline")
    if not hud.cfg["hide_when_idle"]:
        hud.show()
    if os.environ.get("QT_QPA_PLATFORM") == "offscreen":
        QTimer.singleShot(0, app.quit)
    return app.exec()
