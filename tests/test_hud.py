import os

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from poeta.ui.hud import Hud  # noqa: E402
from poeta.ui.link import AgentLink  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def make(cfg=None):
    link = AgentLink("/tmp/inexistente-hud.sock")
    return link, Hud(link, cfg)


def test_flags_and_states(app):
    link, h = make()
    f = h.windowFlags()
    assert f & Qt.WindowType.WindowTransparentForInput and f & Qt.WindowType.WindowStaysOnTopHint
    assert f & Qt.WindowType.FramelessWindowHint and f & Qt.WindowType.WindowDoesNotAcceptFocus
    h.on_event("state", {"state": "thinking"})
    assert h.state == "thinking" and h.timer.isActive()
    h.on_event("state", {"state": "idle"})
    assert not h.timer.isActive()
    link.disconnected.emit()
    assert h.state == "offline"
    h.on_event("gesture", {"name": "x"})  # ignorado
    assert h.state == "offline"
    h.close()


def test_hide_when_idle_and_no_click_through(app):
    link, h = make({"hide_when_idle": True, "click_through": False, "always_on_top": False})
    assert not h.windowFlags() & Qt.WindowType.WindowTransparentForInput
    assert not h.windowFlags() & Qt.WindowType.WindowStaysOnTopHint
    h.set_state("speaking")
    assert h.isVisible()
    h.set_state("idle")
    assert not h.isVisible()
    h.close()


def test_every_state_paints(app):
    link, h = make({"size": 120})
    h.show()
    for st in ("offline", "idle", "listening", "transcribing", "thinking", "tool", "speaking", "confirming", "stopped", "raro"):
        h.set_state(st)
        h._tick()
        img = h.grab().toImage()
        assert img.width() == 120
        if st not in ("offline",):
            assert any(img.pixelColor(x, 60).alpha() for x in range(120))
    h.close()


def test_corner_placement(app):
    link, h = make({"corner": "top-left", "margin": 10})
    g = QApplication.primaryScreen().availableGeometry()
    assert (h.x(), h.y()) == (g.left() + 10, g.top() + 10)
    h.close()
    link, h = make({"corner": "bottom-right", "margin": 10, "size": 100})
    assert (h.x(), h.y()) == (g.right() - 100 - 10 + 1, g.bottom() - 100 - 10 + 1)
    h.close()


def test_permission_card_expands_and_reverts(app):
    from poeta.ui.panel import permission_lines
    link, h = make({"size": 100, "corner": "bottom-right", "margin": 10})
    h.show()
    ev = {"phase": "request", "kind": "desktop_control", "tool": "window_action", "explicit": False, "tainted": True,
          "description": "cerrar la ventana de Firefox", "via": ["voz", "gestos"]}
    h.on_event("permission", ev)
    assert h.width() == 100 + h.PERM_W and h.perm
    g = QApplication.primaryScreen().availableGeometry()
    assert h.x() + h.width() - 1 == g.right() - 10  # la esquina derecha no se mueve
    img = h.grab().toImage()
    assert img.width() == h.width()
    assert any(img.pixelColor(x, 50).alpha() for x in range(0, 200))  # la tarjeta dibuja a la izquierda del circulo
    h.on_event("permission", {**ev, "phase": "decided", "allowed": True})
    assert h.width() == 100 and h.perm is None
    lines = permission_lines(ev)
    assert "controlar el escritorio" in lines[0] and "cerrar la ventana de Firefox" in lines
    assert any("contenido externo" in l for l in lines) and "sí" in lines[-1] and "👍" in lines[-1]
    assert "denegará" in permission_lines({**ev, "via": []})[-1]
    assert "explícita" in " ".join(permission_lines({**ev, "explicit": True}))
    h.close()


def test_permission_card_shows_hidden_hud_and_hides_again(app):
    link, h = make({"hide_when_idle": True, "size": 100})
    h.set_state("idle")
    assert not h.isVisible()
    h.on_event("permission", {"phase": "request", "kind": "x", "via": ["teclado"]})
    assert h.isVisible()
    h.on_event("permission", {"phase": "decided", "allowed": False, "kind": "x"})
    assert not h.isVisible()
    h.close()
