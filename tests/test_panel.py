import asyncio
import functools
import os
import shutil
import tempfile
import threading
import time
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from poeta.ui.link import AgentLink  # noqa: E402
from poeta.ui.panel import Panel  # noqa: E402
from test_ipc import make  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


class Backend:
    """UiServer real en un hilo con su propio loop (el panel corre en el hilo de Qt)."""

    def __init__(self, sockdir):
        self.loop = asyncio.new_event_loop()
        self.bus, self.agent, self.router, self.srv = make(sockdir)
        self.ready = threading.Event()
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()
        assert self.ready.wait(5)

    def _run(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self.srv.start())
        self.ready.set()
        self.loop.run_forever()

    def call(self, fn, *a, **kw):
        self.loop.call_soon_threadsafe(functools.partial(fn, *a, **kw))

    def stop(self):
        asyncio.run_coroutine_threadsafe(self.srv.stop(), self.loop).result(5)
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.t.join(5)


def pump(app, cond, timeout=4.0):
    end = time.time() + timeout
    while time.time() < end:
        app.processEvents()
        if cond():
            return True
        time.sleep(0.01)
    return False


@pytest.fixture
def env(app):
    d = Path(tempfile.mkdtemp(prefix="pt-", dir="/tmp"))
    be = Backend(d)
    link = AgentLink(str(d / "ui.sock"), retry_ms=100)
    panel = Panel(link)
    link.start()
    yield be, link, panel
    link.stop()
    panel.close()
    panel.deleteLater()
    link.deleteLater()
    app.processEvents()
    be.stop()
    shutil.rmtree(d, ignore_errors=True)


def test_snapshot_and_events_reach_panel(app, env):
    be, link, panel = env
    assert pump(app, lambda: panel.snap.get("provider") == "local")
    assert panel.model_lbl.text() == "modelo: bolt"
    assert panel.provider_cb.count() == 2 and panel.reasoning_cb.currentText() == "auto"
    assert panel.mcp_table.rowCount() == 1 and panel.tools_table.rowCount() == 1
    be.call(be.bus.set_state, "speaking")
    assert pump(app, lambda: panel.state_lbl.text() == "Hablando")
    assert panel.stop_btn.isEnabled()
    be.call(be.bus.publish, "resources", ram_total_mb=1000, ram_used_mb=500, gpu_total_mib=8000, gpu_used_mib=4000,
            gpu_util_pct=10, gpu_temp_c=40)
    assert pump(app, lambda: panel.bars["ram"].value() == 50 and panel.bars["gpu"].value() == 50)


def test_commands_change_agent_and_permission_banner(app, env):
    be, link, panel = env
    assert pump(app, lambda: link.is_up and panel.snap)
    panel.reasoning_cb.setCurrentText("on")
    panel._set_reasoning(0)
    assert pump(app, lambda: be.agent.reasoning.mode == "on")
    panel.provider_cb.setCurrentText("nvidia")
    panel._select_provider()
    assert pump(app, lambda: be.router.p == "nvidia" and "NUBE" in panel.cloud_lbl.text())
    be.call(be.bus.publish, "permission", phase="request", kind="desktop_control", tool="window_action", explicit=False)
    assert pump(app, lambda: not panel.banner.isHidden())
    be.call(be.bus.publish, "permission", phase="decided", allowed=False, kind="desktop_control", tool="window_action")
    assert pump(app, lambda: panel.banner.isHidden())


def test_history_tab_queries(app, env):
    be, link, panel = env
    assert pump(app, lambda: link.is_up and panel.snap)
    panel.tabs.setCurrentIndex(1)
    assert pump(app, lambda: panel.conv_list.count() == 1)
    panel.conv_list.setCurrentRow(0)
    assert pump(app, lambda: "[user] texto" in panel.msg_view.toPlainText())


def test_panel_survives_agent_going_away_and_back(app, env):
    be, link, panel = env
    assert pump(app, lambda: link.is_up)
    asyncio.run_coroutine_threadsafe(be.srv.stop(), be.loop).result(5)
    assert pump(app, lambda: panel.state_lbl.text().startswith("Sin conexión"))
    panel.stop_btn.click()  # no debe fallar sin conexion
    asyncio.run_coroutine_threadsafe(be.srv.start(), be.loop).result(5)
    assert pump(app, lambda: link.is_up and panel.snap.get("state") == "idle", 6)
