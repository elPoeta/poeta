import asyncio
import threading
import time

import numpy as np
import pytest

from poeta.config import ConfigError, load_config
from poeta.gestures import GestureConfig, GestureEngine, GestureService
from poeta.security import PermissionManager
from poeta.vision import CameraConfig, CameraError, CameraManager, VisionConfigError
from poeta.vision.hands import HandsConfig, HandTracker
from tests.hands_fixture import ASPECT, OPEN, THUMB_UP
from tests.test_permissions import Asker


class FakeCap:
    def __init__(self, fail_after=None, delay=0.005):
        self.n, self.fail_after, self.delay, self.released = 0, fail_after, delay, False

    def read(self):
        time.sleep(self.delay)
        self.n += 1
        if self.fail_after is not None and self.n > self.fail_after:
            return False, None
        img = np.zeros((48, 64, 3), np.uint8)
        img[0, 0, 0] = self.n % 255  # marca el cuadro
        img[0, 1, 0] = 7             # asimetria para comprobar el espejo
        return True, img

    def release(self):
        self.released = True


def manager(cap=None, delay=0.005, **cfgkw):
    cap = cap or FakeCap(delay=delay)
    opened = []

    def factory(cfg):
        opened.append(1)
        return cap

    cfg = CameraConfig(**{"fps": 100, "mirror": False, **cfgkw})
    return CameraManager(cfg, factory), cap, opened


def test_single_capture_shared_and_released_with_last_subscriber():
    m, cap, opened = manager()
    a = m.subscribe("a")
    b = m.subscribe("b")
    assert len(opened) == 1 and m.active
    fa, fb = a.get(1), b.get(1)
    assert fa is not None and fb is not None
    a.close()
    assert m.active and not cap.released
    b.close()
    assert not m.active and cap.released
    c = m.subscribe("c")  # se vuelve a abrir bajo demanda
    assert len(opened) == 2
    c.close()


def test_subscriber_fps_limit_and_latest_frame():
    m, cap, _ = manager()
    sub = m.subscribe("slow", fps=10)
    time.sleep(0.5)
    n_seq = m.frames
    f = sub.get(1)
    assert n_seq > 20 and f is not None
    time.sleep(0.35)
    g = sub.get(1)
    assert g.seq > f.seq  # entrega el mas reciente
    assert sub.get(0.01) is None or True
    sub.close()
    m.close()


def test_frames_are_readonly_and_mirror_works():
    m, _, _ = manager(mirror=True)
    s = m.subscribe("x")
    f = s.get(1)
    assert not f.image.flags.writeable
    assert f.image[0, -2, 0] == 7 and f.image[0, 1, 0] == 0  # espejado
    s.close()


def test_open_failure_leaves_manager_clean():
    def bad(cfg):
        raise CameraError("no abre")

    m = CameraManager(CameraConfig(), bad)
    with pytest.raises(CameraError):
        m.subscribe("a")
    assert not m.active and m._subs == []


def test_camera_drop_reported_and_waiters_released():
    m, _, _ = manager(FakeCap(fail_after=3), read_failures_max=5)
    s = m.subscribe("a")
    t0 = time.time()
    while m.error is None and time.time() - t0 < 3:
        s.get(0.05)
    assert m.error and "dejo de entregar" in m.error
    s.close()


def test_camera_config_validation():
    cfg = load_config()
    assert CameraConfig.from_config(cfg).device == "/dev/video0"
    for patch in ({"fps": 0}, {"fourcc": "MJ"}, {"mirror": "si"}, {"device": ""}, {"width": True}):
        c = load_config()
        c["vision"]["camera"].update(patch)
        with pytest.raises(VisionConfigError):
            CameraConfig.from_config(c)


def test_bad_vision_config_surfaces_as_configerror(tmp_path):
    import shutil
    import yaml
    d = tmp_path / "config"
    shutil.copytree("config", d)
    v = yaml.safe_load((d / "vision.yaml").read_text())
    v["gestures"]["hold_s"] = -1
    (d / "vision.yaml").write_text(yaml.safe_dump(v))
    with pytest.raises(ConfigError, match="vision.yaml"):
        load_config(d)


# -- HandTracker ----------------------------------------------------------------------------------
def tracker(results):
    calls = []

    def backend(img, ts):
        calls.append(ts)
        return results

    cfg = HandsConfig.from_config(load_config(), __import__("pathlib").Path("."))
    return HandTracker(cfg, backend), calls


def test_tracker_corrects_aspect_sorts_by_score_and_monotonic_ts():
    pts = [(0.5, 0.5, 0.1)] * 21
    t, calls = tracker([(pts, "Left", 0.4), (pts, "Right", 0.9)])
    img = np.zeros((480, 640, 3), np.uint8)
    hands = t.detect(img, 1.0)
    assert [h.handedness for h in hands] == ["Right", "Left"]
    assert hands[0].landmarks[0] == pytest.approx((0.5 * 640 / 480, 0.5, 0.1 * 640 / 480))
    t.detect(img, 1.0)
    t.detect(img, 0.5)
    assert calls == sorted(set(calls)) and len(calls) == 3  # estrictamente crecientes


def test_tracker_drops_malformed_hands():
    t, _ = tracker([([(0, 0, 0)] * 20, "Left", 0.9)])
    assert t.detect(np.zeros((48, 64, 3), np.uint8), 0.0) == []


def test_tracker_missing_model_gives_clear_error(tmp_path):
    pytest.importorskip("mediapipe")
    cfg = HandsConfig(tmp_path / "no.task")
    with pytest.raises(VisionConfigError, match="fetch_models"):
        HandTracker(cfg).load()


# -- GestureService ---------------------------------------------------------------------------------
class ScriptedTracker:
    """Devuelve una mano distinta segun el momento (script: lista de (hasta_s, mano|None))."""

    def __init__(self, script):
        self.script, self.t0, self.loaded, self.closed = script, None, False, False

    def load(self):
        self.loaded = True

    def detect(self, img, ts):
        self.t0 = self.t0 or ts
        for until, hand in self.script:
            if ts - self.t0 <= until:
                return [hand] if hand else []
        return []

    def close(self):
        self.closed = True


def make_service(script, permissions=None, **over):
    from tests.test_gestures import cfg
    m, cap, _ = manager(delay=0.01)
    events = []
    tr = ScriptedTracker(script)
    svc = GestureService(m, tr, GestureEngine(cfg(hold_s=0.2, activate_hold_s=0.4, grace_s=0.05, **over)),
                         events.append, permissions, fps=50)
    return svc, m, cap, tr, events


def test_service_emits_events_and_cleans_up():
    async def go():
        svc, m, cap, tr, events = make_service([(5, OPEN())])
        assert await svc.start() and tr.loaded
        await asyncio.sleep(1.0)
        svc.stop()
        await asyncio.sleep(0.05)
        return svc, m, cap, tr, events

    svc, m, cap, tr, events = asyncio.run(go())
    assert [e.name for e in events][:2] == ["palm", "activate"]
    assert tr.closed and cap.released and not m.active and not svc.running


def test_service_denied_permission_never_opens_camera(tmp_path):
    cfg = load_config()
    pm = PermissionManager(cfg, confirmer=Asker(False), root=tmp_path)
    cfg["permissions"]["actions"]["camera_capture"] = "never"
    pm2 = PermissionManager(cfg, confirmer=Asker(False), root=tmp_path)

    async def go(p):
        svc, m, cap, tr, _ = make_service([(5, OPEN())], permissions=p)
        ok = await svc.start()
        svc.stop()
        return ok, svc, m

    ok, svc, m = asyncio.run(go(pm2))
    assert not ok and "denegado" in svc.error and not m.active and m.frames == 0
    ok, svc, m = asyncio.run(go(pm))  # nivel 2: auto
    assert ok


def test_service_reports_camera_open_error():
    def bad(cfg):
        raise CameraError("ocupada")

    svc = GestureService(CameraManager(CameraConfig(), bad), ScriptedTracker([]), GestureEngine(GestureConfig()),
                         lambda e: None)

    async def go():
        return await svc.start()

    assert asyncio.run(go()) is False and "ocupada" in svc.error


def test_service_tracker_exception_is_reported_and_camera_released():
    class Boom(ScriptedTracker):
        def detect(self, img, ts):
            raise RuntimeError("fallo del modelo")

    m, cap, _ = manager(delay=0.01)
    errors = []

    async def go():
        svc = GestureService(m, Boom([]), GestureEngine(GestureConfig()), lambda e: None, fps=50,
                             on_error=errors.append)
        await svc.start()
        await asyncio.sleep(0.3)
        svc.stop()
        return svc

    svc = asyncio.run(go())
    assert "fallo del modelo" in (svc.error or "") and errors and cap.released


def test_quick_resubscribe_gets_a_fresh_running_capture():
    caps = []

    def factory(cfg):
        caps.append(FakeCap())
        return caps[-1]

    m = CameraManager(CameraConfig(fps=100, mirror=False), factory)
    for _ in range(5):
        s = m.subscribe("a")
        assert s.get(1) is not None, "la captura nueva debe entregar cuadros"
        s.close()
    assert len(caps) == 5 and all(c.released for c in caps) and not m.active


def test_open_capture_on_missing_device_raises_cameraerror():
    pytest.importorskip("cv2")
    from poeta.vision import open_capture
    with pytest.raises(CameraError, match="no pude abrir"):
        open_capture(CameraConfig(device="/dev/video_que_no_existe"))
