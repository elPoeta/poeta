import pytest

from poeta.config import ConfigError, load_config
from poeta.gestures import (GestureConfig, GestureEngine, Thresholds, classify, compute_calibration,
                            load_calibration, mean_finger_ratio, save_calibration)
from poeta.vision import VisionConfigError
from tests.hands_fixture import FIST, OPEN, POINT, THUMB_DOWN, THUMB_UP, make_hand

TH = Thresholds()


@pytest.mark.parametrize("hand, pose", [
    (OPEN(), "open_palm"), (OPEN(thumb="tucked"), "open_palm"), (FIST(), "fist"), (THUMB_UP(), "thumb_up"),
    (THUMB_DOWN(), "thumb_down"), (POINT(), "point"), (POINT(thumb="up"), "point"),
    (make_hand("ext", "ext", "curl", "curl"), "none"), (make_hand("curl", "curl", "curl", "curl", thumb="side"), "none"),
])
def test_classify(hand, pose):
    assert classify(hand, TH) == pose


def test_classify_is_translation_invariant():
    assert classify(OPEN(cx=0.2, cy=-0.1), TH) == "open_palm"
    assert classify(THUMB_UP(cx=-0.2), TH) == "thumb_up"


def test_ambiguous_fingers_are_not_a_pose():
    # dedos a medio camino (ratio entre los dos umbrales) no deben contar como nada
    mid = make_hand(tip_ext=0.52)  # ratio ~1.4: entre 1.25 y 1.5
    assert classify(mid, TH) == "none"


def cfg(**over):
    base = dict(hold_s=0.6, activate_hold_s=1.5, armed_s=10, require_arming=True, cooldown_s=1.0, grace_s=0.2)
    base.update(over)
    return GestureConfig(**base)


def drive(engine, hand_fn, t0, t1, dt=0.05):
    out, t = [], t0
    while t <= t1 + 1e-9:
        out += [(e.name, round(e.ts, 2)) for e in engine.update(hand_fn(t) if hand_fn else None, t)]
        t += dt
    return out


def names(ev):
    return [n for n, _ in ev]


def test_palm_fires_once_at_hold_and_activate_later():
    e = GestureEngine(cfg())
    ev = drive(e, lambda t: OPEN(), 0, 3.0)
    assert names(ev) == ["palm", "activate"]
    assert 0.6 <= ev[0][1] <= 0.7 and 1.5 <= ev[1][1] <= 1.6 and e.armed


def test_short_palm_does_nothing():
    assert drive(GestureEngine(cfg()), lambda t: OPEN(), 0, 0.5) == []


def test_thumbs_ignored_until_armed():
    e = GestureEngine(cfg())
    assert drive(e, lambda t: THUMB_UP(), 0, 2.0) == []
    e2 = GestureEngine(cfg(require_arming=False))
    assert names(drive(e2, lambda t: THUMB_UP(), 0, 2.0)) == ["thumb_up"]


def armed_engine(**over):
    e = GestureEngine(cfg(**over))
    drive(e, lambda t: OPEN(), 0, 1.7)
    assert e.armed
    return e


def test_armed_thumb_up_and_down_with_hold_and_single_fire():
    e = armed_engine()
    ev = drive(e, lambda t: THUMB_UP(), 1.8, 4.0)
    assert names(ev) == ["thumb_up"]  # no repite mientras se sostiene
    ev = drive(e, lambda t: THUMB_DOWN(), 4.05, 6.0)
    assert names(ev) == ["thumb_down"]


def test_point_event():
    e = armed_engine()
    assert names(drive(e, lambda t: POINT(), 1.8, 3.0)) == ["point"]


def test_flicker_within_grace_does_not_reset_hold():
    e = armed_engine()
    seq = lambda t: None if abs(t - 2.2) < 0.06 else THUMB_UP()  # un cuadro perdido  # noqa: E731
    assert names(drive(e, seq, 1.8, 3.0)) == ["thumb_up"]


def test_long_dropout_resets_hold():
    e = armed_engine()
    seq = lambda t: None if 2.0 <= t <= 2.5 else THUMB_UP()  # noqa: E731
    ev = drive(e, seq, 1.8, 2.4)
    assert ev == []
    ev = drive(e, seq, 2.45, 3.0)
    assert ev == []  # reinicio: solo lleva ~0.5 s sostenido
    assert names(drive(e, seq, 3.05, 3.5)) == ["thumb_up"]


def test_cooldown_blocks_rapid_second_gesture():
    e = armed_engine(cooldown_s=2.0)
    assert names(drive(e, lambda t: THUMB_UP(), 1.8, 2.6)) == ["thumb_up"]
    assert drive(e, lambda t: POINT(), 2.65, 3.6) == []  # <2 s desde el ultimo
    assert names(drive(e, lambda t: POINT(), 3.65, 4.5)) == ["point"]


def test_arming_expires():
    e = armed_engine(armed_s=3)
    ev = drive(e, lambda t: None, 1.8, 6.0)
    assert names(ev) == ["deactivate"] and not e.armed
    assert drive(e, lambda t: THUMB_UP(), 6.05, 8.0) == []


def test_gesture_extends_arming():
    e = armed_engine(armed_s=3)
    drive(e, lambda t: THUMB_UP(), 1.8, 3.0)  # evento ~2.4: vence ~5.4
    assert drive(e, None, 3.05, 5.0) == [] and e.armed


def swipe(dx_total, dur=0.4, y_drift=0.0, start=2.0):
    def f(t):
        k = min(max(t - start, 0) / dur, 1)
        return OPEN(cx=dx_total * k, cy=y_drift * k)
    return f


def test_swipe_requires_arming_then_detects_direction():
    e = GestureEngine(cfg())
    ev = drive(e, swipe(0.4, dur=0.3, start=0.0), 0, 1.0)  # rapido y sin activar antes
    assert not [n for n in names(ev) if n.startswith("swipe")]
    e = armed_engine()
    ev = drive(e, swipe(0.4), 2.0, 3.0)
    assert "swipe_right" in names(ev)
    e = armed_engine()
    ev = drive(e, swipe(-0.4), 2.0, 3.0)
    assert "swipe_left" in names(ev)


def test_small_or_diagonal_movement_is_not_a_swipe():
    e = armed_engine()
    assert not [n for n in names(drive(e, swipe(0.1), 2.0, 3.0)) if n.startswith("swipe")]
    e = armed_engine()
    assert not [n for n in names(drive(e, swipe(0.4, y_drift=0.4), 2.0, 3.0)) if n.startswith("swipe")]


def test_time_gap_resets_state():
    e = armed_engine()
    e.update(THUMB_UP(), 2.0)
    e.update(THUMB_UP(), 10.0)  # la camara se paro 8 s
    assert not e.armed


# -- calibracion --------------------------------------------------------------------------------------
def test_compute_calibration_midpoints_and_validation():
    th = compute_calibration([2.0] * 9, [1.0] * 9, TH)
    assert th.finger_extended == pytest.approx(1.65) and th.finger_curled == pytest.approx(1.35)
    assert th.thumb_extended == TH.thumb_extended
    with pytest.raises(ValueError, match="muy pocas"):
        compute_calibration([2.0], [1.0], TH)
    with pytest.raises(ValueError, match="casi no se distinguen"):
        compute_calibration([1.3] * 9, [1.2] * 9, TH)
    # mediana: un cuadro raro no arruina la calibracion
    assert compute_calibration([2.0] * 8 + [0.1], [1.0] * 9, TH).finger_extended == pytest.approx(1.65)


def test_mean_finger_ratio_open_vs_fist():
    assert mean_finger_ratio(OPEN()) > 1.8 and mean_finger_ratio(FIST()) < 1.0


def test_calibration_roundtrip_and_bad_files(tmp_path):
    p = tmp_path / "d" / "cal.json"
    th = compute_calibration([2.0] * 9, [1.0] * 9, TH)
    save_calibration(p, th)
    assert (p.stat().st_mode & 0o777) == 0o600
    got = load_calibration(p, TH)
    assert got.finger_extended == th.finger_extended
    p.write_text("{no json")
    assert load_calibration(p, TH) == TH
    p.write_text('{"finger_extended": 1.0, "finger_curled": 1.4}')
    assert load_calibration(p, TH) == TH  # invertidos: se ignora
    assert load_calibration(tmp_path / "nope.json", TH) == TH


# -- configuracion ------------------------------------------------------------------------------------
def test_repo_config_valid_and_defaults_match():
    c = GestureConfig.from_config(load_config())
    assert c.thresholds == TH and c.require_arming


@pytest.mark.parametrize("patch, msg", [
    ({"thresholds": {"finger_curled": 2.0}}, "finger_curled"),
    ({"hold_s": 2.0, "activate_hold_s": 1.0}, "activate_hold_s"),
    ({"hold_s": -1}, "hold_s"),
    ({"require_arming": "si"}, "require_arming"),
    ({"sensitive_actions_require_more_than_gesture": False}, "nunca autoriza"),
])
def test_invalid_gesture_config(patch, msg):
    cfg_ = load_config()
    cfg_["vision"]["gestures"].update(patch)
    with pytest.raises(VisionConfigError, match=msg):
        GestureConfig.from_config(cfg_)
