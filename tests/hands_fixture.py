"""Manos sinteticas (21 landmarks) para probar la clasificacion sin camara ni modelo."""
from poeta.vision.hands import Hand

ASPECT = 4 / 3
# nudillos (mcp) en y=0.6, muñeca en (0.5, 0.8)
MCP = {"index": 0.45, "middle": 0.50, "ring": 0.55, "pinky": 0.60}
BASE = {"index": 5, "middle": 9, "ring": 13, "pinky": 17}


def make_hand(index="ext", middle="ext", ring="ext", pinky="ext", thumb="tucked", cx=0.0, cy=0.0,
              tip_ext=0.40, tip_curl=0.67) -> Hand:
    pts = [(0.5, 0.8, 0.0)] * 21
    pts = list(pts)
    for name, state in (("index", index), ("middle", middle), ("ring", ring), ("pinky", pinky)):
        b, x = BASE[name], MCP[name]
        pts[b] = (x, 0.6, 0.0)
        pts[b + 1] = (x, 0.5 if state == "ext" else 0.55, 0.0)
        pts[b + 2] = (x, 0.45 if state == "ext" else 0.6, 0.0)
        pts[b + 3] = (x, tip_ext if state == "ext" else tip_curl, 0.0)
    pts[1], pts[2], pts[3] = (0.42, 0.78, 0.0), (0.40, 0.72, 0.0), (0.40, 0.6, 0.0)
    tips = {"up": (0.40, 0.45), "down": (0.40, 0.98), "tucked": (0.50, 0.65), "side": (0.20, 0.72)}
    pts[4] = (*tips[thumb], 0.0)
    pts = [((x + cx) * ASPECT, y + cy, z) for x, y, z in pts]
    return Hand(tuple(pts), "Right", 0.9, ASPECT)


OPEN = lambda **k: make_hand(thumb=k.pop("thumb", "side"), **k)  # noqa: E731
FIST = lambda **k: make_hand("curl", "curl", "curl", "curl", **k)  # noqa: E731
THUMB_UP = lambda **k: make_hand("curl", "curl", "curl", "curl", thumb="up", **k)  # noqa: E731
THUMB_DOWN = lambda **k: make_hand("curl", "curl", "curl", "curl", thumb="down", **k)  # noqa: E731
POINT = lambda **k: make_hand("ext", "curl", "curl", "curl", **k)  # noqa: E731
