#!/usr/bin/env python
"""Identifica tus camaras: abre cada /dev/video*, lee unos cuadros y dice cual parece la webcam RGB y cual el IR.

  python scripts/camera_probe.py            # todas
  python scripts/camera_probe.py --save DIR # guarda un JPG de cada una para verlas
"""
from __future__ import annotations

import argparse
import glob
import sys
import time
from pathlib import Path

try:
    import cv2
    import numpy as np
except ImportError:
    sys.exit("Falta opencv: uv pip install -e '.[vision]'")


def probe(dev: str, save: Path | None) -> None:
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"{dev}: no se pudo abrir (en uso, sin permiso o nodo de metadatos)")
        return
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        frames, t0, last = 0, time.monotonic(), None
        while time.monotonic() - t0 < 2.0:
            ok, img = cap.read()
            if ok and img is not None:
                frames += 1
                last = img
        if last is None:
            print(f"{dev}: abre pero no entrega cuadros (no es una camara de video)")
            return
        h, w = last.shape[:2]
        b, g, r = (last[:, :, i].astype(int) for i in range(3))
        color_diff = float(np.abs(b - g).mean() + np.abs(g - r).mean())
        kind = "monocromo (posible IR)" if color_diff < 2.0 else "color (webcam RGB)"
        print(f"{dev}: {w}x{h}, ~{frames / 2.0:.0f} fps, brillo medio {last.mean():.0f}/255, {kind}")
        if save:
            save.mkdir(parents=True, exist_ok=True)
            out = save / f"{Path(dev).name}.jpg"
            cv2.imwrite(str(out), last)
            print(f"   guardado en {out}")
    finally:
        cap.release()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", type=Path, help="carpeta donde guardar un JPG por camara")
    args = ap.parse_args()
    devs = sorted(glob.glob("/dev/video*"))
    if not devs:
        print("No hay dispositivos /dev/video*")
        return 1
    for d in devs:
        probe(d, args.save)
    print("\nPon en config/vision.yaml camera.device la que sea color y quede 'ok'.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
