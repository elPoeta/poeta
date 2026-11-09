#!/usr/bin/env python
"""Descarga el modelo de MediaPipe para manos (hand_landmarker.task, ~7.5 MB) a models/. Solo se baja una vez."""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
DEST = ROOT / "models" / "hand_landmarker.task"


def main() -> int:
    if DEST.is_file() and DEST.stat().st_size > 1_000_000:
        print(f"Ya existe: {DEST} ({DEST.stat().st_size / 1e6:.1f} MB)")
        return 0
    DEST.parent.mkdir(exist_ok=True)
    tmp = DEST.with_suffix(".part")
    print(f"Descargando {URL}")
    try:
        with urllib.request.urlopen(URL, timeout=60) as r, open(tmp, "wb") as f:
            while chunk := r.read(1 << 16):
                f.write(chunk)
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        print(f"Error de descarga: {exc}\nBajalo a mano desde esa URL y guardalo como {DEST}", file=sys.stderr)
        return 1
    if tmp.stat().st_size < 1_000_000:
        tmp.unlink()
        print("El archivo descargado es demasiado chico: no parece un modelo valido.", file=sys.stderr)
        return 1
    tmp.replace(DEST)
    print(f"Listo: {DEST} ({DEST.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
