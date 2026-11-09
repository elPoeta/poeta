#!/usr/bin/env python
"""Prueba en vivo del reconocimiento de gestos (sin ventana: texto en la terminal).

  python scripts/gesture_test.py                 # muestra la pose por cuadro y los eventos
  python scripts/gesture_test.py --calibrate     # ajusta los umbrales de dedos a tu mano y los guarda
  python scripts/gesture_test.py --seconds 60
Prerrequisitos: uv pip install -e '.[vision]'  y  python scripts/fetch_models.py
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from poeta.config import ConfigError, load_config  # noqa: E402
from poeta.gestures import (GestureConfig, GestureEngine, GestureService, compute_calibration,  # noqa: E402
                            load_calibration, mean_finger_ratio, save_calibration)
from poeta.vision import CameraConfig, CameraManager, VisionConfigError  # noqa: E402
from poeta.vision.hands import HandsConfig, HandTracker  # noqa: E402


async def collect(svc: GestureService, samples: list[float], seconds: float) -> None:
    t0 = time.monotonic()
    while time.monotonic() - t0 < seconds:
        await asyncio.sleep(0.1)


async def run(args: argparse.Namespace) -> int:
    try:
        cfg = load_config()
        cam_cfg, hands_cfg, g_cfg = (CameraConfig.from_config(cfg), HandsConfig.from_config(cfg, ROOT),
                                     GestureConfig.from_config(cfg))
    except (ConfigError, VisionConfigError) as exc:
        print(f"Error de configuracion: {exc}", file=sys.stderr)
        return 1
    cal_path = ROOT / g_cfg.calibration_file
    th = load_calibration(cal_path, g_cfg.thresholds)
    engine = GestureEngine(g_cfg, th)
    camera = CameraManager(cam_cfg)
    samples: list[float] = []
    state = {"collect": False, "last": 0.0}

    def on_frame(frame, hand, pose):
        if hand is not None and state["collect"]:
            samples.append(mean_finger_ratio(hand))
        if not state["collect"] and time.monotonic() - state["last"] > 0.25:
            state["last"] = time.monotonic()
            txt = f"mano: {pose:<11} ratio dedos {mean_finger_ratio(hand):.2f}" if hand else "mano: (no se ve)"
            print(f"\r{txt}   armado={'si' if engine.armed else 'no'}   ", end="", flush=True)

    def on_event(ev):
        print(f"\n>>> {ev.name}")

    svc = GestureService(camera, HandTracker(hands_cfg), engine, on_event, fps=hands_cfg.fps,
                         on_error=lambda m: print(f"\nError: {m}"), on_frame=on_frame)
    if not await svc.start():
        print(f"No pude arrancar: {svc.error}", file=sys.stderr)
        return 1
    try:
        if args.calibrate:
            result = {}
            for key, msg in (("open", "Mostra la mano ABIERTA, dedos separados, mirando a la camara"),
                             ("closed", "Ahora hace un PUÑO")):
                input(f"\n{msg}. Enter para empezar (3 s)... ")
                samples.clear()
                state["collect"] = True
                await collect(svc, samples, 3.0)
                state["collect"] = False
                result[key] = list(samples)
                print(f"  {len(samples)} muestras")
            try:
                new = compute_calibration(result["open"], result["closed"], g_cfg.thresholds)
            except ValueError as exc:
                print(f"No se pudo calibrar: {exc}", file=sys.stderr)
                return 1
            save_calibration(cal_path, new)
            print(f"Guardado en {cal_path}: dedo extendido >= {new.finger_extended}, doblado <= {new.finger_curled}")
            return 0
        print(f"Probando {args.seconds} s. Mano abierta 1.5 s = activar; luego pulgar arriba/abajo, indice, deslizar. Ctrl+C sale.")
        await asyncio.sleep(args.seconds)
    except KeyboardInterrupt:
        pass
    finally:
        svc.stop()
        camera.close()
        print()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--seconds", type=float, default=120)
    try:
        return asyncio.run(run(ap.parse_args()))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
