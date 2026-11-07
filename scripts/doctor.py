#!/usr/bin/env python3
"""Diagnostico del entorno de P.O.E.T.A. (Fase 0).

Comprueba herramientas, rutas y configuracion. Solo lee: no instala ni modifica nada.
FAIL = bloquea la Fase 1. WARN = necesario mas adelante o no verificable aqui.
Codigo de salida 1 si hay algun FAIL.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from poeta.config import ConfigError, load_config  # noqa: E402

results: list[tuple[str, str, str]] = []


def record(level: str, name: str, detail: str = "") -> None:
    results.append((level, name, detail))


def check_tool(name: str, level_if_missing: str, hint: str) -> None:
    path = shutil.which(name)
    if path:
        record("OK", name, path)
    else:
        record(level_if_missing, name, f"no encontrado. {hint}")


def main() -> int:
    # Python
    v = sys.version_info
    if (v.major, v.minor) >= (3, 12):
        record("OK", "python", f"{v.major}.{v.minor}.{v.micro}")
    else:
        record("FAIL", "python", f"{v.major}.{v.minor} (se requiere 3.12 o superior)")

    # Configuracion
    cfg = None
    try:
        cfg = load_config()
        record("OK", "configuracion", "YAML cargados e invariantes de privacidad validos")
    except ConfigError as exc:
        record("FAIL", "configuracion", str(exc))

    # Herramientas
    check_tool("uv", "FAIL", "Necesario para el entorno (plan secc. 16).")
    check_tool("nvidia-smi", "WARN", "Sin driver NVIDIA no hay GPU para el LLM.")
    check_tool("ffmpeg", "WARN", "Usado para audio/video.")
    check_tool("espeak-ng", "WARN", "Necesario para Kokoro: sudo dnf install espeak-ng")
    check_tool("pw-cli", "WARN", "PipeWire para el Audio Manager.")
    check_tool("git", "WARN", "")

    # Rutas del entorno de IA existente
    if cfg:
        paths = cfg["settings"]["paths"]
        for key in ("ai_home", "models", "voices", "output", "cache"):
            p = Path(paths[key])
            record("OK" if p.is_dir() else "WARN", f"ruta {key}", str(p) + ("" if p.is_dir() else " (no existe)"))

        llama_dir = Path(paths["llama_cpp"])
        if llama_dir.is_dir():
            found = sorted(llama_dir.rglob("llama-server"))
            found = [f for f in found if f.is_file()]
            record("OK" if found else "WARN", "llama-server",
                   str(found[0]) if found else f"no encontrado bajo {llama_dir}")
        else:
            record("WARN", "llama.cpp", f"{llama_dir} no existe")

        free_gb = shutil.disk_usage(ROOT).free / 1e9
        record("OK" if free_gb > 20 else "WARN", "espacio libre", f"{free_gb:.0f} GB")

    # Paquetes opcionales de fases posteriores (informativo)
    for mod, phase in (("faster_whisper", "Fase 4"), ("kokoro_onnx", "Fase 4"),
                       ("openwakeword", "Fase 4"), ("cv2", "Fase 5"),
                       ("mediapipe", "Fase 5"), ("mcp", "Fase 6")):
        try:
            __import__(mod)
            record("OK", f"paquete {mod}", "instalado")
        except ImportError:
            record("WARN", f"paquete {mod}", f"no instalado (se necesita en {phase})")

    width = max(len(name) for _, name, _ in results)
    for level, name, detail in results:
        print(f"[{level:<4}] {name:<{width}}  {detail}")

    fails = sum(1 for lvl, _, _ in results if lvl == "FAIL")
    warns = sum(1 for lvl, _, _ in results if lvl == "WARN")
    print(f"\n{fails} FAIL, {warns} WARN")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
