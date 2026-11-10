#!/usr/bin/env python
"""Agrega a config/permissions.yaml las acciones nuevas que faltan (con su nivel sugerido), sin tocar lo que ya tengas.
Guarda una copia en permissions.yaml.bak. Es idempotente.   python scripts/add_permission_actions.py [--dry-run]"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from poeta.security.permissions import SUGGESTED_ACTIONS  # noqa: E402

COMMENTS = {
    "camera_capture": "abrir la webcam (gestos/vision)", "desktop_notify": "mostrar una notificacion",
    "desktop_read": "leer volumen, reproductores, ventanas", "desktop_control": "volumen y multimedia",
    "window_control": "traer al frente, minimizar o cerrar ventanas", "service_read": "estado de servicios de usuario",
    "service_control": "iniciar/detener servicios (ademas: desktop.services.allowlist)",
    "process_read": "listar procesos propios", "process_stop": "pedir a un proceso propio que termine (SIGTERM)",
    "run_automation": "ejecutar una rutina de config/automations.yaml",
    "screen_capture": "captura de pantalla por el portal XDG (se guarda en tu equipo)",
}


def add_actions(text: str) -> tuple[str, list[str]]:
    actions = (yaml.safe_load(text) or {}).get("actions") or {}
    missing = [k for k in SUGGESTED_ACTIONS if k not in actions]
    if not missing:
        return text, []
    lines = text.splitlines(keepends=True)
    idx = next((i for i, ln in enumerate(lines) if re.match(r"^  privileged\s*:", ln)), None)
    if idx is None:  # sin `privileged`: al final del bloque actions
        start = next(i for i, ln in enumerate(lines) if re.match(r"^actions\s*:", ln))
        idx = start + 1
        while idx < len(lines) and (lines[idx].startswith(" ") or not lines[idx].strip()):
            idx += 1
    new = [f"  {k}: {SUGGESTED_ACTIONS[k]}  # {COMMENTS.get(k, '')}\n".replace("  # \n", "\n") for k in missing]
    return "".join(lines[:idx] + new + lines[idx:]), missing


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    path = ROOT / "config" / "permissions.yaml"
    text = path.read_text(encoding="utf-8")
    out, missing = add_actions(text)
    if not missing:
        print("permissions.yaml ya tiene todas las acciones.")
        return 0
    yaml.safe_load(out)  # nunca escribir algo invalido
    print("Faltaban: " + ", ".join(f"{k}: {SUGGESTED_ACTIONS[k]}" for k in missing))
    if ap.parse_args().dry_run:
        return 0
    path.with_suffix(".yaml.bak").write_text(text, encoding="utf-8")
    path.write_text(out, encoding="utf-8")
    print("Listo (copia en permissions.yaml.bak).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
