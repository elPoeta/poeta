#!/usr/bin/env python
"""Crea ~/.local/share/applications/<app_id>.desktop para que el portal/KDE reconozcan a POETA al registrar atajos.

  python scripts/install_portals.py            # instala
  python scripts/install_portals.py --remove   # desinstala
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from poeta.config import load_config  # noqa: E402
from poeta.portals import PortalsConfig  # noqa: E402


def render(app_id: str, root: Path) -> str:
    return (f"[Desktop Entry]\nType=Application\nName=POETA\nComment=Asistente personal local\n"
            f"Exec={root}/.venv/bin/python {root}/main.py voice --wake\nTerminal=true\nNoDisplay=true\n"
            f"Icon=audio-input-microphone\nX-KDE-StartupNotify=false\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--remove", action="store_true")
    a = ap.parse_args()
    app_id = PortalsConfig.from_config(load_config()).app_id
    path = Path.home() / ".local/share/applications" / f"{app_id}.desktop"
    if a.remove:
        path.unlink(missing_ok=True)
        print(f"Eliminado {path}")
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(app_id, ROOT), encoding="utf-8")
    print(f"Instalado {path}\nAhora activa `enabled: true` en config/portals.yaml.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
