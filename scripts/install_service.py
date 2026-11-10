#!/usr/bin/env python
"""Instala POETA como servicio de usuario supervisado por systemd (arranca con tu sesion grafica y se reinicia si falla).

  python scripts/install_service.py --print            # solo muestra la unidad
  python scripts/install_service.py [--gestures]       # la escribe en ~/.config/systemd/user/poeta.service
  python scripts/install_service.py --uninstall
Despues:  systemctl --user enable --now poeta      |  journalctl --user -u poeta -f
El servicio no tiene teclado: arranca en modo --wake --headless (confirmaciones solo por gestos o denegadas).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UNIT_PATH = Path.home() / ".config" / "systemd" / "user" / "poeta.service"


def render_unit(root: Path, python: Path, gestures: bool = False) -> str:
    args = "voice --wake --headless" + (" --gestures" if gestures else "")
    return f"""[Unit]
Description=P.O.E.T.A. - asistente personal local
Documentation=file://{root}/README.md
After=graphical-session.target pipewire.service
PartOf=graphical-session.target
StartLimitIntervalSec=300
StartLimitBurst=5

[Service]
Type=simple
WorkingDirectory={root}
ExecStart={python} {root}/main.py {args}
Environment=PYTHONUNBUFFERED=1
Restart=on-failure
RestartSec=10
# Las apps que POETA abre (Firefox, Spotify...) cuelgan de este servicio: al reiniciarlo solo se detiene POETA, no ellas.
# (Sin NoNewPrivileges/PrivateTmp/TasksMax a proposito: romperian Flatpak y los navegadores que lanza.)
KillMode=process

[Install]
WantedBy=graphical-session.target
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gestures", action="store_true", help="incluye la camara y los gestos")
    ap.add_argument("--print", action="store_true", dest="show")
    ap.add_argument("--uninstall", action="store_true")
    args = ap.parse_args()
    if args.uninstall:
        subprocess.run(["systemctl", "--user", "disable", "--now", "poeta"], check=False)
        UNIT_PATH.unlink(missing_ok=True)
        subprocess.run(["systemctl", "--user", "daemon-reload"], check=False)
        print("Servicio desinstalado.")
        return 0
    python = ROOT / ".venv" / "bin" / "python"
    if not python.exists():
        python = Path(sys.executable)
    unit = render_unit(ROOT, python, args.gestures)
    if args.show:
        print(unit)
        return 0
    UNIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    UNIT_PATH.write_text(unit, encoding="utf-8")
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=False)
    print(f"Escrito {UNIT_PATH}\nPara arrancarlo:  systemctl --user enable --now poeta\nVer el registro:  journalctl --user -u poeta -f")
    return 0


if __name__ == "__main__":
    sys.exit(main())
