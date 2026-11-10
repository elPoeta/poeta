#!/usr/bin/env python
"""Prueba los portales XDG en tu escritorio, sin POETA: atajo global y captura de pantalla.

  python scripts/portal_probe.py            # comprueba el portal, registra el atajo y espera 20 s a que lo pulses
  python scripts/portal_probe.py --shot     # ademas toma una captura (KDE puede pedirte permiso)
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poeta.portals import PortalBus, PortalError, PortalsConfig, ShortcutService  # noqa: E402
from poeta.portals.screenshot import png_size, uri_to_path  # noqa: E402


async def main(shot: bool, wait: float) -> int:
    portal = PortalBus()
    try:
        await portal.connect()
        for iface in ("GlobalShortcuts", "Screenshot"):
            ver = await portal.call("org.freedesktop.DBus.Properties", "Get", "ss",
                                    [f"org.freedesktop.portal.{iface}", "version"])
            print(f"portal {iface}: version {ver[0]}")
    except PortalError as exc:
        print(f"Portal no disponible: {exc}")
        print("Revisa: systemctl --user status xdg-desktop-portal xdg-desktop-portal-kd ; y  uv pip install dbus-fast")
        return 1
    cfg = PortalsConfig(enabled=True)
    hits: list[str] = []
    svc = ShortcutService(portal, cfg, {"talk": lambda: hits.append("talk"), "stop": lambda: hits.append("stop")})
    print("Registrando atajos (KDE puede mostrar un dialogo)...")
    if await svc.start():
        print(f"OK: atajos {svc.bound}. Pulsa {cfg.shortcuts['talk'][1]} en los proximos {wait:.0f} s...")
        for _ in range(int(wait * 10)):
            await asyncio.sleep(0.1)
            if hits:
                break
        print(f"Atajo recibido: {hits}" if hits else "No llego ninguna pulsacion (revisa Preferencias del sistema -> Atajos).")
        await svc.stop()
    else:
        print(f"Atajos NO disponibles: {svc.error}")
    if shot:
        print("Pidiendo una captura de pantalla...")
        try:
            code, res = await portal.request("org.freedesktop.portal.Screenshot", "Screenshot", "s", [""],
                                             {"interactive": __import__("dbus_fast").Variant("b", False)}, timeout=60)
        except PortalError as exc:
            print(f"Captura fallo: {exc}")
        else:
            if code == 0 and res.get("uri"):
                p = uri_to_path(res["uri"])
                print(f"OK: {p} {png_size(p) or ''}")
            else:
                print(f"Captura no realizada (codigo {code}: 1 = cancelada).")
    await portal.close()
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shot", action="store_true")
    ap.add_argument("--wait", type=float, default=20)
    a = ap.parse_args()
    raise SystemExit(asyncio.run(main(a.shot, a.wait)))
