#!/usr/bin/env python
"""Comprueba en TU escritorio que cada utilidad de la Fase 7 existe y responde (solo comandos de lectura)."""
from __future__ import annotations

import asyncio
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from poeta.automation.runner import run_cmd  # noqa: E402

CHECKS = [
    ("notify-send", "notify", ["notify-send", "--version"], "sudo dnf install libnotify"),
    ("playerctl", "media_status/media_control", ["playerctl", "--version"], "sudo dnf install playerctl"),
    ("wpctl", "get_volume/set_volume", ["wpctl", "get-volume", "@DEFAULT_AUDIO_SINK@"], "sudo dnf install wireplumber"),
    ("kdotool", "list_windows/window_action", ["kdotool", "search", "--name", "."], "sudo dnf install kdotool (o ver su repo)"),
    ("systemctl", "service_status/service_control", ["systemctl", "--user", "is-system-running"], ""),
    ("ps", "list_processes", ["ps", "-o", "pid=", "-p", "1"], ""),
]


async def main() -> int:
    bad = 0
    for binary, tools, argv, hint in CHECKS:
        if not shutil.which(binary):
            print(f"[FALTA] {binary:<12} ({tools})  {hint}")
            bad += 1
            continue
        res = await run_cmd(argv, 8)
        ok = res.rc == 0 or binary == "systemctl"
        detail = (res.out or res.err).strip().splitlines()[:1]
        print(f"[{'OK' if ok else 'FALLA'}] {binary:<12} ({tools})  {detail[0][:70] if detail else ''}")
        bad += not ok
    print("\nLas herramientas con utilidad ausente no se registran; el resto funciona igual.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
