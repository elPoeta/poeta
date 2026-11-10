#!/usr/bin/env python
"""Muestra EXACTAMENTE que devuelve kdotool en tu KWin (para diagnosticar list_windows / window_action).
Solo lee e intenta traer al frente la ventana activa (inofensivo).   python scripts/kdotool_probe.py"""
from __future__ import annotations

import asyncio
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from poeta.automation.runner import run_cmd  # noqa: E402


async def show(argv: list[str], timeout: float = 15) -> str:
    res = await run_cmd(argv, timeout)
    print(f"$ {' '.join(argv)}\n   rc={res.rc}  stdout={res.out.strip()[:300]!r}  stderr={res.err.strip()[:300]!r}")
    return res.out


async def main() -> int:
    if not shutil.which("kdotool"):
        print("kdotool no esta instalado.")
        return 1
    await show(["kdotool", "--version"])
    out = await show(["kdotool", "search", "--name", "."])
    ids = [ln.strip() for ln in out.splitlines() if ln.strip()][:3]
    for i in ids:
        await show(["kdotool", "getwindowname", i])
    await show(["kdotool", "search", "--name", ".", "getwindowname"])
    active = (await show(["kdotool", "getactivewindow"])).strip()
    if active:
        await show(["kdotool", "getwindowname", active])
        await show(["kdotool", "windowactivate", active])
    print("\nPasame esta salida completa si list_windows o window_action siguen fallando.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
