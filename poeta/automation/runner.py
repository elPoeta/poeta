"""Ejecucion de comandos externos del escritorio: argv (sin shell), entorno minimo, tiempo y salida acotados."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Awaitable, Callable

# Solo lo necesario para hablar con el escritorio del usuario: nunca claves ni el resto del entorno.
SAFE_ENV = ("PATH", "HOME", "LANG", "LC_ALL", "XDG_RUNTIME_DIR", "DBUS_SESSION_BUS_ADDRESS", "WAYLAND_DISPLAY",
            "DISPLAY", "XDG_SESSION_TYPE", "XDG_CURRENT_DESKTOP", "XDG_DATA_DIRS", "PIPEWIRE_RUNTIME_DIR",
            "KDE_FULL_SESSION")


@dataclass
class CmdResult:
    rc: int
    out: str
    err: str


Runner = Callable[[list[str], float], Awaitable[CmdResult]]


def safe_env() -> dict[str, str]:
    return {k: os.environ[k] for k in SAFE_ENV if k in os.environ}


async def run_cmd(argv: list[str], timeout: float = 10.0, max_bytes: int = 100_000) -> CmdResult:
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv, stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            env=safe_env())
    except FileNotFoundError:
        return CmdResult(127, "", f"no se encontro '{argv[0]}'")
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return CmdResult(124, "", f"'{argv[0]}' no respondio en {timeout:.0f} s")
    return CmdResult(proc.returncode or 0, out[:max_bytes].decode("utf-8", "replace"),
                     err[:max_bytes].decode("utf-8", "replace"))
