"""Monitor de recursos: publica cada pocos segundos RAM, carga y GPU en el bus (para el panel y el HUD)."""

from __future__ import annotations

import asyncio
import contextlib
import os
from pathlib import Path
from typing import Any, Callable

from .metrics import Sampler, rss_mb


def system_memory(meminfo: str = "/proc/meminfo") -> dict[str, float]:
    try:
        kb = {}
        for line in Path(meminfo).read_text().splitlines():
            k, _, v = line.partition(":")
            if k in ("MemTotal", "MemAvailable"):
                kb[k] = float(v.split()[0])
        return {"ram_total_mb": round(kb["MemTotal"] / 1024), "ram_used_mb": round((kb["MemTotal"] - kb["MemAvailable"]) / 1024)}
    except (OSError, KeyError, ValueError, IndexError):
        return {}


def read_resources(sampler: Sampler) -> dict[str, Any]:
    out: dict[str, Any] = {**system_memory(), **sampler()}
    try:
        out["load1"] = round(os.getloadavg()[0], 2)
    except OSError:
        pass
    if "rss_mb" not in out and (r := rss_mb()) is not None:
        out["rss_mb"] = r
    return out


class ResourceMonitor:
    def __init__(self, bus: Any, interval_s: float = 2.0, sampler: Sampler | None = None,
                 reader: Callable[[Sampler], dict[str, Any]] = read_resources):
        self.bus, self.interval_s = bus, interval_s
        self.sampler = sampler or Sampler(ttl=interval_s)
        self._reader = reader
        self._task: asyncio.Task | None = None
        self.last: dict[str, Any] = {}

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run())

    async def _run(self) -> None:
        while True:
            try:
                # nvidia-smi bloquea ~50 ms: fuera del bucle de eventos para no tocar el audio
                self.last = await asyncio.to_thread(self._reader, self.sampler)
                self.bus.publish("resources", **self.last)
            except Exception:  # noqa: BLE001 - un fallo de medicion no debe parar nada
                pass
            await asyncio.sleep(self.interval_s)

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
