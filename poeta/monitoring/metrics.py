"""Metricas por turno (latencias, RAM, GPU) en logs/metrics.jsonl. Nunca guarda texto del usuario ni del modelo."""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger("poeta.metrics")

FIELDS = ("stt_s", "reasoning_s", "first_text_s", "first_audio_s", "total_s")


def rss_mb(statm: str = "/proc/self/statm") -> float | None:
    try:
        pages = int(Path(statm).read_text().split()[1])
        return round(pages * os.sysconf("SC_PAGE_SIZE") / 1e6, 1)
    except (OSError, ValueError, IndexError):
        return None


def gpu_sample(runner: Callable[..., Any] = subprocess.run, which: Callable[[str], str | None] = shutil.which,
               timeout: float = 2.0) -> dict[str, float] | None:
    """Uso de la GPU NVIDIA (MiB usados/totales, % de uso, temperatura). None si no hay nvidia-smi."""
    if not which("nvidia-smi"):
        return None
    try:
        r = runner(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu",
                    "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=timeout, check=False)
        used, total, util, temp = (float(x) for x in r.stdout.strip().splitlines()[0].split(","))
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None
    return {"gpu_used_mib": used, "gpu_total_mib": total, "gpu_util_pct": util, "gpu_temp_c": temp}


class MetricsLog:
    def __init__(self, path: Path | str, max_bytes: int = 2_000_000):
        self.path, self.max_bytes = Path(path), max_bytes

    def record(self, **fields: Any) -> None:
        row = {"ts": round(time.time(), 1), **{k: v for k, v in fields.items() if v is not None}}
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self.path.exists() and self.path.stat().st_size > self.max_bytes:
                self.path.replace(self.path.with_suffix(".jsonl.1"))  # una sola rotacion: es solo diagnostico
            fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            with os.fdopen(fd, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        except OSError:
            log.exception("no se pudo escribir la metrica")

    def read(self, last: int | None = None) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        try:
            for line in self.path.read_text(encoding="utf-8").splitlines():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        except OSError:
            return []
        return rows[-last:] if last else rows


def percentile(values: list[float], p: float) -> float:
    xs = sorted(values)
    if not xs:
        raise ValueError("sin datos")
    k = (len(xs) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"turns": len(rows), "interrupted": sum(1 for r in rows if r.get("interrupted"))}
    for f in FIELDS:
        vals = [float(r[f]) for r in rows if isinstance(r.get(f), (int, float))]
        if vals:
            out[f] = {"n": len(vals), "p50": round(percentile(vals, 50), 2), "p95": round(percentile(vals, 95), 2),
                      "max": round(max(vals), 2)}
    for f in ("rss_mb", "gpu_used_mib"):
        vals = [float(r[f]) for r in rows if isinstance(r.get(f), (int, float))]
        if vals:
            out[f] = {"first": vals[0], "last": vals[-1], "max": max(vals)}
    return out


class Sampler:
    """Muestra RAM del proceso y GPU; la GPU se cachea `ttl` segundos para no lanzar nvidia-smi en cada turno."""

    def __init__(self, ttl: float = 10.0, gpu: Callable[[], dict[str, float] | None] = gpu_sample,
                 clock: Callable[[], float] = time.monotonic):
        self.ttl, self._gpu, self._clock = ttl, gpu, clock
        self._at, self._last = -1e9, None

    def __call__(self) -> dict[str, float]:
        out: dict[str, float] = {}
        r = rss_mb()
        if r is not None:
            out["rss_mb"] = r
        if self._clock() - self._at >= self.ttl:
            self._last, self._at = self._gpu(), self._clock()
        if self._last:
            out.update(self._last)
        return out
