"""Registro de auditoria append-only con cadena de hashes (plan secc. 19, punto 9).

Cada entrada incluye el hash de la anterior. Editar o borrar una entrada intermedia rompe la
cadena y `verify_chain` lo detecta. Limites honestos: es evidencia de manipulacion, no
proteccion. Alguien con acceso de escritura al archivo podria reescribir toda la cadena, y
truncar las ultimas entradas no se detecta.

Que NO se registra: el contenido de mensajes, archivos o resultados de herramientas. Solo
metadatos (proveedor, modelo, tamanos, huella sha256 del contenido). Los secretos que aparezcan
en los campos se redactan.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

GENESIS = "0" * 64
RESERVED = {"seq", "ts", "event", "prev", "hash"}
MAX_STR = 300
MAX_ITEMS = 50

_SECRET_KEY = re.compile(r"(api[_-]?key|secret|password|passwd|authorization|credential|(^|_)tokens?$)", re.I)
_SECRET_VALUE = re.compile(r"(nvapi-[A-Za-z0-9_\-]{8,}|sk-[A-Za-z0-9_\-]{16,}|Bearer\s+[A-Za-z0-9._\-]{8,})")


class AuditError(Exception):
    """El registro de auditoria esta danado o no se puede escribir."""


def redact(value: Any, _depth: int = 0) -> Any:
    """Copia segura para registrar: oculta secretos, acota cadenas y listas."""
    if _depth > 6:
        return "..."
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in value.items():
            key = str(k)
            if _SECRET_KEY.search(key) and not isinstance(v, (int, float, bool, type(None))):
                out[key] = "***"
            else:
                out[key] = redact(v, _depth + 1)
        return out
    if isinstance(value, (list, tuple)):
        items = [redact(v, _depth + 1) for v in list(value)[:MAX_ITEMS]]
        if len(value) > MAX_ITEMS:
            items.append(f"...(+{len(value) - MAX_ITEMS} elementos)")
        return items
    if isinstance(value, str):
        text = _SECRET_VALUE.sub("***", value)
        if len(text) > MAX_STR:
            text = text[:MAX_STR] + f"...(+{len(text) - MAX_STR} caracteres)"
        return text
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return redact(str(value), _depth + 1)


def audit_path(cfg: dict[str, Any], root: Path | str) -> Path:
    rel = ((cfg["permissions"].get("audit") or {}).get("file")) or "logs/audit.jsonl"
    p = Path(rel)
    return p if p.is_absolute() else Path(root) / p


def _entry_hash(entry: dict[str, Any]) -> str:
    body = {k: v for k, v in entry.items() if k != "hash"}
    raw = json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _scan(path: Path) -> tuple[bool, int, str | None, str]:
    """Recorre el archivo: (integro, entradas verificadas, error, hash de la ultima entrada)."""
    prev, verified = GENESIS, 0
    if not path.exists():
        return True, 0, None, prev
    with path.open(encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                return False, verified, f"linea {lineno}: JSON invalido (escritura interrumpida o archivo alterado)", prev
            if not isinstance(entry, dict):
                return False, verified, f"linea {lineno}: la entrada no es un objeto", prev
            expected = verified + 1
            if entry.get("seq") != expected:
                return False, verified, f"linea {lineno}: secuencia {entry.get('seq')} en lugar de {expected} (posible entrada borrada)", prev
            if entry.get("prev") != prev:
                return False, verified, f"linea {lineno}: la cadena de hashes esta rota", prev
            if entry.get("hash") != _entry_hash(entry):
                return False, verified, f"linea {lineno}: el contenido fue modificado", prev
            prev, verified = entry["hash"], verified + 1
    return True, verified, None, prev


def verify_chain(path: Path | str) -> tuple[bool, int, str | None]:
    """Comprueba la integridad de la cadena. Devuelve (integro, entradas verificadas, error)."""
    ok, n, err, _ = _scan(Path(path))
    return ok, n, err


def read_tail(path: Path | str, n: int = 20) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    entries: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries[-n:] if n > 0 else entries


class AuditLog:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        ok, verified, err, last = _scan(self.path)
        if not ok:
            raise AuditError(
                f"Registro de auditoria danado ({self.path}): {err}. "
                f"Revisalo con 'python main.py audit --verify'; para continuar mueve el archivo a otro nombre."
            )
        self._seq, self._prev = verified, last

    @classmethod
    def from_config(cls, cfg: dict[str, Any], root: Path | str) -> "AuditLog":
        return cls(audit_path(cfg, root))

    def record(self, event: str, **fields: Any) -> dict[str, Any]:
        clash = RESERVED & set(fields)
        if clash:
            raise ValueError(f"Campos reservados en la auditoria: {sorted(clash)}")
        with self._lock:
            entry: dict[str, Any] = {
                "seq": self._seq + 1,
                "ts": datetime.now(timezone.utc).isoformat(),
                "event": event,
                **redact(fields),
                "prev": self._prev,
            }
            entry["hash"] = _entry_hash(entry)
            line = json.dumps(entry, sort_keys=True, ensure_ascii=False) + "\n"
            try:
                fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
                try:
                    os.write(fd, line.encode("utf-8"))
                    os.fsync(fd)
                finally:
                    os.close(fd)
            except OSError as exc:
                raise AuditError(f"No se pudo escribir en {self.path}: {exc}") from exc
            self._seq, self._prev = entry["seq"], entry["hash"]
            return entry
