"""Carga del archivo .env del proyecto (sin dependencias externas).

Reglas:
- Una variable que ya existe (no vacia) en el entorno del shell gana sobre el .env.
- Los valores vacios se ignoran.
- Nunca se imprime ni se devuelve un valor: el resultado solo contiene nombres y numeros de linea.
- Una linea mal formada se informa por su numero (no por su contenido, que podria ser una clave) y se omite.

Formato admitido: CLAVE=valor, export CLAVE=valor, valores entre comillas simples o dobles,
comentarios con # (en linea propia, o tras un espacio al final de la linea).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_INLINE_COMMENT = re.compile(r"\s#")


@dataclass
class EnvLoadResult:
    path: Path
    found: bool = False
    loaded: list[str] = field(default_factory=list)
    kept_existing: list[str] = field(default_factory=list)
    malformed_lines: list[int] = field(default_factory=list)
    insecure_permissions: bool = False
    error: str | None = None


def _parse_value(raw: str) -> str | None:
    """Devuelve el valor, o None si esta mal formado (comilla sin cerrar o texto tras la comilla)."""
    raw = raw.strip()
    if not raw:
        return ""
    if raw[0] in "\"'":
        end = raw.find(raw[0], 1)
        if end == -1:
            return None
        rest = raw[end + 1 :].strip()
        if rest and not rest.startswith("#"):
            return None
        return raw[1:end]
    match = _INLINE_COMMENT.search(raw)
    if match:
        raw = raw[: match.start()]
    return raw.strip()


def load_env_file(path: Path | str) -> EnvLoadResult:
    path = Path(path)
    result = EnvLoadResult(path)
    if not path.is_file():
        return result
    result.found = True
    try:
        result.insecure_permissions = bool(path.stat().st_mode & 0o077)
        text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError) as exc:
        result.error = type(exc).__name__
        return result

    for number, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped[7:].lstrip()
        key, sep, raw_value = stripped.partition("=")
        key = key.strip()
        value = _parse_value(raw_value) if sep else None
        if value is None or not _KEY.match(key):
            result.malformed_lines.append(number)
            continue
        if not value:
            continue
        if os.environ.get(key):
            result.kept_existing.append(key)
            continue
        os.environ[key] = value
        result.loaded.append(key)
    return result
