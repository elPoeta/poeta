"""Protocolo local panel<->agente: un objeto JSON por linea (UTF-8), maximo 1 MiB por mensaje.

Servidor -> cliente:  {"t":"hello","v":1}  {"t":"snapshot","data":{...}}  {"t":"event","kind":..,"data":{..},"ts":..}
                      {"t":"reply","id":N,"ok":true,"data":..} | {"t":"reply","id":N,"ok":false,"error":".."}
Cliente -> servidor:  {"t":"cmd","id":N,"name":"..","args":{..}}
Los eventos no llevan lo que dices ni lo que responde el modelo; el contenido (historial, memoria) solo sale por `query`.
Unica excepcion: el evento `permission` (phase=request) incluye una descripcion corta de la accion que pide permiso.
"""

from __future__ import annotations

import json
from typing import Any

VERSION = 1
MAX_LINE = 1_048_576


class ProtocolError(ValueError):
    pass


def encode(obj: dict[str, Any]) -> bytes:
    data = json.dumps(obj, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8") + b"\n"
    if len(data) > MAX_LINE:
        raise ProtocolError("mensaje demasiado grande")
    return data


def decode(line: bytes) -> dict[str, Any]:
    if len(line) > MAX_LINE:
        raise ProtocolError("mensaje demasiado grande")
    try:
        obj = json.loads(line.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"JSON invalido: {exc}") from exc
    if not isinstance(obj, dict) or not isinstance(obj.get("t"), str):
        raise ProtocolError("el mensaje debe ser un objeto con campo 't'")
    return obj
