"""Confirmador de linea de comandos para el Permission Manager.

Muestra exactamente lo que se enviara o ejecutara. Solo una respuesta afirmativa autoriza:
Enter, EOF, Ctrl+C o cualquier otra cosa cancelan. Las solicitudes explicitas exigen escribir CONFIRMAR.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from poeta.security import ActionRequest

PREVIEW_CHARS = 300
YES = {"s", "si", "sí", "y", "yes"}


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict) and p.get("type") == "text":
                parts.append(str(p.get("text", "")))
            else:
                kind = p.get("type", "?") if isinstance(p, dict) else "?"
                parts.append(f"[contenido no textual: {kind}]")
        return " ".join(parts)
    return "" if content is None else str(content)


def render(req: ActionRequest, explicit: bool, full: bool = False) -> str:
    lines = ["", "=" * 64, "SOLICITUD DE AUTORIZACION" + (" (EXPLICITA)" if explicit else ""), req.description]
    lines.append(f"Accion: {req.kind}   Origen: {req.origin}" + ("   [TRAS LEER CONTENIDO NO CONFIABLE]" if req.tainted else ""))
    for key, value in req.details.items():
        lines.append(f"  {key}: {value}")

    payload = req.payload
    if isinstance(payload, dict) and "messages" in payload:
        lines.append("Se enviara este contenido:")
        for m in payload["messages"]:
            text = _text_of(m.get("content"))
            if not full and len(text) > PREVIEW_CHARS:
                text = f"{text[:PREVIEW_CHARS]}... (+{len(text) - PREVIEW_CHARS} caracteres; se enviaran completos, 'v' para verlos)"
            lines.append(f"  [{m.get('role', '?')}] {text}")
        if payload.get("tools"):
            lines.append(f"  (+ {len(payload['tools'])} definiciones de herramientas)")
    elif payload is not None:
        text = json.dumps(payload, ensure_ascii=False, default=str, indent=2)
        if not full and len(text) > 1000:
            text = text[:1000] + f"... (+{len(text) - 1000} caracteres, 'v' para verlos)"
        lines.append(text)
    return "\n".join(lines)


async def _stdin_reader(prompt: str) -> str:
    return await asyncio.to_thread(input, prompt)


def make_cli_confirmer(read_line=_stdin_reader):
    """Confirmador de terminal. `read_line(prompt)` (async) permite usar otra fuente de teclado,
    por ejemplo la cola compartida del modo voz, donde dos lectores de stdin se pelearian."""

    async def confirmer(req: ActionRequest, explicit: bool) -> bool:
        full = False
        while True:
            print(render(req, explicit, full))
            prompt = (
                "Escribe CONFIRMAR para autorizar, 'v' para ver todo, o Enter para cancelar: "
                if explicit
                else "Autorizar? [s/N, v = ver completo] "
            )
            try:
                answer = (await read_line(prompt)).strip()
            except (EOFError, KeyboardInterrupt):
                return False
            if answer.lower() == "v":
                full = True
                continue
            if explicit:
                return answer == "CONFIRMAR"
            return answer.lower() in YES

    return confirmer


cli_confirmer = make_cli_confirmer()
