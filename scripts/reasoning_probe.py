#!/usr/bin/env python
"""Verifica que tu llama-server acepta el control de razonamiento POR PEDIDO (chat_template_kwargs.enable_thinking).
Envia la misma pregunta con razonamiento activado y desactivado y compara. Necesita llama-server corriendo.
   python scripts/reasoning_probe.py"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import httpx  # noqa: E402

from poeta.config import load_config  # noqa: E402

PROMPT = "Tengo 3 cajas con 4 manzanas cada una y regalo 5. ¿Cuántas me quedan? Responde en una frase."


async def ask(base: str, model: str, think: bool) -> tuple[float | None, int, str]:
    body = {"model": model, "stream": True, "messages": [{"role": "user", "content": PROMPT}],
            "chat_template_kwargs": {"enable_thinking": think}, "max_tokens": 600}
    t0, first, reasoning, text = time.perf_counter(), None, 0, ""
    async with httpx.AsyncClient(timeout=120) as c:
        async with c.stream("POST", f"{base}/chat/completions", json=body) as r:
            if r.status_code >= 400:
                raise SystemExit(f"HTTP {r.status_code}: {(await r.aread()).decode()[:300]}")
            async for line in r.aiter_lines():
                if not line.startswith("data:") or line.endswith("[DONE]"):
                    continue
                import json
                for ch in json.loads(line[5:]).get("choices") or []:
                    d = ch.get("delta") or {}
                    reasoning += len(d.get("reasoning_content") or d.get("reasoning") or "")
                    if d.get("content"):
                        if first is None:
                            first = time.perf_counter() - t0
                        text += d["content"]
    return first, reasoning, text.strip()


async def main() -> int:
    cfg = load_config()
    base = cfg["providers"]["providers"]["local"]["base_url"]
    model = next(iter(cfg["models"]["llm"]["local"].values()))["alias"]
    print(f"Servidor {base}, modelo {model}\n")
    res = {}
    try:
        for think in (True, False):
            res[think] = await ask(base, model, think)
            first, rchars, text = res[think]
            print(f"enable_thinking={think!s:<5}  primer texto: {first and round(first, 2)} s   "
                  f"caracteres de razonamiento: {rchars}\n   respuesta: {text[:100]}")
    except httpx.HTTPError as exc:
        print(f"No pude conectar con llama-server: {exc}")
        return 1
    on, off = res[True], res[False]
    if on[1] > 0 and off[1] == 0:
        print("\nOK: el servidor respeta el control por pedido. Con agent.reasoning.mode: auto POETA razona solo cuando hace falta.")
        return 0
    if on[1] == 0 and off[1] == 0:
        print("\nEl modelo no razono en ninguno de los dos casos: revisa que llama-server tenga '--reasoning on'.")
        return 1
    print("\nEl servidor IGNORO enable_thinking (razona igual): este modelo/plantilla no lo soporta. "
          "Usa agent.reasoning.mode: server y elige 'on' u 'off' en models.yaml.")
    return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
