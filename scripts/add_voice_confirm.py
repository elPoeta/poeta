#!/usr/bin/env python
"""Agrega a config/voice.yaml la seccion `confirm:` (confirmar permisos por voz) sin tocar el resto de tu configuracion.

  python scripts/add_voice_confirm.py            # agrega la seccion (apagada) si falta
  python scripts/add_voice_confirm.py --enable   # ademas la deja en enabled: true
"""

import argparse
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
BLOCK = """
# Confirmar permisos hablando (ademas del teclado y los gestos). El microfono no distingue quien habla: cualquier voz o
# audio cercano podria responder "si". Mitigaciones: solo vale una palabra suelta (si/dale/autorizo o no/cancelar),
# nunca envios a la nube, y lo explicito o posterior a datos no confiables queda fuera salvo que lo habilites.
confirm:
  enabled: false
  allow_explicit: false        # nivel 4: exige decir la frase exacta (explicit_phrase) en vez de teclear CONFIRMAR
  explicit_phrase: confirmo la accion
  allow_tainted: false         # acciones pedidas tras leer web/herramientas/MCP (riesgo de inyeccion)
  timeout_s: 20                # cuanto escucha antes de rendirse (quedan el teclado y los gestos)
  retries: 2                   # respuestas no validas tolerables antes de rendirse
  service_timeout_s: 60        # modo servicio (--headless): cuanto esperar antes de denegar
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--enable", action="store_true")
    a = ap.parse_args()
    path = ROOT / "config" / "voice.yaml"
    text = path.read_text(encoding="utf-8")
    if "confirm" not in (yaml.safe_load(text) or {}):
        text = text.rstrip("\n") + "\n" + BLOCK
        print("Seccion `confirm:` agregada.")
    if a.enable:
        text, n = re.subn(r"(?m)^(confirm:\n(?:[ \t]+.*\n|\n|#.*\n)*?  enabled:) false", r"\1 true", text, count=1)
        print("Confirmacion por voz ACTIVADA." if n else "Ya estaba activada.")
    path.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
