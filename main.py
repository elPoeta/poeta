"""Punto de entrada de P.O.E.T.A.

  python main.py          Carga y valida la configuracion.
  python main.py chat     Chat de texto en streaming (Fase 1).
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from poeta.config import PROJECT_ROOT, ConfigError, load_config
from poeta.core import AgentCore
from poeta.logging_setup import setup_logging
from poeta.models import ModelRouter, RouterError, TransferRequest
from poeta.providers import ProviderError

HELP = """Comandos:
  /status                  Proveedor y modelo activos
  /provider NOMBRE [MODELO] Cambia de proveedor (manual; nunca ocurre solo)
  /model NOMBRE            Cambia de modelo
  /reset                   Borra el historial
  /quit                    Salir"""


async def cli_authorizer(req: TransferRequest) -> bool:
    """Autorizacion minima de la Fase 1: muestra exactamente lo que se enviaria."""
    print("\n--- TRANSFERENCIA A PROVEEDOR EXTERNO ---")
    print(f"Proveedor: {req.provider}   Modelo: {req.model}   Operacion: {req.operation}")
    print("Se enviara:")
    for m in req.messages:
        print(f"  [{m['role']}] {m.get('content', '')}")
    if req.tools:
        print(f"  (+ {len(req.tools)} herramientas)")
    answer = await asyncio.to_thread(input, "Enviar esto al proveedor externo? [s/N] ")
    return answer.strip().lower() in {"s", "si", "sí", "y", "yes"}


async def chat(cfg: dict, provider: str | None, model: str | None) -> int:
    router = ModelRouter(cfg, authorizer=cli_authorizer)
    try:
        router.select(provider, model)
    except RouterError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    agent_cfg = cfg["settings"].get("agent", {})
    agent = AgentCore(
        router,
        system_prompt=agent_cfg.get("system_prompt", "Eres POETA, un asistente personal."),
        max_history=int(agent_cfg.get("max_history_messages", 40)),
    )

    st = router.status()
    print(f"POETA listo. Proveedor: {st['provider']}  Modelo: {st['model']}")
    print("Escribe /help para ver los comandos.\n")

    while True:
        try:
            line = (await asyncio.to_thread(input, "> ")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line:
            continue

        if line.startswith("/"):
            cmd, *rest = line.split()
            if cmd in {"/quit", "/exit"}:
                return 0
            if cmd == "/help":
                print(HELP)
            elif cmd == "/status":
                print(router.status())
            elif cmd == "/reset":
                agent.reset()
                print("Historial borrado.")
            elif cmd == "/provider" and rest:
                try:
                    router.select(rest[0], rest[1] if len(rest) > 1 else None)
                    print(router.status())
                except RouterError as exc:
                    print(f"Error: {exc}")
            elif cmd == "/model" and rest:
                router.select(model=rest[0])
                print(router.status())
            else:
                print("Comando no reconocido. /help")
            continue

        try:
            async for ev in agent.respond(line):
                if ev.kind == "text":
                    print(ev.text, end="", flush=True)
                elif ev.kind == "tool_calls":
                    print(f"\n[el modelo pidio herramientas: {[t.name for t in ev.tool_calls]} (aun no se ejecutan)]")
            print("\n")
        except (RouterError, ProviderError) as exc:
            print(f"\nError: {exc}\n")


def main() -> int:
    parser = argparse.ArgumentParser(prog="poeta")
    sub = parser.add_subparsers(dest="command")
    chat_p = sub.add_parser("chat", help="Chat de texto en streaming")
    chat_p.add_argument("--provider", help="Proveedor (por defecto, el de providers.yaml)")
    chat_p.add_argument("--model", help="Modelo")
    args = parser.parse_args()

    try:
        cfg = load_config()
    except ConfigError as exc:
        print(f"Error de configuracion: {exc}", file=sys.stderr)
        return 1

    log = setup_logging(cfg, PROJECT_ROOT)

    if args.command == "chat":
        return asyncio.run(chat(cfg, args.provider, args.model))

    defaults = cfg["providers"]["defaults"]
    log.info(
        "configuracion validada",
        extra={"provider": defaults["provider"], "cloud_fallback": defaults["cloud_fallback"]},
    )
    print("Configuracion valida. Usa 'python main.py chat' o 'python scripts/doctor.py'.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
