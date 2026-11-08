"""Punto de entrada de P.O.E.T.A.

  python main.py                      Carga y valida la configuracion.
  python main.py chat                 Chat de texto en streaming.
  python main.py audit [--verify]     Muestra o verifica el registro de auditoria.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from poeta.config import PROJECT_ROOT, ConfigError, assistant_name, load_config, system_prompt
from poeta.core import AgentCore
from poeta.envfile import load_env_file
from poeta.interface import cli_confirmer
from poeta.logging_setup import setup_logging
from poeta.models import ModelRouter, RouterError
from poeta.providers import ProviderError
from poeta.security import AuditError, AuditLog, PermissionManager, audit_path, read_tail, verify_chain

HELP = """Comandos:
  /status                  Proveedor y modelo activos
  /provider NOMBRE [MODELO] Cambia de proveedor (manual; nunca ocurre solo)
  /model NOMBRE            Cambia de modelo
  /reset                   Borra el historial
  /quit                    Salir"""


async def chat(cfg: dict, provider: str | None, model: str | None) -> int:
    try:
        audit = AuditLog.from_config(cfg, PROJECT_ROOT)
    except AuditError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    permissions = PermissionManager(cfg, confirmer=cli_confirmer, audit=audit, root=PROJECT_ROOT)
    router = ModelRouter(cfg, permissions=permissions)
    try:
        router.select(provider, model)
    except RouterError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    agent_cfg = cfg["settings"].get("agent", {})
    agent = AgentCore(
        router,
        system_prompt=system_prompt(cfg),
        max_history=int(agent_cfg.get("max_history_messages", 40)),
    )

    st = router.status()
    print(f"{assistant_name(cfg)} listo. Proveedor: {st['provider']}  Modelo: {st['model']}")
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


def audit_cmd(cfg: dict, verify: bool, tail: int) -> int:
    path = audit_path(cfg, PROJECT_ROOT)
    if verify:
        ok, n, err = verify_chain(path)
        if ok:
            print(f"Cadena integra: {n} eventos verificados ({path}).")
            return 0
        print(f"CADENA ROTA tras {n} eventos verificados: {err}", file=sys.stderr)
        return 1
    entries = read_tail(path, tail)
    if not entries:
        print("Sin eventos registrados.")
        return 0
    for e in entries:
        rest = {k: v for k, v in e.items() if k not in {"seq", "ts", "event", "prev", "hash"}}
        summary = json.dumps(rest, ensure_ascii=False)
        print(f"{e.get('seq', '?'):>5} {str(e.get('ts', ''))[:19]} {e.get('event', ''):<22} {summary[:160]}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="poeta")
    sub = parser.add_subparsers(dest="command")
    chat_p = sub.add_parser("chat", help="Chat de texto en streaming")
    chat_p.add_argument("--provider", help="Proveedor (por defecto, el de providers.yaml)")
    chat_p.add_argument("--model", help="Modelo")
    audit_p = sub.add_parser("audit", help="Registro de auditoria")
    audit_p.add_argument("--verify", action="store_true", help="Verificar la integridad de la cadena de hashes")
    audit_p.add_argument("--tail", type=int, default=20, help="Cuantos eventos mostrar (por defecto 20)")
    args = parser.parse_args()

    env = load_env_file(PROJECT_ROOT / ".env")  # antes de load_config: las rutas pueden usar variables del .env
    try:
        cfg = load_config()
    except ConfigError as exc:
        print(f"Error de configuracion: {exc}", file=sys.stderr)
        return 1

    log = setup_logging(cfg, PROJECT_ROOT)
    if env.found:
        log.info("archivo .env cargado", extra={"variables": env.loaded, "ya_definidas_en_el_shell": env.kept_existing})
        if env.error:
            log.warning("no se pudo leer .env", extra={"error": env.error})
        if env.malformed_lines:
            log.warning(".env: lineas mal formadas omitidas", extra={"lineas": env.malformed_lines})
        if env.insecure_permissions:
            log.warning(".env es legible por otros usuarios; protegelo con: chmod 600 .env")

    if args.command == "chat":
        return asyncio.run(chat(cfg, args.provider, args.model))
    if args.command == "audit":
        return audit_cmd(cfg, args.verify, args.tail)

    defaults = cfg["providers"]["defaults"]
    log.info(
        "configuracion validada",
        extra={"provider": defaults["provider"], "cloud_fallback": defaults["cloud_fallback"]},
    )
    print("Configuracion valida. Usa 'python main.py chat' o 'python scripts/doctor.py'.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
