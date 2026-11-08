"""Punto de entrada de P.O.E.T.A.

  python main.py                        Carga y valida la configuracion.
  python main.py chat [--continue]      Chat de texto en streaming (--continue retoma la ultima conversacion).
  python main.py audit [--verify]       Muestra o verifica el registro de auditoria.
  python main.py memory stats|export ARCHIVO|wipe   Administra la memoria.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from poeta.config import PROJECT_ROOT, ConfigError, assistant_name, load_config, system_prompt
from poeta.core import AgentCore
from poeta.envfile import load_env_file
from poeta.interface import cli_confirmer
from poeta.logging_setup import setup_logging
from poeta.memory import KINDS, MemoryStore, MemoryStoreError
from poeta.models import ModelRouter, RouterError
from poeta.providers import ProviderError
from poeta.security import AuditError, AuditLog, PermissionManager, audit_path, read_tail, verify_chain

HELP = """Comandos:
  /status                   Proveedor, modelo y estado de la memoria
  /provider NOMBRE [MODELO] Cambia de proveedor (manual; nunca ocurre solo)
  /model NOMBRE             Cambia de modelo
  /reset                    Borra el historial y empieza una conversacion nueva
  /remember TEXTO           Guarda una nota
  /memories [CONSULTA]      Lista o busca notas y preferencias
  /forget ID                Olvida una nota
  /prefer CLAVE=VALOR       Guarda una preferencia (sin argumentos: las lista)
  /unprefer CLAVE           Borra una preferencia
  /conversations            Lista las conversaciones guardadas
  /resume ID                Retoma una conversacion
  /incognito                Activa o desactiva el guardado de esta sesion
  /quit                     Salir"""


def _short(text: str, n: int = 100) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 3] + "..."


def open_memory(cfg: dict[str, Any], audit: AuditLog | None, log: Any = None) -> MemoryStore | None:
    """Abre la memoria si esta habilitada. Si falla, el asistente sigue sin ella y se avisa."""
    if not cfg["memory"].get("enabled", True):
        return None
    try:
        store = MemoryStore.from_config(cfg, PROJECT_ROOT, audit)
        removed = store.purge_expired()
        if log is not None and any(removed.values()):
            log.info("memoria: filas expiradas eliminadas", extra=removed)
        return store
    except MemoryStoreError as exc:
        print(f"Aviso: {exc}\nSigo SIN memoria en esta sesion.", file=sys.stderr)
        return None


async def run_command(line: str, router: ModelRouter, agent: AgentCore) -> bool:
    """Ejecuta un comando '/...'. Devuelve True si hay que salir."""
    cmd, _, arg = line.partition(" ")
    arg = arg.strip()
    memory = agent.memory

    if cmd in {"/quit", "/exit"}:
        return True
    if cmd == "/help":
        print(HELP)
    elif cmd == "/status":
        status = dict(router.status())
        status["memoria"] = (
            "deshabilitada" if memory is None else ("incognito (no se guarda)" if agent.incognito else "activa")
        )
        status["conversacion"] = agent.conversation_id
        print(status)
    elif cmd == "/reset":
        agent.reset()
        print("Historial borrado. La proxima conversacion sera nueva.")
    elif cmd == "/provider" and arg:
        parts = arg.split()
        try:
            router.select(parts[0], parts[1] if len(parts) > 1 else None)
            print(router.status())
        except RouterError as exc:
            print(f"Error: {exc}")
    elif cmd == "/model" and arg:
        router.select(model=arg.split()[0])
        print(router.status())
    elif cmd == "/incognito":
        agent.incognito = not agent.incognito
        print("Incognito ACTIVADO: esta sesion no se guarda." if agent.incognito else "Incognito desactivado: se vuelve a guardar.")
    elif cmd in {"/remember", "/memories", "/forget", "/prefer", "/unprefer", "/conversations", "/resume"}:
        if memory is None:
            print("La memoria esta deshabilitada o no se pudo abrir.")
            return False
        try:
            _memory_command(cmd, arg, agent, memory)
        except (ValueError, LookupError) as exc:
            print(f"Error: {exc}")
    else:
        print("Comando no reconocido. /help")
    return False


def _memory_command(cmd: str, arg: str, agent: AgentCore, memory: MemoryStore) -> None:
    if cmd == "/remember":
        if not arg:
            raise ValueError("Uso: /remember TEXTO")
        print(f"Guardado (#{memory.add_memory('semantic', arg, conversation_id=agent.conversation_id)}).")
    elif cmd == "/memories":
        if arg:
            rows = memory.search_memories(arg, limit=10)
        else:
            rows = memory.list_memories(limit=20)
            for key, value in memory.get_preferences().items():
                print(f"  preferencia {key}: {_short(value)}")
        for m in rows:
            print(f"  #{m['id']} [{m['kind']}] {_short(m['content'])}")
        if not rows and not (not arg and memory.get_preferences()):
            print("Sin resultados.")
    elif cmd == "/forget":
        if not arg.isdigit():
            raise ValueError("Uso: /forget ID")
        print("Olvidado." if memory.forget(int(arg)) else f"No existe la nota #{arg}.")
    elif cmd == "/prefer":
        if not arg:
            prefs = memory.get_preferences()
            for key, value in prefs.items():
                print(f"  {key} = {_short(value)}")
            if not prefs:
                print("Sin preferencias.")
            return
        key, sep, value = arg.partition("=")
        if not sep:
            raise ValueError("Uso: /prefer CLAVE=VALOR")
        memory.set_preference(key, value)
        print("Preferencia guardada.")
    elif cmd == "/unprefer":
        print("Preferencia borrada." if memory.delete_preference(arg) else "No existe esa preferencia.")
    elif cmd == "/conversations":
        rows = memory.list_conversations(10)
        for c in rows:
            print(f"  #{c['id']} {c['started_at'][:16].replace('T', ' ')}  {c['messages']:>3} msgs  {_short(c['title'] or '(sin titulo)', 60)}")
        if not rows:
            print("Sin conversaciones guardadas.")
    elif cmd == "/resume":
        if not arg.isdigit():
            raise ValueError("Uso: /resume ID")
        n = agent.resume(int(arg))
        print(f"Retomada la conversacion #{arg} ({n} mensajes cargados).")


async def chat(cfg: dict, provider: str | None, model: str | None, resume_last: bool = False) -> int:
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

    memory = open_memory(cfg, audit)
    agent_cfg = cfg["settings"].get("agent", {})
    agent = AgentCore(
        router,
        system_prompt=system_prompt(cfg),
        max_history=int(agent_cfg.get("max_history_messages", 40)),
        memory=memory,
        memory_cfg=cfg["memory"],
    )

    st = router.status()
    print(f"{assistant_name(cfg)} listo. Proveedor: {st['provider']}  Modelo: {st['model']}")
    if resume_last and memory is not None:
        last = memory.last_conversation_id()
        if last is None:
            print("No hay conversaciones guardadas; empiezo una nueva.")
        else:
            print(f"Retomada la conversacion #{last} ({agent.resume(last)} mensajes cargados).")
    print("Escribe /help para ver los comandos.\n")

    warned = False
    while True:
        try:
            line = (await asyncio.to_thread(input, "> ")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line:
            continue

        if line.startswith("/"):
            if await run_command(line, router, agent):
                return 0
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
        if agent.memory_errors and not warned:
            warned = True
            print("Aviso: la memoria fallo y esta conversacion puede no haberse guardado (ver logs/poeta.log).\n")


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


def memory_cmd(cfg: dict, action: str, target: str | None) -> int:
    try:
        audit = AuditLog.from_config(cfg, PROJECT_ROOT)
    except AuditError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    try:
        store = MemoryStore.from_config(cfg, PROJECT_ROOT, audit)
    except MemoryStoreError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if action == "stats":
        s = store.stats()
        print(f"Base: {s['path']}  ({s['bytes'] / 1024:.0f} KiB)")
        for table, n in s["counts"].items():
            print(f"  {table:<14} {n}")
        return 0
    if action == "export":
        if not target:
            print("Uso: python main.py memory export ARCHIVO.json", file=sys.stderr)
            return 2
        try:
            counts = store.export(Path(target).expanduser())
        except FileExistsError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        print(f"Exportado a {target} (archivo privado, 0600): {counts}\nEs texto plano: guardalo con el mismo cuidado que la base.")
        return 0
    if action == "wipe":
        s = store.stats()["counts"]
        print(f"Esto BORRA toda la memoria: {s}. No se puede deshacer.")
        try:
            answer = input("Escribe BORRAR TODO para continuar: ").strip()
        except (EOFError, KeyboardInterrupt):
            answer = ""
        if answer != "BORRAR TODO":
            print("Cancelado.")
            return 1
        store.wipe()
        print("Memoria borrada.")
        return 0
    return 2


def main() -> int:
    parser = argparse.ArgumentParser(prog="poeta")
    sub = parser.add_subparsers(dest="command")
    chat_p = sub.add_parser("chat", help="Chat de texto en streaming")
    chat_p.add_argument("--provider", help="Proveedor (por defecto, el de providers.yaml)")
    chat_p.add_argument("--model", help="Modelo")
    chat_p.add_argument("--continue", dest="resume_last", action="store_true", help="Retoma la ultima conversacion guardada")
    audit_p = sub.add_parser("audit", help="Registro de auditoria")
    audit_p.add_argument("--verify", action="store_true", help="Verificar la integridad de la cadena de hashes")
    audit_p.add_argument("--tail", type=int, default=20, help="Cuantos eventos mostrar (por defecto 20)")
    mem_p = sub.add_parser("memory", help="Administrar la memoria")
    mem_p.add_argument("action", choices=["stats", "export", "wipe"])
    mem_p.add_argument("target", nargs="?", help="Archivo de destino (solo export)")
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
        return asyncio.run(chat(cfg, args.provider, args.model, args.resume_last))
    if args.command == "audit":
        return audit_cmd(cfg, args.verify, args.tail)
    if args.command == "memory":
        return memory_cmd(cfg, args.action, args.target)

    defaults = cfg["providers"]["defaults"]
    log.info(
        "configuracion validada",
        extra={"provider": defaults["provider"], "cloud_fallback": defaults["cloud_fallback"]},
    )
    print("Configuracion valida. Usa 'python main.py chat' o 'python scripts/doctor.py'.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
