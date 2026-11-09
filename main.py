"""Punto de entrada de P.O.E.T.A.

  python main.py                        Carga y valida la configuracion.
  python main.py chat [--continue]      Chat de texto en streaming (--continue retoma la ultima conversacion).
  python main.py voice [--wake|--always]  Conversacion por voz (Enter para hablar; Enter interrumpe).
  python main.py audit [--verify]       Muestra o verifica el registro de auditoria.
  python main.py memory stats|export ARCHIVO|wipe   Administra la memoria.
  python main.py tool NOMBRE ['{"arg": ...}']   Ejecuta una herramienta sin el modelo (diagnostico); 'tool list' las lista.
  python main.py mcp list|approve NOMBRE   Servidores MCP: ver sus herramientas y aprobarlos.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

from poeta.config import PROJECT_ROOT, ConfigError, assistant_name, load_config, system_prompt
from poeta.automation import DesktopConfig, load_routines, register_desktop, register_routines
from poeta.core import AgentCore
from poeta.envfile import load_env_file
from poeta.interface import cli_confirmer, make_cli_confirmer
from poeta.logging_setup import setup_logging
from poeta.memory import KINDS, MemoryStore, MemoryStoreError
from poeta.models import ModelRouter, RouterError
from poeta.providers import ProviderError, ToolCall
from poeta.tools import ToolRegistry, ToolsConfig, register_builtin
from poeta.mcp import McpConfig, McpManager
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
  /run NOMBRE               Ejecuta una rutina de config/automations.yaml
  /tools                    Herramientas disponibles (integradas y MCP)
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
    elif cmd == "/run":
        reg = agent.tools
        if reg is None or reg.get("run_automation") is None:
            print("No hay rutinas disponibles (config/automations.yaml).")
        elif not arg:
            print("Uso: /run NOMBRE")
        else:
            res = await reg.execute(ToolCall(id="cmd-run", name="run_automation", arguments=json.dumps({"name": arg})))
            print(res.text)
    elif cmd == "/tools":
        reg = agent.tools
        if reg is None or not reg.names():
            print("No hay herramientas habilitadas.")
        for t in (reg.tools() if reg else []):
            print(f"  {t.name:<28} {t.source:<14} {t.description[:70]}")
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


def announce_permission_warnings(permissions: PermissionManager, audit: AuditLog | None) -> None:
    """Los ajustes mas permisivos que lo recomendado se permiten, pero nunca en silencio."""
    if not permissions.warnings:
        return
    for w in permissions.warnings:
        print(f"Aviso de permisos: {w}", file=sys.stderr)
    logging.getLogger("poeta").warning("permisos relajados", extra={"avisos": len(permissions.warnings)})
    if audit is not None:
        try:
            audit.record("permissions_relaxed", warnings=permissions.warnings)
        except Exception:  # noqa: BLE001
            pass


def build_agent(cfg: dict, provider: str | None, model: str | None, confirmer=None):
    """Crea auditoria, permisos, router, memoria y agente. Devuelve (router, agent, memory) o None."""
    try:
        audit = AuditLog.from_config(cfg, PROJECT_ROOT)
    except AuditError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return None
    permissions = PermissionManager(cfg, confirmer=confirmer or cli_confirmer, audit=audit, root=PROJECT_ROOT)
    announce_permission_warnings(permissions, audit)
    router = ModelRouter(cfg, permissions=permissions)
    try:
        router.select(provider, model)
    except RouterError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return None
    memory = open_memory(cfg, audit)
    agent_cfg = cfg["settings"].get("agent", {})
    tools_cfg = ToolsConfig.from_config(cfg)
    registry = None
    mcp = None
    if tools_cfg.enabled:
        registry = ToolRegistry(permissions, tools_cfg.max_output_chars, tools_cfg.timeout_s)
        register_builtin(registry, tools_cfg, PROJECT_ROOT)
        desk = DesktopConfig.from_config(cfg)
        if desk.enabled:
            _, skipped = register_desktop(registry, desk)
            for name, why in skipped.items():
                logging.getLogger("poeta").info("herramienta de escritorio omitida: %s (%s)", name, why)
            _, skipped_r = register_routines(registry, load_routines(cfg))
            for name, why in skipped_r.items():
                print(f"Aviso: la rutina '{name}' no se cargo: {why}", file=sys.stderr)
        for kind, level in permissions.missing_actions(
                t.action for t in registry.tools() if isinstance(t.action, str)).items():
            print(f"Aviso de permisos: falta la accion '{kind}' en permissions.yaml (se denegara). "
                  f"Agrega en 'actions:' la linea  {kind}: {level}", file=sys.stderr)
        mcp = McpManager(McpConfig.from_config(cfg), registry, PROJECT_ROOT, audit)
    agent = AgentCore(
        router,
        system_prompt=system_prompt(cfg),
        max_history=int(agent_cfg.get("max_history_messages", 40)),
        memory=memory,
        memory_cfg=cfg["memory"],
        tools=registry,
        max_tool_rounds=tools_cfg.max_tool_rounds,
    )
    agent.mcp = mcp  # se inicia/detiene con start_mcp/stop_mcp
    return router, agent, memory


async def start_mcp(agent: AgentCore) -> None:
    """Conecta los servidores MCP autorizados y aprobados; informa de los omitidos."""
    mcp = getattr(agent, "mcp", None)
    if mcp is None:
        return
    for st in await mcp.start():
        if st.state == "conectado":
            print(f"MCP {st.name}: {len(st.tools)} herramientas")
        elif st.state in {"omitido", "error"}:
            print(f"MCP {st.name}: {st.state}. {st.detail}")


async def stop_mcp(agent: AgentCore) -> None:
    mcp = getattr(agent, "mcp", None)
    if mcp is not None:
        await mcp.stop()


def show_tool_event(ev, out=print) -> None:
    if ev.kind == "tool_start" and ev.tool_calls:
        out(f"\n[herramienta: {', '.join(t.name for t in ev.tool_calls)}]")
    elif ev.kind == "tool_result" and ev.result is not None:
        r = ev.result
        out(f"[{r.name}: {'denegada' if r.denied else ('ok' if r.ok else 'error')}]")


async def chat(cfg: dict, provider: str | None, model: str | None, resume_last: bool = False) -> int:
    built = build_agent(cfg, provider, model)
    if built is None:
        return 1
    router, agent, memory = built

    st = router.status()
    print(f"{assistant_name(cfg)} listo. Proveedor: {st['provider']}  Modelo: {st['model']}")
    if resume_last and memory is not None:
        last = memory.last_conversation_id()
        if last is None:
            print("No hay conversaciones guardadas; empiezo una nueva.")
        else:
            print(f"Retomada la conversacion #{last} ({agent.resume(last)} mensajes cargados).")
    await start_mcp(agent)
    print("Escribe /help para ver los comandos.\n")

    try:
        return await _chat_loop(cfg, router, agent)
    finally:
        await stop_mcp(agent)


async def _chat_loop(cfg: dict, router: ModelRouter, agent: AgentCore) -> int:
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
                else:
                    show_tool_event(ev)
            print("\n")
        except (RouterError, ProviderError) as exc:
            print(f"\nError: {exc}\n")
        if agent.memory_errors and not warned:
            warned = True
            print("Aviso: la memoria fallo y esta conversacion puede no haberse guardado (ver logs/poeta.log).\n")


async def voice(cfg: dict, provider: str | None, model: str | None, resume_last: bool, always: bool,
                wake: bool = False) -> int:
    import time

    from poeta.voice.audio import AudioConfig, AudioError, AudioManager
    from poeta.voice.loop import LineQueue, VoiceLoop
    from poeta.voice.stt import SttConfig, SttError, Transcriber
    from poeta.voice.tts import SpeechPipeline, TtsConfig, TtsError, make_engine
    from poeta.voice.wake import TranscriptWake, WakeConfig, make_chime

    keys = LineQueue()  # unico lector de teclado: tambien responde las confirmaciones de permisos
    built = build_agent(cfg, provider, model, confirmer=make_cli_confirmer(keys.ask))
    if built is None:
        return 1
    router, agent, memory = built
    audio = None
    pipe = None
    try:
        t0 = time.perf_counter()
        stt = Transcriber(SttConfig.from_config(cfg))
        engine = make_engine(TtsConfig.from_config(cfg))
        print(f"Cargando voz (STT {stt.config.model}, TTS {engine.__class__.__name__})...", flush=True)
        await asyncio.to_thread(stt.load)
        await asyncio.to_thread(stt.warmup)
        await asyncio.to_thread(engine.synthesize, "Listo.")
        detector = None
        wcfg = None
        if wake:
            wcfg = WakeConfig.from_config(cfg)
            detector = TranscriptWake(wcfg, Transcriber(SttConfig(
                model=wcfg.stt_model, device=wcfg.stt_device, compute_type=wcfg.stt_compute_type,
                language=stt.config.language, download_root=stt.config.download_root)))
            print(f"Cargando detector de '{wcfg.phrase}' ({wcfg.stt_model}, {wcfg.stt_device})...", flush=True)
            await asyncio.to_thread(detector.load)
        audio = AudioManager(AudioConfig.from_config(cfg))
        audio.start()
        print(f"Voz lista en {time.perf_counter() - t0:.1f} s.")
    except (SttError, TtsError, AudioError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        if audio is not None:
            audio.stop()
        return 1
    pipe = SpeechPipeline(engine, audio, on_error=lambda e: print(f"\n(error de voz: {e})"))

    st = router.status()
    print(f"{assistant_name(cfg)} listo por voz. Proveedor: {st['provider']}  Modelo: {st['model']}")
    if resume_last and memory is not None:
        last = memory.last_conversation_id()
        if last is not None:
            print(f"Retomada la conversacion #{last} ({agent.resume(last)} mensajes cargados).")

    await start_mcp(agent)
    keys.start_stdin_reader(sys.stdin)

    async def on_command(line: str) -> bool:
        return await run_command(line, router, agent)

    loop = VoiceLoop(audio=audio, stt=stt, pipeline=pipe, agent=agent, keys=keys,
                     mode="wake" if wake else ("always" if always else "ptt"),
                     wake=detector, followup_s=wcfg.followup_s if wcfg else 8.0,
                     chime=(make_chime(), 24000) if wcfg and wcfg.chime else None, on_command=on_command)
    try:
        return await loop.run()
    finally:
        pipe.close()
        audio.stop()
        await stop_mcp(agent)


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


def mcp_cmd(cfg: dict, action: str, name: str | None) -> int:
    try:
        audit = AuditLog.from_config(cfg, PROJECT_ROOT)
    except AuditError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    mcp_cfg = McpConfig.from_config(cfg)
    mgr = McpManager(mcp_cfg, ToolRegistry(None, 1, 1), PROJECT_ROOT, audit)
    if action == "list":
        pins = mgr._pins()
        if not mcp_cfg.servers:
            print("No hay servidores definidos en config/mcp.yaml.")
        for n, sc in mcp_cfg.servers.items():
            estado = ("habilitado" if sc.enabled else "deshabilitado") + (", autorizado" if n in mcp_cfg.authorized else ", NO autorizado")
            print(f"{n:<14} {estado}; riesgo por defecto: {sc.risk}; aprobado: {'si' if n in pins else 'no'}")
        return 0
    if not name or name not in mcp_cfg.servers:
        print(f"Uso: python main.py mcp approve NOMBRE (servidores: {', '.join(mcp_cfg.servers) or 'ninguno'})", file=sys.stderr)
        return 2
    if not mcp_cfg.eligible(name):
        print(f"'{name}' debe tener enabled: true y estar en authorized_servers de config/mcp.yaml.", file=sys.stderr)
        return 1

    async def inspect():
        return await mgr.inspect(name)

    try:
        tools, fp = asyncio.run(inspect())
    except Exception as exc:  # noqa: BLE001
        print(f"No pude conectar con '{name}': {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    sc = mcp_cfg.servers[name]
    print(f"Servidor '{name}' ({sc.command or sc.url}) expone {len(tools)} herramientas:")
    for t in tools:
        print(f"  - {t.name} [riesgo {sc.tool_risk.get(t.name, sc.risk)}]: {' '.join((t.description or '').split())[:100]}")
    print(f"Huella: {fp[:16]}...\nEstas descripciones las escribio el servidor (codigo de terceros): no son instrucciones de confianza.")
    if action == "approve":
        if input("Escribe APROBAR para fijar esta huella: ").strip() != "APROBAR":
            print("Cancelado.")
            return 1
        mgr.approve(name, fp, [t.name for t in tools])
        print("Aprobado. Si el servidor cambia sus herramientas, se desactivara hasta aprobar de nuevo.")
    return 0


async def tool_cmd(cfg: dict, name: str | None, raw: str | None) -> int:
    """Ejecuta una herramienta directamente (sin modelo): separa 'fallo la herramienta' de 'el modelo no la uso'."""
    built = build_agent(cfg, None, None)
    if built is None:
        return 1
    _, agent, _ = built
    reg = agent.tools
    if reg is None:
        print("Las herramientas estan deshabilitadas (config/tools.yaml).")
        return 1
    await start_mcp(agent)
    try:
        if not name or name == "list":
            for t in reg.tools():
                print(f"{t.name:<28} {json.dumps(t.parameters.get('properties', {}), ensure_ascii=False)[:90]}")
            return 0
        res = await reg.execute(ToolCall(id="cli", name=name, arguments=raw or "{}"))
        print(f"[{'denegada' if res.denied else ('ok' if res.ok else 'ERROR')}] {res.text}")
        return 0 if res.ok else 1
    finally:
        await stop_mcp(agent)


def main() -> int:
    parser = argparse.ArgumentParser(prog="poeta")
    sub = parser.add_subparsers(dest="command")
    chat_p = sub.add_parser("chat", help="Chat de texto en streaming")
    chat_p.add_argument("--provider", help="Proveedor (por defecto, el de providers.yaml)")
    chat_p.add_argument("--model", help="Modelo")
    chat_p.add_argument("--continue", dest="resume_last", action="store_true", help="Retoma la ultima conversacion guardada")
    voice_p = sub.add_parser("voice", help="Conversacion por voz (Enter para hablar)")
    voice_p.add_argument("--provider", help="Proveedor (por defecto, el de providers.yaml)")
    voice_p.add_argument("--model", help="Modelo")
    voice_p.add_argument("--continue", dest="resume_last", action="store_true", help="Retoma la ultima conversacion")
    mode_g = voice_p.add_mutually_exclusive_group()
    mode_g.add_argument("--wake", action="store_true",
                        help="Manos libres: solo responde cuando empiezas con 'poeta' (lo demas se descarta)")
    mode_g.add_argument("--always", action="store_true",
                        help="Escucha continuamente, sin palabra de activacion (todo se transcribe localmente)")
    audit_p = sub.add_parser("audit", help="Registro de auditoria")
    audit_p.add_argument("--verify", action="store_true", help="Verificar la integridad de la cadena de hashes")
    audit_p.add_argument("--tail", type=int, default=20, help="Cuantos eventos mostrar (por defecto 20)")
    mem_p = sub.add_parser("memory", help="Administrar la memoria")
    mem_p.add_argument("action", choices=["stats", "export", "wipe"])
    mem_p.add_argument("target", nargs="?", help="Archivo de destino (solo export)")
    tool_p = sub.add_parser("tool", help="Ejecuta una herramienta sin el modelo (diagnostico)")
    tool_p.add_argument("name", nargs="?", help="Nombre de la herramienta, o 'list'")
    tool_p.add_argument("args", nargs="?", help="Argumentos en JSON, ej: '{\"name\": \"Firefox\"}'")
    mcp_p = sub.add_parser("mcp", help="Servidores MCP")
    mcp_p.add_argument("action", choices=["list", "approve", "inspect"])
    mcp_p.add_argument("name", nargs="?", help="Servidor (approve/inspect)")
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
    if args.command == "voice":
        return asyncio.run(voice(cfg, args.provider, args.model, args.resume_last, args.always, args.wake))
    if args.command == "audit":
        return audit_cmd(cfg, args.verify, args.tail)
    if args.command == "tool":
        return asyncio.run(tool_cmd(cfg, args.name, args.args))
    if args.command == "mcp":
        return mcp_cmd(cfg, args.action, args.name)
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
