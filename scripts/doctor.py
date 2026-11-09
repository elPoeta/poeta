#!/usr/bin/env python3
"""Diagnostico del entorno de P.O.E.T.A. (Fase 0).

Comprueba herramientas, rutas y configuracion. Solo lee: no instala ni modifica nada.
FAIL = bloquea la Fase 1. WARN = necesario mas adelante o no verificable aqui.
Codigo de salida 1 si hay algun FAIL.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import sqlite3
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from poeta.config import ConfigError, load_config  # noqa: E402
from poeta.envfile import load_env_file  # noqa: E402
from poeta.memory import MemoryStore, MemoryStoreError, memory_path  # noqa: E402
from poeta.security import audit_path, verify_chain  # noqa: E402

results: list[tuple[str, str, str]] = []


def record(level: str, name: str, detail: str = "") -> None:
    results.append((level, name, detail))


def check_tool(name: str, level_if_missing: str, hint: str) -> None:
    path = shutil.which(name)
    if path:
        record("OK", name, path)
    else:
        record(level_if_missing, name, f"no encontrado. {hint}")


def check_server_port(base_url: str) -> None:
    """Indica si el puerto del proveedor local esta libre, ocupado por llama-server u ocupado por otro servicio."""
    parsed = urllib.parse.urlparse(base_url)
    host, port = parsed.hostname or "127.0.0.1", parsed.port or 80
    with socket.socket() as s:
        s.settimeout(1)
        in_use = s.connect_ex((host, port)) == 0
    if not in_use:
        record("OK", f"puerto {port}", "libre (llama-server no esta corriendo)")
        return
    try:
        with urllib.request.urlopen(f"{base_url.rstrip('/')}/models", timeout=3) as resp:
            ids = [m.get("id") for m in json.load(resp).get("data", [])]
        record("OK", f"servidor :{port}", f"responde como API OpenAI; modelos: {ids}")
    except Exception:
        record("WARN", f"puerto {port}", "ocupado por otro servicio (no responde como API OpenAI). Ver: ss -ltnp | grep :" + str(port))


def check_memory(cfg: dict) -> None:
    """Solo lee: no crea la base si no existe."""
    if not cfg["memory"].get("enabled", True):
        record("OK", "memoria", "deshabilitada en memory.yaml")
        return
    try:
        sqlite3.connect(":memory:").execute("CREATE VIRTUAL TABLE t USING fts5(c)")
    except sqlite3.Error:
        record("FAIL", "memoria FTS5", "este SQLite no incluye FTS5")
        return
    path = memory_path(cfg, ROOT)
    if not path.exists():
        record("OK", "memoria", f"FTS5 disponible; la base se crea al primer uso ({path})")
        return
    try:
        store = MemoryStore(path)
        check, stats = store.integrity_check(), store.stats()
        store.close()
    except MemoryStoreError as exc:
        record("FAIL", "memoria", str(exc))
        return
    detail = f"{path}: {stats['counts']['messages']} mensajes, {stats['counts']['memories']} notas, {stats['bytes'] / 1024:.0f} KiB"
    if check != "ok":
        record("FAIL", "memoria integridad", check)
    else:
        record("OK", "memoria", detail)
    if oct(path.stat().st_mode & 0o777) != "0o600":
        record("WARN", "memoria permisos", f"{path} deberia ser 0600: chmod 600 {path}")


def main() -> int:
    # Python
    v = sys.version_info
    if (v.major, v.minor) >= (3, 12):
        record("OK", "python", f"{v.major}.{v.minor}.{v.micro}")
    else:
        record("FAIL", "python", f"{v.major}.{v.minor} (se requiere 3.12 o superior)")

    # .env (se carga antes que la configuracion, igual que en main.py)
    env = load_env_file(ROOT / ".env")
    if not env.found:
        record("WARN", ".env", "no existe; solo hace falta para claves cloud (ver .env.example)")
    elif env.error:
        record("FAIL", ".env", f"no se pudo leer ({env.error})")
    else:
        names = ", ".join(env.loaded) or "ninguna nueva"
        record("OK", ".env", f"variables cargadas: {names}" + (f"; ya definidas en el shell: {', '.join(env.kept_existing)}" if env.kept_existing else ""))
        if env.malformed_lines:
            record("WARN", ".env formato", f"lineas mal formadas (se omiten): {env.malformed_lines}")
        if env.insecure_permissions:
            record("WARN", ".env permisos", "legible por otros usuarios; protegelo con: chmod 600 .env")

    # Configuracion
    cfg = None
    try:
        cfg = load_config()
        record("OK", "configuracion", "YAML cargados e invariantes de privacidad validos")
    except ConfigError as exc:
        record("FAIL", "configuracion", str(exc))

    # Herramientas
    check_tool("uv", "FAIL", "Necesario para el entorno (plan secc. 16).")
    check_tool("nvidia-smi", "WARN", "Sin driver NVIDIA no hay GPU para el LLM.")
    check_tool("ffmpeg", "WARN", "Usado para audio/video.")
    check_tool("espeak-ng", "WARN", "Necesario para Kokoro: sudo dnf install espeak-ng")
    check_tool("pw-cli", "WARN", "PipeWire para el Audio Manager.")
    check_tool("git", "WARN", "")

    # Rutas del entorno de IA existente
    if cfg:
        paths = cfg["settings"]["paths"]
        for key in ("ai_home", "models", "voices", "output", "cache"):
            p = Path(paths[key])
            record("OK" if p.is_dir() else "WARN", f"ruta {key}", str(p) + ("" if p.is_dir() else " (no existe)"))

        llama_dir = Path(paths["llama_cpp"])
        if llama_dir.is_dir():
            found = sorted(llama_dir.rglob("llama-server"))
            found = [f for f in found if f.is_file()]
            record("OK" if found else "WARN", "llama-server",
                   str(found[0]) if found else f"no encontrado bajo {llama_dir}")
        else:
            record("WARN", "llama.cpp", f"{llama_dir} no existe")

        llm = cfg["models"]["llm"]
        key = llm["default"]
        m = llm["local"].get(key) or {}
        if m.get("file"):
            mp = Path(paths["models"]) / (m.get("dir") or "") / m["file"]
            record("OK" if mp.is_file() else "FAIL", f"modelo {key}",
                   f"{mp}" + ("" if mp.is_file() else " (no existe)"))
        else:
            record("WARN", f"modelo {key}", "sin 'file' en models.yaml")

        for pname, p in cfg["providers"]["providers"].items():
            if p.get("cloud") and p.get("enabled") and p.get("api_key_env"):
                present = bool(os.environ.get(p["api_key_env"]))
                record("OK" if present else "WARN", f"clave {p['api_key_env']}",
                       "definida" if present else f"no definida (necesaria para usar '{pname}')")

        check_server_port(cfg["providers"]["providers"]["local"]["base_url"])

        check_memory(cfg)

        ap = audit_path(cfg, ROOT)
        if ap.exists():
            ok, n, err = verify_chain(ap)
            record("OK" if ok else "FAIL", "auditoria", f"{n} eventos, cadena integra" if ok else f"cadena rota tras {n} eventos: {err}")
        else:
            record("OK", "auditoria", "sin eventos todavia")

        free_gb = shutil.disk_usage(ROOT).free / 1e9
        record("OK" if free_gb > 20 else "WARN", "espacio libre", f"{free_gb:.0f} GB")

    # Paquetes opcionales de fases posteriores (informativo)
    for mod, phase in (("faster_whisper", "Fase 4"), ("kokoro_onnx", "Fase 4"),
                       ("openwakeword", "Fase 4"), ("cv2", "Fase 5"),
                       ("mediapipe", "Fase 5"), ("mcp", "Fase 6")):
        try:
            __import__(mod)
            record("OK", f"paquete {mod}", "instalado")
        except ImportError:
            record("WARN", f"paquete {mod}", f"no instalado (se necesita en {phase})")

    # Audio: PortAudio (para sounddevice) y servidor PipeWire
    try:
        import sounddevice as sd
        devs = sd.query_devices()
        n_in = sum(1 for d in devs if d["max_input_channels"] > 0)
        record("OK" if n_in else "WARN", "audio (sounddevice)", f"{n_in} entradas, {len(devs) - n_in} otras")
    except ImportError:
        record("WARN", "audio (sounddevice)", "no instalado (uv pip install -e '.[voice]')")
    except OSError:
        record("FAIL", "audio (PortAudio)", "falta la libreria del sistema: sudo dnf install portaudio")

    width = max(len(name) for _, name, _ in results)
    for level, name, detail in results:
        print(f"[{level:<4}] {name:<{width}}  {detail}")

    fails = sum(1 for lvl, _, _ in results if lvl == "FAIL")
    warns = sum(1 for lvl, _, _ in results if lvl == "WARN")
    print(f"\n{fails} FAIL, {warns} WARN")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
