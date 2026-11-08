#!/usr/bin/env python3
"""Lanza llama-server con un modelo local de config/models.yaml.

  python scripts/run_llama_server.py --dry-run             # imprime el comando del modelo por defecto
  python scripts/run_llama_server.py --model CLAVE         # otro modelo del catalogo
  python scripts/run_llama_server.py                       # lo ejecuta (reemplaza este proceso)

Los flags de cada modelo viven en su bloque `server` de models.yaml y son independientes entre modelos.
Este lanzador es opcional: POETA solo necesita un llama-server escuchando en el puerto de
providers.yaml y cuyo alias coincida con el del modelo activo (llm.default).
"""

from __future__ import annotations

import argparse
import os
import shlex
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from poeta.config import ConfigError, load_config  # noqa: E402
from poeta.envfile import load_env_file  # noqa: E402

# clave en el bloque `server` del modelo -> flag de llama-server (se omite si el valor es null)
VALUE_FLAGS = (
    ("gpu_layers", "-ngl"),
    ("flash_attn", "--flash-attn"),
    ("batch", "-b"),
    ("ubatch", "-ub"),
    ("threads", "-t"),
    ("parallel", "--parallel"),
    ("reasoning", "--reasoning"),
)


def find_binary(llama_dir: Path) -> Path:
    candidate = llama_dir / "build" / "bin" / "llama-server"
    if candidate.is_file():
        return candidate
    found = [p for p in llama_dir.rglob("llama-server") if p.is_file()]
    if not found:
        raise FileNotFoundError(f"No se encontro llama-server bajo {llama_dir}")
    return found[0]


def build_command(cfg: dict, key: str | None = None) -> list[str]:
    local = cfg["models"]["llm"]["local"]
    key = key or cfg["models"]["llm"]["default"]
    if key not in local:
        raise ConfigError(f"El modelo '{key}' no existe en models.yaml (disponibles: {', '.join(local)}).")
    model = local[key]
    if not model.get("file"):
        raise ConfigError(f"El modelo '{key}' no tiene 'file' definido en models.yaml.")
    server = model.get("server")
    if not server:
        raise ConfigError(f"El modelo '{key}' no tiene bloque 'server' en models.yaml: define sus flags de llama-server.")

    paths = cfg["settings"]["paths"]
    model_path = Path(paths["models"]) / (model.get("dir") or "") / model["file"]
    if not model_path.is_file():
        raise FileNotFoundError(f"No existe el modelo: {model_path}")

    common = cfg["models"].get("llama_server", {})
    cmd = [
        str(find_binary(Path(paths["llama_cpp"]))),
        "-m", str(model_path),
        "-a", model.get("alias") or key,
        "--host", str(common.get("host", "127.0.0.1")),
        "--port", str(common.get("port", 8088)),
    ]
    if model.get("context") is not None:
        cmd += ["-c", str(model["context"])]
    for cfg_key, flag in VALUE_FLAGS:
        if server.get(cfg_key) is not None:
            cmd += [flag, str(server[cfg_key])]
    if server.get("jinja", True):
        cmd.append("--jinja")
    cmd += [str(a) for a in server.get("extra_args") or []]
    return cmd


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="Clave del modelo en models.yaml (por defecto, llm.default)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    load_env_file(ROOT / ".env")  # antes de load_config: las rutas pueden usar variables del .env
    try:
        cmd = build_command(load_config(), args.model)
    except (ConfigError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(shlex.join(cmd))
    if args.dry_run:
        return 0

    host = cmd[cmd.index("--host") + 1]
    port = int(cmd[cmd.index("--port") + 1])
    with socket.socket() as s:
        s.settimeout(1)
        if s.connect_ex((host, port)) == 0:
            print(
                f"Error: el puerto {port} ya esta en uso. Cambialo en config/models.yaml (llama_server.port) "
                f"y providers.yaml (local.base_url), o revisa quien lo usa: ss -ltnp | grep :{port}",
                file=sys.stderr,
            )
            return 1
    os.execv(cmd[0], cmd)


if __name__ == "__main__":
    sys.exit(main())
