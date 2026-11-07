#!/usr/bin/env python3
"""Lanza llama-server con el modelo local por defecto de config/models.yaml.

  python scripts/run_llama_server.py --dry-run   # solo imprime el comando
  python scripts/run_llama_server.py             # lo ejecuta (reemplaza este proceso)

Los flags de config/models.yaml (llama_server) reproducen el script que ya funciona en este equipo.
Este lanzador es opcional: POETA solo necesita un llama-server escuchando en el puerto de
providers.yaml y cuyo alias coincida con el de models.yaml.
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

# clave en models.yaml -> flag de llama-server (se omite si el valor es null)
VALUE_FLAGS = (
    ("context", "-c"),
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


def build_command(cfg: dict) -> list[str]:
    key = cfg["models"]["llm"]["default"]
    model = cfg["models"]["llm"]["local"][key]
    if not model.get("file"):
        raise ConfigError(f"El modelo '{key}' no tiene 'file' definido en models.yaml.")

    paths = cfg["settings"]["paths"]
    model_path = Path(paths["models"]) / (model.get("dir") or "") / model["file"]
    if not model_path.is_file():
        raise FileNotFoundError(f"No existe el modelo: {model_path}")

    srv = cfg["models"].get("llama_server", {})
    cmd = [
        str(find_binary(Path(paths["llama_cpp"]))),
        "-m", str(model_path),
        "-a", model.get("alias") or key,
        "--host", str(srv.get("host", "127.0.0.1")),
        "--port", str(srv.get("port", 8088)),
    ]
    for cfg_key, flag in VALUE_FLAGS:
        if srv.get(cfg_key) is not None:
            cmd += [flag, str(srv[cfg_key])]
    if srv.get("jinja", True):
        cmd.append("--jinja")
    cmd += [str(a) for a in srv.get("extra_args") or []]
    return cmd


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    try:
        cmd = build_command(load_config())
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
