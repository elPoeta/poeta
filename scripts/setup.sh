#!/usr/bin/env bash
# Fase 0: crea el entorno Python 3.12 independiente, instala el proyecto, corre pruebas y diagnostico.
# No toca el Python global ni otros proyectos.
set -euo pipefail

cd "$(dirname "$0")/.."

if ! command -v uv >/dev/null 2>&1; then
    echo "uv no encontrado. Instalalo antes de continuar." >&2
    exit 1
fi

uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev]"

echo
echo "== Pruebas =="
.venv/bin/python -m pytest -q

echo
echo "== Diagnostico =="
.venv/bin/python scripts/doctor.py || true

echo
echo "Fase 0 lista. Activa el entorno con: source .venv/bin/activate"
