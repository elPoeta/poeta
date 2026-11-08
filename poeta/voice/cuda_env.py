"""Precarga de las librerias CUDA 12 runtime instaladas como wheels de pip.

CTranslate2 (motor de faster-whisper) se compila contra CUDA 12 aunque el
sistema tenga otro toolkit (por ejemplo 13.x, que trae libcublas.so.13). La
solucion probada en vozboard: instalar nvidia-cublas-cu12, nvidia-cudnn-cu12 y
nvidia-cuda-runtime-cu12 y precargarlas con ctypes ANTES de crear el modelo
(cambiar LD_LIBRARY_PATH a mitad de proceso no sirve: glibc lo lee al inicio).
"""

from __future__ import annotations

import ctypes
import importlib.util
import os
import sys

_CUDA_LIB_MODULES = ("nvidia.cublas.lib", "nvidia.cudnn.lib", "nvidia.cuda_runtime.lib")


def _find_module_dirs() -> list[str]:
    dirs: list[str] = []
    for name in _CUDA_LIB_MODULES:
        try:
            spec = importlib.util.find_spec(name)
        except ModuleNotFoundError:  # el paquete padre 'nvidia' no existe
            continue
        if spec is None or not spec.submodule_search_locations:
            continue
        for loc in spec.submodule_search_locations:
            if loc not in dirs:
                dirs.append(loc)
    return dirs


def _preload(lib_dirs: list[str]) -> list[str]:
    pending = [
        os.path.join(d, e)
        for d in lib_dirs
        for e in sorted(os.listdir(d))
        if e.startswith("lib") and ".so." in e
    ]
    loaded: list[str] = []
    while pending:
        progressed = False
        remaining: list[str] = []
        for path in pending:
            try:
                ctypes.CDLL(path)
                loaded.append(os.path.basename(path))
                progressed = True
            except OSError:  # dependencia aun no cargada: reintentar
                remaining.append(path)
        pending = remaining
        if not progressed:
            break
    return loaded


def patch_cuda_libs() -> list[str]:
    """Devuelve las carpetas encontradas (vacio si no hay wheels o no es Linux)."""
    if sys.platform != "linux":
        return []
    lib_dirs = _find_module_dirs()
    if not lib_dirs:
        return []
    _preload(lib_dirs)
    current = os.environ.get("LD_LIBRARY_PATH", "")
    parts = current.split(os.pathsep) if current else []
    new = [d for d in lib_dirs if d not in parts]
    if new:
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(new + parts)
    return lib_dirs
