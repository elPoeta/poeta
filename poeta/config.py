"""Carga y validacion de la configuracion YAML de P.O.E.T.A.

Valida los invariantes de privacidad del plan:
- cloud_fallback debe ser false (nunca hay sustitucion automatica a cloud).
- cloud.require_confirmation debe ser true (toda transferencia externa se confirma).
- El proveedor por defecto debe existir y no puede ser cloud.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

CONFIG_FILES = (
    "settings",
    "providers",
    "models",
    "voice",
    "vision",
    "audio",
    "mcp",
    "permissions",
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


class ConfigError(Exception):
    """Configuracion ausente o invalida."""


def _expand_str(value: str) -> str:
    def sub(m: re.Match[str]) -> str:
        env = os.environ.get(m.group(1))
        if env:
            return env
        if m.group(2) is not None:
            return m.group(2)
        raise ConfigError(f"Variable de entorno sin definir y sin valor por defecto: {m.group(1)}")

    return os.path.expanduser(_VAR.sub(sub, value))


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        return _expand_str(value)
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


def _validate(cfg: dict[str, dict[str, Any]]) -> None:
    providers_cfg = cfg["providers"]
    defaults = providers_cfg.get("defaults") or {}
    providers = providers_cfg.get("providers") or {}

    if defaults.get("cloud_fallback") is not False:
        raise ConfigError("providers.defaults.cloud_fallback debe ser false: no hay fallback automatico a cloud.")

    default_name = defaults.get("provider")
    if default_name not in providers:
        raise ConfigError(f"El proveedor por defecto '{default_name}' no esta definido en providers.")
    if providers[default_name].get("cloud") is not False:
        raise ConfigError("El proveedor por defecto debe ser local (cloud: false).")

    for name, p in providers.items():
        if "cloud" not in p:
            raise ConfigError(f"El proveedor '{name}' debe declarar explicitamente cloud: true|false.")
        if p["cloud"] and "api_key" in p:
            raise ConfigError(f"El proveedor cloud '{name}' no puede llevar api_key en el YAML; usa api_key_env.")

    cloud_perms = (cfg["permissions"].get("cloud") or {})
    if cloud_perms.get("require_confirmation") is not True:
        raise ConfigError("permissions.cloud.require_confirmation debe ser true.")

    name = (cfg["settings"].get("assistant") or {}).get("name")
    if not isinstance(name, str) or not name.strip():
        raise ConfigError("settings.assistant.name debe ser un texto no vacio.")


DEFAULT_SYSTEM_PROMPT = "Eres {assistant_name}, un asistente personal."


def assistant_name(cfg: dict[str, dict[str, Any]]) -> str:
    """Nombre del asistente segun settings.assistant.name."""
    return cfg["settings"]["assistant"]["name"].strip()


def system_prompt(cfg: dict[str, dict[str, Any]]) -> str:
    """System prompt con {assistant_name} reemplazado. Usa replace (no format) para tolerar otras llaves."""
    template = (cfg["settings"].get("agent") or {}).get("system_prompt") or DEFAULT_SYSTEM_PROMPT
    return " ".join(template.split()).replace("{assistant_name}", assistant_name(cfg))


def load_config(config_dir: str | Path | None = None) -> dict[str, dict[str, Any]]:
    """Carga todos los YAML, expande variables de entorno y valida invariantes."""
    base = Path(config_dir) if config_dir else PROJECT_ROOT / "config"
    cfg: dict[str, dict[str, Any]] = {}
    for name in CONFIG_FILES:
        path = base / f"{name}.yaml"
        if not path.is_file():
            raise ConfigError(f"Falta el archivo de configuracion: {path}")
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            raise ConfigError(f"YAML invalido en {path}: {exc}") from exc
        if not isinstance(data, dict):
            raise ConfigError(f"{path} debe contener un mapa en el nivel superior.")
        cfg[name] = _expand(data)
    _validate(cfg)
    return cfg
