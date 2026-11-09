"""Rutinas: secuencias fijas de herramientas definidas por el usuario en config/automations.yaml.

El modelo solo elige el NOMBRE de una rutina; los pasos y argumentos son los del YAML (no los modifica). Ejecutar una
rutina pide confirmacion mostrando todos los pasos, y ademas cada paso pasa por su propia comprobacion de permisos.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from poeta.providers import ToolCall
from poeta.tools import Tool, ToolError, ToolRegistry

_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,40}$")
MAX_STEPS = 20
RESERVED = {"run_automation"}


class RoutineConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Step:
    tool: str
    args: dict[str, Any]


@dataclass(frozen=True)
class Routine:
    name: str
    description: str
    steps: tuple[Step, ...]
    stop_on_error: bool = True


def load_routines(cfg: dict[str, Any]) -> dict[str, Routine]:
    out: dict[str, Routine] = {}
    for name, r in ((cfg.get("automations") or {}).get("automations") or {}).items():
        if not isinstance(name, str) or not _NAME.match(name):
            raise RoutineConfigError(f"nombre de rutina invalido: {name!r} (minusculas, numeros, - y _)")
        if not isinstance(r, dict):
            raise RoutineConfigError(f"{name}: debe ser un mapa")
        steps = r.get("steps")
        if not isinstance(steps, list) or not 1 <= len(steps) <= MAX_STEPS:
            raise RoutineConfigError(f"{name}: steps debe ser una lista de 1 a {MAX_STEPS} pasos")
        parsed = []
        for i, s in enumerate(steps, 1):
            if not isinstance(s, dict) or not isinstance(s.get("tool"), str) or not isinstance(s.get("args", {}), dict):
                raise RoutineConfigError(f"{name}, paso {i}: debe tener 'tool' (texto) y 'args' (mapa)")
            if s["tool"] in RESERVED:
                raise RoutineConfigError(f"{name}, paso {i}: una rutina no puede ejecutar '{s['tool']}'")
            parsed.append(Step(s["tool"], dict(s.get("args") or {})))
        desc = r.get("description", "")
        if not isinstance(desc, str):
            raise RoutineConfigError(f"{name}: description debe ser texto")
        out[name] = Routine(name, " ".join(desc.split())[:200], tuple(parsed), r.get("stop_on_error", True) is not False)
    return out


def _fmt(step: Step) -> str:
    return f"{step.tool}({json.dumps(step.args, ensure_ascii=False)})"


def register_routines(registry: ToolRegistry, routines: dict[str, Routine]) -> tuple[list[str], dict[str, str]]:
    """Registra `run_automation` con las rutinas cuyos pasos usan herramientas disponibles."""
    usable: dict[str, Routine] = {}
    skipped: dict[str, str] = {}
    for name, r in routines.items():
        missing = [s.tool for s in r.steps if registry.get(s.tool) is None]
        if missing:
            skipped[name] = f"usa herramientas no disponibles: {sorted(set(missing))}"
        else:
            usable[name] = r
    if not usable:
        return [], skipped

    def describe(a: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        r = usable.get(a["name"])
        if r is None:
            raise ToolError(f"no existe la rutina '{a['name']}'. Disponibles: {', '.join(sorted(usable))}")
        return (f"Ejecutar la rutina '{r.name}': " + "; ".join(_fmt(s) for s in r.steps),
                {"routine": r.name, "steps": [s.tool for s in r.steps]})

    async def run(a: dict[str, Any]) -> str:
        r = usable[a["name"]]
        lines: list[str] = []
        for i, step in enumerate(r.steps, 1):
            res = await registry.execute(ToolCall(id=f"routine-{r.name}-{i}", name=step.tool,
                                                  arguments=json.dumps(step.args)))
            status = "denegado" if res.denied else ("ok" if res.ok else "error")
            first = res.text.strip().splitlines()[-1][:120] if res.text.strip() else ""
            lines.append(f"{i}. {step.tool}: {status}" + (f" - {first}" if not res.ok else ""))
            if not res.ok and r.stop_on_error:
                lines.append("Rutina detenida en este paso.")
                break
        return "\n".join(lines)

    registry.register(Tool(
        "run_automation",
        "Ejecuta una rutina predefinida por el usuario. Rutinas: " + "; ".join(
            f"{n} ({r.description or len(r.steps)} pasos)" for n, r in sorted(usable.items())),
        {"type": "object", "properties": {"name": {"type": "string", "enum": sorted(usable)}},
         "required": ["name"], "additionalProperties": False},
        run, "run_automation", describe=describe, source="desktop", timeout_s=300))
    return sorted(usable), skipped
