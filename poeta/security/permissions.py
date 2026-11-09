"""Permission Manager (plan secc. 10.3, 12.2, 19).

Punto unico de decision para toda accion con impacto: transferencias cloud, archivos, comandos,
herramientas. Principios:

- Falla cerrado: accion no catalogada, sin interfaz de confirmacion, error, tiempo agotado o fallo
  al escribir la auditoria => se deniega.
- La autorizacion la valida el nucleo, no el modelo: la confirmacion solo puede venir del
  `confirmer` (la interfaz con el usuario). Nada que el modelo escriba cuenta como aprobacion.
- Cada operacion se autoriza por separado: no hay aprobaciones que se reutilicen.
- Los niveles recomendados (RECOMMENDED) se pueden bajar desde el YAML: es tu equipo y tu decision.
  Hacerlo genera un aviso con el nivel sugerido (permission_warnings); solo se rechazan los
  errores estructurales y `privileged`, que siempre es `never`.
- La politica se copia al crear el manager: modificar el dict de configuracion despues no la altera.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import inspect
import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

from .audit import AuditLog

log = logging.getLogger("poeta.permissions")

AUTO, CONFIRM, EXPLICIT, NEVER = "auto", "confirm", "confirm_explicit", "never"
_POLICIES = (AUTO, CONFIRM, EXPLICIT)
_SEVERITY = {AUTO: 0, CONFIRM: 1, EXPLICIT: 2}

# Nivel de riesgo recomendado por accion y por que. El YAML puede bajarlo (con aviso) o subirlo.
RECOMMENDED = {
    "modify_file": (3, "modifica archivos tuyos"),
    "run_command": (3, "ejecuta comandos en tu equipo"),
    "install_package": (3, "instala software en tu equipo"),
    "cloud_transfer": (3, "envia tu texto a un proveedor externo"),
    "delete_data": (4, "borra datos de forma irreversible"),
}
# Acciones que operan sobre una ruta (details["path"]).
PATH_ACTIONS = {"read_authorized_file", "create_document", "modify_file", "delete_data"}

Confirmer = Callable[["ActionRequest", bool], "bool | Awaitable[bool]"]


class PermissionConfigError(Exception):
    """permissions.yaml viola un invariante de seguridad."""


@dataclass
class ActionRequest:
    kind: str                                   # clave de `actions` en permissions.yaml
    description: str                            # frase que se le muestra al usuario
    details: dict[str, Any] = field(default_factory=dict)  # metadatos: se muestran y se auditan (sin contenido)
    payload: Any = None                         # contenido exacto a enviar/ejecutar: se muestra, NUNCA se audita
    origin: str = "user"                        # "user" | "model" | "tool"
    tainted: bool = False                       # la accion se origino tras leer contenido no confiable (web, herramientas)


@dataclass
class Decision:
    allowed: bool
    reason: str
    level: int | None
    policy: str
    request_id: str
    digest: str | None


@dataclass
class _Resolution:
    policy: str
    level: int | None
    deny_reason: str | None = None


def validate_config(perm: dict[str, Any]) -> list[str]:
    """Rechaza lo estructuralmente invalido y devuelve avisos por las relajaciones de seguridad."""
    levels = perm.get("risk_levels") or {}
    for lvl in (1, 2, 3, 4):
        policy = (levels.get(lvl) or {}).get("policy")
        if policy not in _POLICIES:
            raise PermissionConfigError(f"risk_levels.{lvl}.policy debe ser uno de {list(_POLICIES)}.")
    if "privileged_operations" not in (perm.get("never_automatic") or []):
        raise PermissionConfigError("never_automatic debe incluir privileged_operations.")

    actions = perm.get("actions") or {}
    if actions.get("privileged") != NEVER:
        raise PermissionConfigError("actions.privileged debe ser never.")
    for kind, value in actions.items():
        if value != NEVER and (isinstance(value, bool) or value not in (1, 2, 3, 4)):
            raise PermissionConfigError(f"actions.{kind} invalido: {value!r} (usa 1-4 o never).")
    rc = (perm.get("cloud") or {}).get("require_confirmation", True)
    if not isinstance(rc, bool):
        raise PermissionConfigError("cloud.require_confirmation debe ser true o false.")
    return permission_warnings(perm)


def permission_warnings(perm: dict[str, Any]) -> list[str]:
    """Avisos (con el nivel sugerido) por cada ajuste mas permisivo que lo recomendado."""
    out: list[str] = []
    levels = perm.get("risk_levels") or {}
    if (levels.get(3) or {}).get("policy") == AUTO:
        out.append("risk_levels.3.policy = auto: modificar archivos, ejecutar comandos, instalar y enviar a la nube "
                   "ocurririan sin pedirte permiso. Recomendado: confirm.")
    if (levels.get(4) or {}).get("policy") != EXPLICIT:
        out.append(f"risk_levels.4.policy = {(levels.get(4) or {}).get('policy')}: las acciones irreversibles "
                   "deberian exigir escribir CONFIRMAR. Recomendado: confirm_explicit.")
    if (perm.get("cloud") or {}).get("require_confirmation", True) is False:
        out.append("cloud.require_confirmation = false: cada mensaje se envia al proveedor cloud sin que lo veas ni lo "
                   "apruebes. Recomendado: true (o conservarlo y usar actions.cloud_transfer solo si confias en el proveedor).")
    for kind, (minimum, why) in RECOMMENDED.items():
        value = (perm.get("actions") or {}).get(kind)
        if value is not None and value != NEVER and value < minimum:
            out.append(f"actions.{kind} = {value}: {why}. Recomendado: {minimum}.")
    return out


def _digest(payload: Any) -> str | None:
    if payload is None:
        return None
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def chat_payload_summary(
    messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None, params: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Resumen de lo que contiene una peticion de chat (para mostrar y auditar, sin el contenido)."""
    by_role: dict[str, int] = {}
    chars = media = 0
    for m in messages:
        role = str(m.get("role", "?"))
        by_role[role] = by_role.get(role, 0) + 1
        content = m.get("content")
        if isinstance(content, str):
            chars += len(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    chars += len(str(part.get("text", "")))
                else:
                    media += 1  # imagen, audio u otro contenido no textual
        for tc in m.get("tool_calls") or []:
            chars += len(json.dumps(tc, default=str))
    return {
        "messages": len(messages),
        "by_role": by_role,
        "chars": chars,
        "media_parts": media,
        "tool_results": by_role.get("tool", 0),
        "tools_defined": len(tools or []),
        "params": sorted((params or {}).keys()),
    }


def _within(path: Path, base: Path) -> bool:
    return path == base or base in path.parents


class PermissionManager:
    def __init__(
        self,
        cfg: dict[str, Any],
        confirmer: Confirmer | None = None,
        audit: AuditLog | None = None,
        root: Path | str | None = None,
    ) -> None:
        perm = copy.deepcopy(cfg["permissions"])
        self.warnings: list[str] = validate_config(perm)
        self._cloud_confirm = (perm.get("cloud") or {}).get("require_confirmation", True) is not False
        self._root = Path(root) if root else Path(__file__).resolve().parents[2]
        self._levels = {int(k): v["policy"] for k, v in perm["risk_levels"].items()}
        self._actions: dict[str, Any] = dict(perm["actions"])
        fa = perm.get("file_access") or {}
        self._allowed = [self._abs(p) for p in fa.get("allowed_dirs") or []]
        self._auto_create = [self._abs(p) for p in fa.get("auto_create_dirs") or []]
        self._protected = [self._abs(p) for p in fa.get("protected") or []]
        self._allowlist = set((perm.get("commands") or {}).get("allowlist") or [])
        self._timeout = perm.get("confirmation_timeout_s")
        self._confirmer = confirmer
        self.audit = audit

    def _abs(self, p: str) -> Path:
        path = Path(str(p)).expanduser()
        return (path if path.is_absolute() else self._root / path).resolve()

    # -- resolucion de politica ---------------------------------------------

    def _deny(self, reason: str, level: int | None = None) -> _Resolution:
        return _Resolution(NEVER, level, reason)

    def _check_path(self, raw: Any) -> tuple[str | None, Path | None]:
        if not isinstance(raw, str) or not raw:
            return "falta la ruta (details.path)", None
        p = Path(raw).expanduser()
        if not p.is_absolute():
            return "la ruta debe ser absoluta", None
        p = p.resolve()  # sigue enlaces simbolicos: un enlace dentro de una carpeta permitida no escapa
        if any(_within(p, prot) for prot in self._protected):
            return "ruta protegida (configuracion, registros o secretos)", None
        if not any(_within(p, ok) for ok in self._allowed):
            return "ruta fuera de las carpetas autorizadas (file_access.allowed_dirs)", None
        return None, p

    def _check_command(self, argv: Any) -> str | None:
        if not isinstance(argv, list) or not argv or not all(isinstance(a, str) for a in argv):
            return "el comando debe darse como lista argv, nunca como texto de shell"
        # Coincidencia exacta con nombres sin ruta: "/tmp/x/ls" nunca coincide con "ls".
        if argv[0] not in self._allowlist:
            return f"comando '{argv[0]}' fuera de la lista permitida (commands.allowlist)"
        return None

    def _resolve(self, req: ActionRequest) -> _Resolution:
        spec = self._actions.get(req.kind)
        if spec is None:
            return self._deny(f"accion no catalogada: '{req.kind}'")
        if spec == NEVER:
            return self._deny("operacion que nunca se autoriza automaticamente")
        level = int(spec)

        if req.kind in PATH_ACTIONS:
            err, path = self._check_path(req.details.get("path"))
            if err or path is None:
                return self._deny(err or "ruta invalida", level)
            if req.kind == "create_document":
                if path.exists():  # sobrescribir equivale a modificar
                    modify = self._actions.get("modify_file", 3)
                    if modify == NEVER:
                        return self._deny("modificar archivos esta deshabilitado", level)
                    level = int(modify)
                elif any(_within(path, d) for d in self._auto_create):
                    level = 1

        if req.kind == "run_command":
            err = self._check_command(req.details.get("argv"))
            if err:
                return self._deny(err, level)

        policy = self._levels[level]

        if req.kind == "cloud_transfer":
            if not isinstance(req.payload, dict) or "messages" not in req.payload:
                return self._deny("transferencia sin contenido que mostrar al usuario", level)
            # El resumen se recalcula desde el contenido real: no se confia en lo que declare el solicitante.
            summary = chat_payload_summary(req.payload["messages"], req.payload.get("tools"), req.payload.get("params"))
            if not self._cloud_confirm:  # decision explicita del usuario (ver permission_warnings)
                policy = AUTO
            if summary["media_parts"] > 0:  # imagenes o audio: nunca con confirmacion simple
                policy = EXPLICIT

        if req.tainted and level >= 3:
            policy = max(policy, EXPLICIT, key=_SEVERITY.get)
        return _Resolution(policy, level)

    # -- confirmacion -----------------------------------------------------------

    async def _call_confirmer(self, req: ActionRequest, explicit: bool) -> Any:
        fn = self._confirmer
        assert fn is not None
        if inspect.iscoroutinefunction(fn) or inspect.iscoroutinefunction(getattr(fn, "__call__", None)):
            return await fn(req, explicit)
        result = await asyncio.to_thread(fn, req, explicit)
        return await result if inspect.isawaitable(result) else result

    async def _ask(self, req: ActionRequest, explicit: bool) -> tuple[bool, str]:
        if self._confirmer is None:
            return False, "no hay interfaz de confirmacion disponible"
        try:
            call = self._call_confirmer(req, explicit)
            answer = await (asyncio.wait_for(call, self._timeout) if self._timeout else call)
        except asyncio.TimeoutError:
            return False, "tiempo de confirmacion agotado"
        except Exception as exc:  # noqa: BLE001 - cualquier fallo del confirmador es una denegacion
            return False, f"error en la confirmacion: {type(exc).__name__}"
        # Solo un True explicito autoriza: un valor "verdadero" cualquiera (1, "si", objeto) no basta.
        if answer is True:
            return True, "autorizado por el usuario" + (" (confirmacion explicita)" if explicit else "")
        return False, "rechazado por el usuario"

    # -- auditoria ----------------------------------------------------------------

    def _record(self, event: str, **fields: Any) -> bool:
        if self.audit is None:
            return True
        try:
            self.audit.record(event, **fields)
            return True
        except Exception:  # noqa: BLE001
            log.exception("no se pudo escribir en la auditoria")
            return False

    # -- API publica ------------------------------------------------------------------

    async def check(self, req: ActionRequest) -> Decision:
        rid = uuid.uuid4().hex[:12]
        digest = _digest(req.payload)
        res = self._resolve(req)
        explicit = res.policy == EXPLICIT
        base: dict[str, Any] = {
            "request_id": rid, "kind": req.kind, "origin": req.origin, "tainted": req.tainted,
            "level": res.level, "policy": res.policy, "details": req.details, "digest": digest,
        }

        if res.policy == NEVER:
            allowed, reason = False, res.deny_reason or "denegado por politica"
        elif res.policy == AUTO:
            allowed, reason = True, "politica automatica"
        else:
            if not self._record("permission_requested", **base):
                allowed, reason = False, "no se pudo escribir el registro de auditoria"
            else:
                allowed, reason = await self._ask(req, explicit)

        if allowed and not self._record("permission_decided", allowed=True, reason=reason, **base):
            allowed, reason = False, "no se pudo escribir el registro de auditoria"
        if not allowed:
            self._record("permission_decided", allowed=False, reason=reason, **base)

        (log.info if allowed else log.warning)(
            "permiso %s", "concedido" if allowed else "DENEGADO",
            extra={"kind": req.kind, "reason": reason, "request_id": rid},
        )
        return Decision(allowed, reason, res.level, res.policy, rid, digest)
