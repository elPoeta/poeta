"""Model Router: seleccion manual de proveedor y modelo.

Garantias (plan secc. 5.1, 7.1 y 19):
- La seleccion es siempre manual: no hay sustitucion automatica entre proveedores.
- Si el proveedor falla, el error se propaga; nunca se reintenta contra otro proveedor.
- Un proveedor cloud no recibe nada sin autorizacion especifica de la operacion, decidida por el
  Permission Manager. Sin manager o sin interfaz de confirmacion, se deniega.
- Lo que se muestra al usuario es exactamente lo que se envia: el contenido se copia antes de pedir
  la autorizacion y se envia esa copia, aunque el original cambie mientras se espera.
"""

from __future__ import annotations

import copy
import logging
import os
import time
from typing import Any, AsyncIterator

from poeta.providers import Chunk, MissingCredentials, OpenAICompatibleProvider, Provider
from poeta.security import ActionRequest, Decision, PermissionManager, chat_payload_summary

log = logging.getLogger("poeta.router")


class RouterError(Exception):
    pass


class ProviderDisabled(RouterError):
    """El proveedor no existe o esta deshabilitado en providers.yaml."""


class ModelNotSelected(RouterError):
    """No hay modelo definido para el proveedor activo."""


class CloudNotAuthorized(RouterError):
    """La transferencia a un proveedor cloud no fue autorizada."""


class ModelRouter:
    def __init__(self, cfg: dict[str, dict[str, Any]], permissions: PermissionManager | None = None) -> None:
        # Sin manager explicito se crea uno sin confirmador: todo lo que requiera confirmacion se deniega.
        self._permissions = permissions or PermissionManager(cfg)
        self._provider_cfg: dict[str, dict[str, Any]] = cfg["providers"]["providers"]
        llm = cfg["models"].get("llm", {})
        default_key: str | None = llm.get("default")
        local_entry = (llm.get("local") or {}).get(default_key) or {}
        # Nombre con que llama-server expone el modelo (-a); si no hay alias, la clave del catalogo.
        self._local_model_name: str | None = local_entry.get("alias") or default_key
        self._cloud_models: dict[str, str | None] = {
            name: (c or {}).get("model") for name, c in (llm.get("cloud") or {}).items()
        }

        self._providers: dict[str, Provider] = {}
        for name, p in self._provider_cfg.items():
            if not p.get("enabled"):
                continue
            key_env = p.get("api_key_env")
            api_key = os.environ.get(key_env) if key_env else p.get("api_key")
            self._providers[name] = OpenAICompatibleProvider(
                name=name, base_url=p["base_url"], api_key=api_key or None, cloud=bool(p["cloud"])
            )

        self._provider_name: str = cfg["providers"]["defaults"]["provider"]
        self._model: str | None = self._default_model(self._provider_name)

    # -- seleccion manual ---------------------------------------------------

    def _default_model(self, provider: str) -> str | None:
        if self._provider_cfg.get(provider, {}).get("cloud"):
            return self._cloud_models.get(provider)
        return self._local_model_name

    def select(self, provider: str | None = None, model: str | None = None) -> None:
        if provider is not None:
            if provider not in self._providers:
                raise ProviderDisabled(f"Proveedor '{provider}' no existe o esta deshabilitado.")
            if provider != self._provider_name:
                self._provider_name = provider
                self._model = self._default_model(provider)
        if model is not None:
            self._model = model
        log.info(
            "seleccion de modelo",
            extra={"provider": self._provider_name, "model": self._model},
        )

    @property
    def active(self) -> Provider:
        return self._providers[self._provider_name]

    @property
    def active_model(self) -> str | None:
        return self._model

    def status(self) -> dict[str, Any]:
        return {
            "provider": self._provider_name,
            "model": self._model,
            "cloud": self.active.is_cloud,
            "providers": sorted(self._providers),
        }

    # -- inferencia -----------------------------------------------------------

    async def _authorize_cloud(
        self,
        provider: str,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        params: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None, dict[str, Any], Decision]:
        """Pide autorizacion y devuelve la copia exacta que se mostro, que es la que se debe enviar."""
        snapshot = copy.deepcopy({"messages": messages, "tools": tools, "params": params})
        summary = chat_payload_summary(snapshot["messages"], snapshot["tools"], snapshot["params"])
        request = ActionRequest(
            kind="cloud_transfer",
            description=f"Enviar la conversacion a {provider} ({model})",
            details={"provider": provider, "model": model, "operation": "chat", **summary},
            payload=snapshot,
        )
        decision = await self._permissions.check(request)
        if not decision.allowed:
            raise CloudNotAuthorized(f"Transferencia a '{provider}' no autorizada: {decision.reason}.")
        return snapshot["messages"], snapshot["tools"], snapshot["params"], decision

    async def stream_chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **params: Any,
    ) -> AsyncIterator[Chunk]:
        provider = self.active
        if self._model is None:
            raise ModelNotSelected(f"Selecciona un modelo para '{provider.name}' (router.select(model=...)).")

        decision: Decision | None = None
        if provider.is_cloud:
            if not provider.api_key:
                key_env = self._provider_cfg[provider.name].get("api_key_env")
                raise MissingCredentials(
                    f"Falta la variable {key_env} para '{provider.name}'. "
                    f"Definela en el entorno o en el archivo .env de la raiz del proyecto y reinicia."
                )
            messages, tools, params, decision = await self._authorize_cloud(
                provider.name, self._model, messages, tools, params
            )

        t0 = time.perf_counter()
        first: float | None = None
        status = "ok"
        error: str | None = None
        try:
            async for chunk in provider.stream_chat(messages, self._model, tools=tools, **params):
                if first is None and (chunk.text or chunk.tool_calls):
                    first = time.perf_counter() - t0
                yield chunk
        except GeneratorExit:
            status = "interrumpido"
            raise
        except Exception as exc:
            status = "error"
            error = type(exc).__name__
            raise
        finally:
            total = round(time.perf_counter() - t0, 3)
            log.info(
                "chat finalizado",
                extra={
                    "provider": provider.name,
                    "model": self._model,
                    "status": status,
                    "first_token_s": round(first, 3) if first is not None else None,
                    "total_s": total,
                },
            )
            if decision is not None and self._permissions.audit is not None:
                try:
                    self._permissions.audit.record(
                        "cloud_transfer_result",
                        request_id=decision.request_id,
                        digest=decision.digest,
                        provider=provider.name,
                        model=self._model,
                        status=status,
                        error=error,
                        total_s=total,
                    )
                except Exception:  # noqa: BLE001 - el resultado ya ocurrio; no se oculta la respuesta
                    log.exception("no se pudo registrar el resultado de la transferencia")
