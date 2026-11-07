"""Model Router: seleccion manual de proveedor y modelo.

Garantias (plan secc. 5.1, 7.1 y 19):
- La seleccion es siempre manual: no hay sustitucion automatica entre proveedores.
- Si el proveedor falla, el error se propaga; nunca se reintenta contra otro proveedor.
- Un proveedor cloud no recibe nada sin autorizacion especifica de la operacion.
  En la Fase 1 la autorizacion es un callback minimo; la Fase 2 lo reemplaza por el Permission Manager.
"""

from __future__ import annotations

import inspect
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Awaitable, Callable

from poeta.providers import Chunk, MissingCredentials, OpenAICompatibleProvider, Provider

log = logging.getLogger("poeta.router")


class RouterError(Exception):
    pass


class ProviderDisabled(RouterError):
    """El proveedor no existe o esta deshabilitado en providers.yaml."""


class ModelNotSelected(RouterError):
    """No hay modelo definido para el proveedor activo."""


class CloudNotAuthorized(RouterError):
    """La transferencia a un proveedor cloud no fue autorizada."""


@dataclass
class TransferRequest:
    """Lo que se mostrara al usuario antes de enviar datos a un proveedor externo."""

    provider: str
    model: str
    operation: str
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None


Authorizer = Callable[[TransferRequest], "bool | Awaitable[bool]"]


class ModelRouter:
    def __init__(self, cfg: dict[str, dict[str, Any]], authorizer: Authorizer | None = None) -> None:
        self._authorizer = authorizer
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

    async def _authorize(self, req: TransferRequest) -> None:
        if self._authorizer is None:
            allowed = False
        else:
            result = self._authorizer(req)
            allowed = await result if inspect.isawaitable(result) else bool(result)
        if not allowed:
            log.warning(
                "transferencia cloud NO autorizada",
                extra={"provider": req.provider, "model": req.model, "operation": req.operation},
            )
            raise CloudNotAuthorized(f"Transferencia a '{req.provider}' no autorizada.")
        log.info(
            "transferencia cloud autorizada",
            extra={"provider": req.provider, "model": req.model, "operation": req.operation},
        )

    async def stream_chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **params: Any,
    ) -> AsyncIterator[Chunk]:
        provider = self.active
        if self._model is None:
            raise ModelNotSelected(f"Selecciona un modelo para '{provider.name}' (router.select(model=...)).")

        if provider.is_cloud:
            if not provider.api_key:
                key_env = self._provider_cfg[provider.name].get("api_key_env")
                raise MissingCredentials(f"Falta la variable de entorno {key_env} para '{provider.name}'.")
            await self._authorize(TransferRequest(provider.name, self._model, "chat", messages, tools))

        t0 = time.perf_counter()
        first: float | None = None
        status = "ok"
        try:
            async for chunk in provider.stream_chat(messages, self._model, tools=tools, **params):
                if first is None and (chunk.text or chunk.tool_calls):
                    first = time.perf_counter() - t0
                yield chunk
        except GeneratorExit:
            status = "interrumpido"
            raise
        except Exception:
            status = "error"
            raise
        finally:
            log.info(
                "chat finalizado",
                extra={
                    "provider": provider.name,
                    "model": self._model,
                    "status": status,
                    "first_token_s": round(first, 3) if first is not None else None,
                    "total_s": round(time.perf_counter() - t0, 3),
                },
            )
