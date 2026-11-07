"""Interfaz comun de proveedores de modelos y tipos compartidos."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator


class ProviderError(Exception):
    """Error devuelto por el proveedor o durante la comunicacion."""

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class ProviderUnavailable(ProviderError):
    """No se pudo conectar con el proveedor."""


class MissingCredentials(ProviderError):
    """Falta la clave de API del proveedor (variable de entorno sin definir)."""


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # JSON crudo tal como lo envio el modelo; se valida al ejecutar


@dataclass
class Chunk:
    """Fragmento de una respuesta en streaming."""

    text: str = ""
    reasoning: str = ""  # razonamiento interno del modelo: nunca se debe hablar ni mostrar como respuesta
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None


class Provider(ABC):
    name: str
    is_cloud: bool

    @abstractmethod
    def stream_chat(
        self,
        messages: list[dict[str, Any]],
        model: str,
        tools: list[dict[str, Any]] | None = None,
        **params: Any,
    ) -> AsyncIterator[Chunk]:
        """Devuelve un iterador asincrono de Chunk."""

    @abstractmethod
    async def list_models(self) -> list[str]:
        """Identificadores de modelos que expone el proveedor."""

    async def health(self) -> bool:
        try:
            await self.list_models()
            return True
        except ProviderError:
            return False
