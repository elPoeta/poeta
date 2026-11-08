"""Interfaz de embeddings, desacoplada del almacen (plan secc. 11.2).

Hoy la recuperacion usa solo FTS5. Cuando llegue la busqueda vectorial, un motor que cumpla este
protocolo se conecta sin cambiar la API de MemoryStore. Nada la usa todavia.
"""

from __future__ import annotations

from typing import Protocol


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Un vector de `dim` floats por texto, en el mismo orden."""
        ...
