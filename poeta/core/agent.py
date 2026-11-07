"""Agent Core (Fase 1): contexto conversacional y coordinacion con el Model Router.

Alcance actual: conversacion en streaming con historial en memoria.
Pendiente en fases siguientes: memoria persistente (Fase 3), voz (Fase 4),
ejecucion de herramientas MCP (Fase 6). Las llamadas a herramientas que devuelva
el modelo se emiten como evento pero todavia no se ejecutan.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from poeta.models import ModelRouter
from poeta.providers import ToolCall

log = logging.getLogger("poeta.core")


@dataclass
class AgentEvent:
    kind: str  # "text" | "reasoning" | "tool_calls" | "done"
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)


class AgentCore:
    def __init__(self, router: ModelRouter, system_prompt: str, max_history: int = 40) -> None:
        self.router = router
        self.system_prompt = system_prompt
        self.max_history = max_history
        self.history: list[dict[str, Any]] = []

    def reset(self) -> None:
        self.history.clear()

    def _messages(self) -> list[dict[str, Any]]:
        recent = self.history[-self.max_history :]
        return [{"role": "system", "content": self.system_prompt}, *recent]

    async def respond(
        self,
        user_text: str,
        tools: list[dict[str, Any]] | None = None,
        **params: Any,
    ) -> AsyncIterator[AgentEvent]:
        """Procesa un mensaje del usuario y emite eventos en streaming.

        Si el consumidor interrumpe (aclose / cancelacion), el texto ya generado se conserva
        en el historial como respuesta parcial: es lo que el usuario alcanzo a recibir.
        Si falla sin haber generado texto, el mensaje del usuario se retira del historial para
        poder reintentar sin duplicarlo.
        """
        self.history.append({"role": "user", "content": user_text})
        parts: list[str] = []
        try:
            async for chunk in self.router.stream_chat(self._messages(), tools=tools, **params):
                if chunk.reasoning:
                    yield AgentEvent("reasoning", text=chunk.reasoning)
                if chunk.text:
                    parts.append(chunk.text)
                    yield AgentEvent("text", text=chunk.text)
                if chunk.tool_calls:
                    yield AgentEvent("tool_calls", tool_calls=chunk.tool_calls)
        except Exception:
            if not parts:
                self.history.pop()
            raise
        finally:
            if parts:
                self.history.append({"role": "assistant", "content": "".join(parts)})
        yield AgentEvent("done")
