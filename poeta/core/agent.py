"""Agent Core: contexto conversacional, memoria y coordinacion con el Model Router.

Conversacion en streaming con historial en memoria, memoria persistente opcional y herramientas.

Herramientas (Fase 6): si hay un ToolRegistry, el modelo puede encadenar hasta `max_tool_rounds`
llamadas por turno. Cada llamada pasa por el Permission Manager (en el registro). Los mensajes
intermedios de herramientas viven solo durante el turno: al historial y a la memoria llegan
unicamente el mensaje del usuario y la respuesta final. Sin registro, las llamadas del modelo se
emiten como evento "tool_calls" y no se ejecutan.

Memoria (Fase 3):
- Cada intercambio terminado (o cortado a mitad) se guarda al finalizar el turno, salvo en incognito.
  Un turno que falla sin producir texto no se guarda.
- Antes de cada turno se busca memoria relevante y se agrega al PROMPT DEL SISTEMA (un solo mensaje de
  sistema: algunas plantillas de chat rechazan varios). No se guarda en el historial.
- Con un proveedor cloud la memoria no se inyecta salvo `memory.context.inject_into_cloud: true`.
- Un fallo de la memoria nunca interrumpe la conversacion: se registra y se sigue.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from poeta.memory import MemoryStore, build_memory_block
from poeta.models import ModelRouter
from poeta.providers import ToolCall
from poeta.tools import ToolRegistry, ToolResult

log = logging.getLogger("poeta.core")


@dataclass
class AgentEvent:
    kind: str  # "text" | "reasoning" | "tool_calls" | "tool_start" | "tool_result" | "done"
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    result: ToolResult | None = None  # solo en "tool_result"


class AgentCore:
    def __init__(
        self,
        router: ModelRouter,
        system_prompt: str,
        max_history: int = 40,
        memory: MemoryStore | None = None,
        memory_cfg: dict[str, Any] | None = None,
        tools: ToolRegistry | None = None,
        max_tool_rounds: int = 6,
    ) -> None:
        self.router = router
        self.system_prompt = system_prompt
        self.max_history = max_history
        self.history: list[dict[str, Any]] = []
        self.memory = memory
        self._mcfg = memory_cfg or {}
        self.conversation_id: int | None = None
        self.incognito = False
        self.memory_errors = 0
        self.tools = tools
        self.max_tool_rounds = max_tool_rounds

    # -- conversaciones --------------------------------------------------------------

    def reset(self) -> None:
        """Borra el historial y la proxima conversacion guardada sera una nueva."""
        self.history.clear()
        self.conversation_id = None

    def resume(self, conversation_id: int) -> int:
        """Retoma una conversacion guardada: carga sus ultimos mensajes. Devuelve cuantos."""
        if self.memory is None:
            raise RuntimeError("La memoria esta deshabilitada.")
        if not self.memory.conversation_exists(conversation_id):
            raise LookupError(f"No existe la conversacion #{conversation_id}.")
        limit = int(self._mcfg.get("resume_last_messages", 20))
        messages = self.memory.get_messages(conversation_id, limit=limit)
        self.history = [{"role": m["role"], "content": m["content"]} for m in messages]
        self.conversation_id = conversation_id
        return len(messages)

    # -- construccion del prompt ----------------------------------------------------------

    def _memory_block(self, user_text: str) -> str | None:
        ctx = self._mcfg.get("context") or {}
        if self.memory is None or not ctx.get("inject", True):
            return None
        if self.router.active.is_cloud and not ctx.get("inject_into_cloud", False):
            return None
        try:
            return build_memory_block(self.memory, user_text, self.conversation_id, ctx)
        except Exception:  # noqa: BLE001 - sin memoria el asistente sigue funcionando
            self.memory_errors += 1
            log.exception("no se pudo leer la memoria")
            return None

    def _messages(self, user_text: str) -> list[dict[str, Any]]:
        system = self.system_prompt
        block = self._memory_block(user_text)
        if block:
            system = f"{system}\n\n{block}"
        recent = self.history[-self.max_history :]
        return [{"role": "system", "content": system}, *recent]

    def _persist(self, user_text: str, answer: str, partial: bool) -> None:
        if self.memory is None or self.incognito or not self._mcfg.get("save_conversations", True):
            return
        try:
            if self.conversation_id is None:
                self.conversation_id = self.memory.start_conversation()
            status = self.router.status()
            self.memory.add_exchange(
                self.conversation_id, user_text, answer, provider=status["provider"], model=status["model"], partial=partial
            )
        except Exception:  # noqa: BLE001
            self.memory_errors += 1
            log.exception("no se pudo guardar el intercambio en la memoria")

    # -- turno -----------------------------------------------------------------------------------

    def _log_tool(self, call: ToolCall, result: ToolResult) -> None:
        if self.memory is None or self.incognito or not self._mcfg.get("save_conversations", True):
            return
        try:
            try:
                args = json.loads(call.arguments) if call.arguments else {}
            except json.JSONDecodeError:
                args = {"_raw": call.arguments[:200]}
            status = "denied" if result.denied else ("ok" if result.ok else "error")
            self.memory.log_tool(call.name, args if isinstance(args, dict) else {"_value": args},
                                 result.text[:500], status, self.conversation_id)
        except Exception:  # noqa: BLE001
            self.memory_errors += 1
            log.exception("no se pudo registrar el uso de la herramienta")

    async def respond(
        self,
        user_text: str,
        tools: list[dict[str, Any]] | None = None,
        **params: Any,
    ) -> AsyncIterator[AgentEvent]:
        """Procesa un mensaje del usuario y emite eventos en streaming.

        Si el consumidor interrumpe (aclose / cancelacion), el texto ya generado se conserva
        en el historial y en la memoria como respuesta parcial: es lo que el usuario alcanzo a recibir.
        Si falla sin haber generado texto, el mensaje del usuario se retira del historial para
        poder reintentar sin duplicarlo, y no se guarda nada.
        """
        registry = self.tools
        if registry is not None and tools is None and registry.names():
            tools = registry.specs()
        self.history.append({"role": "user", "content": user_text})
        parts: list[str] = []
        work: list[dict[str, Any]] = []  # mensajes de herramientas de ESTE turno (no van al historial)
        completed = False
        try:
            if registry is not None:
                registry.new_turn()
            for round_no in range(self.max_tool_rounds + 1):
                # En la ultima ronda no se ofrecen herramientas: obliga a responder con texto.
                offered = tools if (tools and round_no < self.max_tool_rounds) else None
                calls: list[ToolCall] = []
                round_text: list[str] = []
                if parts and not parts[-1].endswith((" ", "\n")):
                    parts.append(" ")
                async for chunk in self.router.stream_chat(self._messages(user_text) + work, tools=offered, **params):
                    if chunk.reasoning:
                        yield AgentEvent("reasoning", text=chunk.reasoning)
                    if chunk.text:
                        parts.append(chunk.text)
                        round_text.append(chunk.text)
                        yield AgentEvent("text", text=chunk.text)
                    if chunk.tool_calls:
                        calls.extend(chunk.tool_calls)
                if not calls:
                    break
                if registry is None:  # sin ejecutor: se informa y termina el turno
                    yield AgentEvent("tool_calls", tool_calls=calls)
                    break
                if offered is None:  # ultima ronda: no se ofrecieron herramientas, asi que no se ejecuta nada
                    break
                work.append({
                    "role": "assistant",
                    "content": "".join(round_text) or None,
                    "tool_calls": [{"id": c.id, "type": "function",
                                    "function": {"name": c.name, "arguments": c.arguments or "{}"}} for c in calls],
                })
                for c in calls:
                    yield AgentEvent("tool_start", text=c.name, tool_calls=[c])
                    result = await registry.execute(c)
                    self._log_tool(c, result)
                    yield AgentEvent("tool_result", text=c.name, result=result)
                    work.append({"role": "tool", "tool_call_id": c.id, "content": result.text})
            completed = True
        except Exception:
            if not parts:
                self.history.pop()
            raise
        finally:
            answer = "".join(parts)
            if answer.strip():
                self.history.append({"role": "assistant", "content": answer})
                self._persist(user_text, answer, partial=not completed)
        yield AgentEvent("done")
