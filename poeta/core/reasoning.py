"""Politica de razonamiento por turno (solo modelos LOCALES con llama.cpp).

Razonar mejora el uso de herramientas en modelos chicos pero retrasa ~3 s la primera palabra; para charlar no hace
falta. En modo `auto` se razona solo cuando el pedido parece una orden o consulta que necesita herramientas, o es largo.
Se controla por pedido con `chat_template_kwargs: {enable_thinking: bool}` (el servidor se deja con `--reasoning on`).
Los proveedores cloud no reciben nada: se manejan solos.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

MODES = ("auto", "on", "off", "server")  # server = no enviar nada: manda lo que tenga llama-server

DEFAULT_TRIGGERS = (
    r"\b(abr[eií]\w*|cerr[aáeé]\w*|busc[aáeé]\w*|investig\w*|googlea\w*|mostr[aáeé]\w*|list[aáeé]\w*|baj[aáeé]\w*|"
    r"sub[eií]\w*|silenci\w*|paus[aáeé]\w*|reproduc\w*|pon[eé]\w*|activ[aáeé]\w*|desactiv\w*|ejecut\w*|cre[aáeé]\w*|"
    r"escrib\w*|guard[aáeé]\w*|le[eé]\w*|borr[aáeé]\w*|elimin\w*|renombr\w*|mand[aáeé]\w*|envi[aáeé]\w*|"
    r"copi[aáeé]\w*|mov[eé]\w*|record[aáeé]\w*|agend[aáeé]\w*|notific\w*|minimiz\w*|reinici\w*|inici[aáeé]\w*|"
    r"detené\w*|deten\w*|cambi[aáeé]\w*|trae\w*|traé\w*|descarg\w*|calcul\w*|compar\w*)\b",
    r"\b(qu[eé] hora|qu[eé] d[ií]a|fecha|clima|noticias|internet|web|archivo\w*|carpeta\w*|ventana\w*|vol[uú]men\w*|"
    r"spotify|firefox|navegador\w*|rutina\w*|servicio\w*|proceso\w*|aplicaci[oó]n|aplicaciones|programa\w*|"
    r"canci[oó]n|canciones|m[uú]sica|memoria|recuerdas?)\b",
)


class ReasoningConfigError(ValueError):
    pass


@dataclass
class ReasoningPolicy:
    mode: str = "auto"
    long_words: int = 30
    triggers: tuple[re.Pattern[str], ...] = field(
        default_factory=lambda: tuple(re.compile(t, re.IGNORECASE) for t in DEFAULT_TRIGGERS))

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "ReasoningPolicy":
        r = ((cfg.get("settings") or {}).get("agent") or {}).get("reasoning") or {}
        if not isinstance(r, dict):
            raise ReasoningConfigError("agent.reasoning debe ser un mapa (mode, long_words, triggers)")
        mode = r.get("mode", "auto")
        if mode not in MODES:
            raise ReasoningConfigError(f"agent.reasoning.mode debe ser uno de {MODES}")
        words = r.get("long_words", 30)
        if isinstance(words, bool) or not isinstance(words, int) or words < 1:
            raise ReasoningConfigError("agent.reasoning.long_words debe ser un entero positivo")
        trig = r.get("triggers")
        if trig is None:
            return cls(mode, words)
        if not isinstance(trig, list) or not all(isinstance(t, str) for t in trig):
            raise ReasoningConfigError("agent.reasoning.triggers debe ser una lista de expresiones regulares")
        try:
            return cls(mode, words, tuple(re.compile(t, re.IGNORECASE) for t in trig))
        except re.error as exc:
            raise ReasoningConfigError(f"agent.reasoning.triggers: expresion invalida: {exc}") from exc

    def wants(self, text: str, tools_available: bool) -> bool:
        if self.mode == "on":
            return True
        if self.mode == "off":
            return False
        if tools_available and any(p.search(text) for p in self.triggers):
            return True
        return len(text.split()) >= self.long_words

    def params_for(self, text: str, tools_available: bool, is_cloud: bool) -> tuple[dict[str, Any], bool | None]:
        """(parametros extra del pedido, decision). Decision None = no se controla."""
        if is_cloud or self.mode == "server":
            return {}, None
        think = self.wants(text, tools_available)
        return {"chat_template_kwargs": {"enable_thinking": think}}, think
