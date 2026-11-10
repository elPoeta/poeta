"""Confirmar permisos con la voz (opcional, apagado por defecto).

Politica (el micro no distingue quien habla; por eso es mas estricta que el teclado):
- Solo acciones de confirmacion simple (nivel 3). Nunca cloud_transfer; las explicitas (nivel 4) y las que siguen a
  datos no confiables solo si lo habilitas a proposito (`allow_explicit` con una frase fija, `allow_tainted`).
- POETA dice que accion pide permiso, espera a terminar de hablar y escucha. Solo vale un enunciado que sea EXACTAMENTE
  una palabra de la lista (si, dale, autorizo / no, cancelar...). Frases largas o ambiguas no cuentan.
- Sin respuesta valida en `timeout_s` no decide nada: el teclado o los gestos siguen disponibles; en modo servicio vence y se deniega.
- Se audita que hubo una confirmacion por voz (accion y resultado), nunca la grabacion ni el texto.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
import unicodedata
from dataclasses import dataclass
from typing import Any, Callable

log = logging.getLogger("poeta.voice")

NEVER_BY_VOICE = {"cloud_transfer"}
YES_WORDS = {"si", "dale", "autorizo", "autorizado", "confirmo", "confirmado", "ok", "okey", "adelante", "hacelo",
             "hazlo", "correcto", "de acuerdo", "si autorizo", "si dale", "si por favor", "si hacelo", "si hazlo"}
NO_WORDS = {"no", "cancelar", "cancela", "cancelo", "cancelado", "denegar", "deniego", "para", "detente", "negativo",
            "no gracias", "no autorizo", "mejor no", "no hacelo", "no lo hagas"}


def normalize(text: str) -> str:
    t = unicodedata.normalize("NFKD", text.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", t)).strip()


def classify(text: str, explicit_phrase: str | None = None) -> bool | None:
    """True = autorizar, False = cancelar, None = no es una respuesta valida. Solo cuenta el enunciado completo."""
    t = normalize(text)
    if not t:
        return None
    if t in {normalize(w) for w in NO_WORDS}:
        return False
    if explicit_phrase is not None:
        return True if t == normalize(explicit_phrase) else None
    return True if t in {normalize(w) for w in YES_WORDS} else None


@dataclass(frozen=True)
class VoiceConfirmConfig:
    enabled: bool = False
    allow_explicit: bool = False
    explicit_phrase: str = "confirmo la accion"
    allow_tainted: bool = False
    timeout_s: float = 20.0
    retries: int = 2
    service_timeout_s: float = 60.0  # sin teclado: cuanto esperar antes de denegar

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "VoiceConfirmConfig":
        raw = (cfg.get("voice") or {}).get("confirm") or {}
        return cls(**{k: raw[k] for k in cls.__dataclass_fields__ if k in raw})


class VoiceConfirmer:
    """Pregunta en voz alta y escucha la respuesta. Se enlaza al audio cuando la voz ya esta lista (`attach`)."""

    def __init__(self, config: VoiceConfirmConfig, out: Callable[..., None] = print, audit: Any = None):
        self.config, self.out, self.audit = config, out, audit
        self.audio = self.stt = self.pipe = None

    def attach(self, audio: Any, stt: Any, pipe: Any) -> None:
        self.audio, self.stt, self.pipe = audio, stt, pipe

    @property
    def active(self) -> bool:
        return self.config.enabled and self.pipe is not None

    def eligible(self, req: Any, explicit: bool) -> bool:
        c = self.config
        if not self.active or getattr(req, "kind", "") in NEVER_BY_VOICE:
            return False
        if explicit and not c.allow_explicit:
            return False
        return c.allow_tainted or not getattr(req, "tainted", False)

    # -- audio (bloqueante: se llama desde un hilo) ---------------------------------------------------------
    def _say(self, text: str) -> None:
        samples = self.pipe.engine.synthesize(text)
        self.audio.speak(samples, self.pipe.engine.sample_rate)

    def _listen(self, timeout: float) -> Any:
        with self.pipe.open_mic():
            self.audio.flush_input()
            return self.audio.listen(timeout)

    def prompt_for(self, req: Any, explicit: bool) -> str:
        what = " ".join(str(req.description).split())[:160]
        if explicit:
            return f"Accion sensible: {what}. Para autorizar, di exactamente: {self.config.explicit_phrase}. Para cancelar, di no."
        return f"Necesito permiso para {what}. ¿Autorizas? Di si o no."

    async def ask(self, req: Any, explicit: bool) -> bool | None:
        """True/False si el usuario respondio con claridad; None si no hubo respuesta valida a tiempo."""
        from poeta.voice.audio import to_float32

        c = self.config
        deadline = time.monotonic() + c.timeout_s
        phrase = c.explicit_phrase if explicit else None
        attempts = 0
        try:
            await asyncio.to_thread(self._say, self.prompt_for(req, explicit))
            while attempts <= c.retries and time.monotonic() < deadline:
                utt = await asyncio.to_thread(self._listen, max(0.5, min(8.0, deadline - time.monotonic())))
                if utt is None:
                    continue
                attempts += 1
                try:
                    heard = (await asyncio.to_thread(self.stt.transcribe, to_float32(utt))).text
                except Exception as exc:  # noqa: BLE001
                    log.warning("STT de confirmacion fallo: %s", exc)
                    continue
                verdict = classify(heard, phrase)
                if verdict is not None:
                    self._audit(req, verdict, attempts)
                    self.out(f"\n(decidido por voz: {'autorizado' if verdict else 'cancelado'})")
                    return verdict
                if attempts <= c.retries:
                    await asyncio.to_thread(self._say, "No te entendi. Di " + ("la frase exacta o no." if explicit else "si o no."))
            return None
        except asyncio.CancelledError:
            raise
        finally:
            if self.audio is not None:
                self.audio.cancel_listen()  # si otro canal decidio, el hilo de escucha vuelve de inmediato

    def _audit(self, req: Any, allowed: bool, attempts: int) -> None:
        if self.audit is None:
            return
        try:
            self.audit.record("voice_confirmation", kind=req.kind, tool=(req.details or {}).get("tool"),
                              allowed=allowed, attempts=attempts)
        except Exception:  # noqa: BLE001
            log.exception("no se pudo auditar la confirmacion por voz")


def make_voice_confirmer(base: Callable[..., Any], vc: VoiceConfirmer, bus: Any = None) -> Callable[..., Any]:
    """Gana la primera respuesta valida entre la voz y el confirmador de abajo (teclado y/o gestos)."""

    async def confirm(req: Any, explicit: bool) -> bool:
        if not vc.eligible(req, explicit):
            return await base(req, explicit)
        if bus is not None:
            bus.set_state("confirming")
        other = asyncio.ensure_future(base(req, explicit))
        voice = asyncio.ensure_future(vc.ask(req, explicit))
        try:
            pending = {other, voice}
            while pending:
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                if other in done:
                    return bool(other.result())
                if voice in done:
                    verdict = voice.result()
                    if verdict is not None:
                        return verdict
                    # la voz se rindio: sigue esperando al teclado/gestos (o al vencimiento del modo servicio)
            return False
        finally:
            for t in (other, voice):
                if not t.done():
                    t.cancel()
            await asyncio.gather(other, voice, return_exceptions=True)

    return confirm


def make_headless_base(timeout_s: float, out: Callable[..., None] = print) -> Callable[..., Any]:
    """Base del modo servicio: no hay teclado, asi que espera a gestos/voz y deniega al vencer el plazo."""

    async def confirm(req: Any, explicit: bool) -> bool:
        await asyncio.sleep(timeout_s)
        out(f"\n(sin respuesta en {timeout_s:.0f} s: se deniega '{req.description}')")
        return False

    return confirm
