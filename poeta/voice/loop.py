"""Bucle de conversacion por voz: microfono -> STT -> agente (LLM) -> TTS.

Modos de activacion:
- "ptt": Enter para hablar (o escribir un mensaje). Es el modo por defecto: el microfono solo
  se abre cuando tu lo pides.
- "always": escucha continuamente (opt-in; todo lo que se oiga se transcribe, localmente, hasta
  que exista la palabra de activacion).

En ambos modos, pulsar Enter mientras POETA habla (o genera) la interrumpe. El audio nunca se
guarda; solo viaja a la nube el texto que el router autorice, como en el chat de texto.
"""

from __future__ import annotations

import asyncio
import re
import threading
import time
from typing import Any, Awaitable, Callable

from .audio import to_float32

# Frases que Whisper "alucina" con ruido o silencio.
_HALLUCINATIONS = (
    "subtitulos realizados por", "subtítulos realizados por", "amara.org", "gracias por ver",
    "suscribete", "suscríbete", "no olvides suscribirte", "gracias por su atencion",
)


def is_noise_transcript(text: str) -> bool:
    t = text.strip().lower()
    if len(re.sub(r"\W", "", t)) < 2:
        return True
    return any(h in t for h in _HALLUCINATIONS)


class LineQueue:
    """Cola asincrona de lineas de teclado (llenada desde un hilo que lee stdin)."""

    def __init__(self) -> None:
        self._q: asyncio.Queue[str] = asyncio.Queue()
        self._front: list[str] = []  # lineas devueltas con unget(): salen primero
        self._loop = asyncio.get_running_loop()
        self._awaiting: asyncio.Future[str] | None = None  # una confirmacion esta esperando respuesta
        self.ask_wait_s = 0.0  # tiempo total que el usuario tardo en responder confirmaciones

    def put(self, line: str) -> None:  # thread-safe
        self._loop.call_soon_threadsafe(self._deliver, line)

    def _deliver(self, line: str) -> None:
        # Mientras hay una confirmacion pendiente, la linea es SU respuesta: ni interrumpe ni se encola.
        if self._awaiting is not None and not self._awaiting.done():
            self._awaiting.set_result(line)
        else:
            self._q.put_nowait(line)

    async def ask(self, prompt: str) -> str:
        """Muestra `prompt` y devuelve la siguiente linea escrita (para confirmaciones de permisos).
        Debe llamarse desde el bucle de eventos."""
        fut: asyncio.Future[str] = self._loop.create_future()
        self._awaiting = fut
        print(prompt, end="", flush=True)
        t0 = time.perf_counter()
        try:
            return await fut
        finally:
            self._awaiting = None
            self.ask_wait_s += time.perf_counter() - t0

    def unget(self, line: str) -> None:
        self._front.insert(0, line)

    async def get(self) -> str:
        if self._front:
            return self._front.pop(0)
        return await self._q.get()

    def get_nowait(self) -> str | None:
        if self._front:
            return self._front.pop(0)
        try:
            return self._q.get_nowait()
        except asyncio.QueueEmpty:
            return None

    def start_stdin_reader(self, stream: Any) -> threading.Thread:
        def run() -> None:
            for line in stream:
                self.put(line.rstrip("\n"))
            self.put("/quit")  # EOF

        t = threading.Thread(target=run, daemon=True, name="stdin")
        t.start()
        return t


class VoiceLoop:
    def __init__(self, *, audio: Any, stt: Any, pipeline: Any, agent: Any, keys: LineQueue,
                 mode: str = "ptt", wake: Any = None, followup_s: float = 8.0, chime: Any = None,
                 on_command: Callable[[str], Awaitable[bool]] | None = None,
                 out: Callable[..., None] = print, listen_timeout: float = 15.0,
                 bus: Any = None, metrics: Any = None, sampler: Callable[[], dict[str, Any]] | None = None,
                 meta: dict[str, Any] | None = None):
        if mode not in ("ptt", "always", "wake"):
            raise ValueError(f"modo invalido: {mode!r}")
        if mode == "wake" and wake is None:
            raise ValueError("el modo wake necesita un detector (wake=...)")
        self.wake, self.followup_s, self.chime = wake, followup_s, chime
        self.audio, self.stt, self.pipe, self.agent, self.keys = audio, stt, pipeline, agent, keys
        self.mode, self.on_command, self.out, self.listen_timeout = mode, on_command, out, listen_timeout
        self.turns = 0
        self.bus, self.metrics, self.sampler = bus, metrics, sampler
        self.meta = dict(meta or {})  # etiquetas fijas de cada fila: proveedor, modelo, STT/dispositivo...
        self._stop_evt = asyncio.Event()  # gesto de palma u otra orden externa de callar a POETA
        self._responding = False

    def _state(self, state: str) -> None:
        if self.bus is not None:
            self.bus.set_state(state)

    def interrupt(self) -> bool:
        """Calla a POETA si esta respondiendo (lo usa el gesto de palma). Devuelve si hizo algo."""
        if self._responding:
            self._stop_evt.set()
            return True
        return False

    # -- entrada ------------------------------------------------------------
    async def run(self) -> int:
        if self.mode == "ptt":
            self.out("Enter = hablar | escribe para enviar texto | /help | /quit")
        elif self.mode == "wake":
            self.out("Di 'poeta' seguido de tu orden. Lo que no empiece con la palabra de activacion se descarta. "
                     "Enter interrumpe. /quit para salir.")
        else:
            self.out("Escuchando todo el tiempo (sin palabra de activacion). Enter interrumpe a POETA. /quit para salir.")
        self._state("idle")
        while True:
            try:
                if await self._step():
                    self._state("stopped")
                    return 0
            except (KeyboardInterrupt, asyncio.CancelledError):
                self.audio.cancel_listen()
                self.pipe.stop()
                self._state("stopped")
                return 0

    async def _handle_line(self, line: str) -> bool | None:
        """True = salir; False = manejado; None = linea vacia (Enter)."""
        line = line.strip()
        if not line:
            return None
        if line.startswith("/") and self.on_command:
            return bool(await self.on_command(line))
        await self._respond(line)
        return False

    async def _step(self) -> bool:
        if self.mode == "ptt":
            line = await self.keys.get()
            handled = await self._handle_line(line)
            if handled is not None:
                return handled
            self.out("Escuchando...")
            return await self._listen_once(self.listen_timeout)
        pending = self.keys.get_nowait()
        if pending is not None:
            handled = await self._handle_line(pending)
            if handled:
                return True
            if handled is None:  # Enter o atajo global "hablar": escucha ahora, sin palabra de activacion
                return await self._listen_once(self.listen_timeout)
        if self.mode == "wake":
            return await self._wake_once()
        return await self._listen_once(0.5, quiet=True)

    # -- palabra de activacion ------------------------------------------------
    async def _transcribe(self, utt: Any) -> tuple[str | None, float]:
        t0 = time.perf_counter()
        try:
            res = await asyncio.to_thread(self.stt.transcribe, to_float32(utt))
        except Exception as e:  # noqa: BLE001
            self.out(f"Error de STT: {e}")
            return None, 0.0
        return res.text.strip(), time.perf_counter() - t0

    async def _wake_once(self) -> bool:
        utt = await asyncio.to_thread(self.audio.listen, 0.5)
        if utt is None:
            return False
        try:
            rest = await asyncio.to_thread(self.wake.detect, to_float32(utt))
        except Exception as e:  # noqa: BLE001
            self.out(f"Error de la palabra de activacion: {e}")
            return False
        if rest is None:
            return False  # no era para POETA: se descarta sin transcribir con el modelo principal
        text, stt_s = await self._transcribe(utt)  # la orden, con el modelo principal (mas preciso)
        if text is None:
            return False
        after = self.wake.strip(text)
        command = after if after is not None else rest
        if len(re.sub(r"\W", "", command)) >= 2 and not is_noise_transcript(command):
            self.out(f"Tu: {command}")
            await self._respond(command, stt_s=stt_s)
            return False
        return await self._followup()

    async def _followup(self) -> bool:
        """Solo dijo 'poeta': tono de aviso y espera la orden en el siguiente enunciado."""
        if self.chime is not None:
            await asyncio.to_thread(self.audio.speak, self.chime[0], self.chime[1])
        self.out("(te escucho)")
        self.audio.flush_input()
        self._state("listening")
        utt = await asyncio.to_thread(self.audio.listen, self.followup_s)
        if utt is None:
            self.out("(no te escuche)")
            self._state("idle")
            return False
        self._state("transcribing")
        text, stt_s = await self._transcribe(utt)
        if not text or is_noise_transcript(text):
            self.out(f"(ignorado: {text!r})")
            return False
        self.out(f"Tu: {text}")
        await self._respond(text, stt_s=stt_s)
        return False

    async def _listen_once(self, timeout: float, quiet: bool = False) -> bool:
        if not quiet:
            self.audio.flush_input()  # descarta el clic de la tecla y lo capturado antes
            self._state("listening")
        utt = await asyncio.to_thread(self.audio.listen, timeout)
        if utt is None:
            if not quiet:
                self.out("(no te escuche)")
                self._state("idle")
            return False
        self._state("transcribing")
        t0 = time.perf_counter()
        try:
            res = await asyncio.to_thread(self.stt.transcribe, to_float32(utt))
        except Exception as e:  # noqa: BLE001
            self.out(f"Error de STT: {e}")
            self._state("idle")
            return False
        stt_s = time.perf_counter() - t0
        text = res.text.strip()
        if is_noise_transcript(text):
            if not quiet:
                self.out(f"(ignorado: {text!r})")
            self._state("idle")
            return False
        self.out(f"Tu: {text}")
        await self._respond(text, stt_s=stt_s)
        return False

    def _requeue(self, key: "asyncio.Task[str]") -> None:
        """Enter solo interrumpe; si la linea trae texto o un comando, se procesa despues."""
        line = key.result()
        if line.strip():
            self.keys.unget(line)

    # -- respuesta ----------------------------------------------------------
    async def _respond(self, text: str, stt_s: float | None = None) -> None:
        self.turns += 1
        t0 = time.perf_counter()
        w0 = getattr(self.keys, "ask_wait_s", 0.0)
        first = {"text": None, "reason": None}
        tools_used = 0
        self._stop_evt.clear()
        self._responding = True
        self._state("thinking")

        def since_start() -> float:  # no cuenta lo que tardaste en autorizar
            return time.perf_counter() - t0 - (getattr(self.keys, "ask_wait_s", 0.0) - w0)

        self.pipe.begin()
        self.out("POETA: ", end="", flush=True)

        async def consume() -> None:
            nonlocal tools_used
            async for ev in self.agent.respond(text):
                if ev.kind == "reasoning" and first["reason"] is None:
                    first["reason"] = since_start()
                elif ev.kind == "text":
                    if first["text"] is None:
                        first["text"] = since_start()
                    self._state("speaking")
                    self.out(ev.text, end="", flush=True)
                    self.pipe.feed(ev.text)
                elif ev.kind == "tool_start" and ev.tool_calls:
                    tools_used += len(ev.tool_calls)
                    self._state("tool")
                    self.out(f"\n[herramienta: {', '.join(t.name for t in ev.tool_calls)}]\nPOETA: ", end="", flush=True)
                elif ev.kind == "tool_result" and ev.result is not None:
                    r = ev.result
                    self._state("thinking")
                    self.out(f"[{r.name}: {'denegada' if r.denied else ('ok' if r.ok else 'error')}]\nPOETA: ",
                             end="", flush=True)
            self.pipe.end()

        task = asyncio.create_task(consume())
        key = asyncio.create_task(self.keys.get())
        stop = asyncio.create_task(self._stop_evt.wait())
        interrupted = False
        try:
            done, _ = await asyncio.wait({task, key, stop}, return_when=asyncio.FIRST_COMPLETED)
            if key in done:
                interrupted = True
                self._requeue(key)
            elif stop in done and task not in done:
                interrupted = True
            else:
                err = task.exception()
                if err is not None:
                    self.out(f"\nError: {err}")
                    self.pipe.stop()
                    key.cancel()
                    stop.cancel()
                    self._state("idle")
                    return
                self._state("speaking")
                speech = asyncio.create_task(asyncio.to_thread(self.pipe.wait))
                done, _ = await asyncio.wait({speech, key, stop}, return_when=asyncio.FIRST_COMPLETED)
                if key in done:
                    interrupted = True
                    self._requeue(key)
                elif stop in done and speech not in done:
                    interrupted = True
                else:
                    key.cancel()
        except asyncio.CancelledError:
            interrupted = True
            raise
        finally:
            self._responding = False
            stop.cancel()
            if interrupted:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)  # el agente guarda el parcial
                self.pipe.stop()
                self.out("\n[interrumpido]")
        self._state("idle")
        parts = []
        if stt_s is not None:
            parts.append(f"stt {stt_s:.2f}s")
        waited = getattr(self.keys, "ask_wait_s", 0.0) - w0
        if waited > 0:
            parts.append(f"autorizacion {waited:.1f}s (excluida)")
        if first["reason"] is not None:
            parts.append(f"razonando desde {first['reason']:.2f}s")
        if first["text"] is not None:
            parts.append(f"1er texto {first['text']:.2f}s")
        if self.pipe.first_audio_s is not None:
            parts.append(f"1er audio {max(0.0, self.pipe.first_audio_s - waited):.2f}s")
        self.out("\n[" + " | ".join(parts) + "]\n" if parts else "\n")
        self._record_turn(stt_s, first, waited, interrupted, tools_used, time.perf_counter() - t0)

    def _record_turn(self, stt_s: float | None, first: dict[str, Any], waited: float, interrupted: bool,
                     tools: int, total: float) -> None:
        """Una fila de metricas por turno (solo numeros). Nunca debe afectar la conversacion."""
        if self.metrics is None and self.bus is None:
            return
        row = {**self.meta, "mode": self.mode, "stt_s": stt_s, "reasoning_s": first["reason"], "first_text_s": first["text"],
               "first_audio_s": (max(0.0, self.pipe.first_audio_s - waited) if self.pipe.first_audio_s is not None
                                 else None),
               "auth_wait_s": round(waited, 2) if waited > 0 else None, "total_s": round(max(0.0, total - waited), 2),
               "interrupted": interrupted, "tools": tools,
               "think": getattr(self.agent, "last_reasoning", None)}
        row = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items() if v is not None}
        try:
            if self.sampler is not None:
                row.update(self.sampler())
            if self.metrics is not None:
                self.metrics.record(**row)
            if self.bus is not None:
                self.bus.publish("turn", **row)
        except Exception:  # noqa: BLE001
            pass
