#!/usr/bin/env python3
"""Prueba en vivo del habla por frases con la voz configurada en config/voice.yaml.

  python scripts/speak_test.py                 # texto de ejemplo, simula streaming del LLM
  python scripts/speak_test.py "Tu texto aqui"
  python scripts/speak_test.py --interrupt 1.5 # corta a los 1,5 s para probar la interrupcion
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poeta.config import load_config
from poeta.voice.audio import AudioConfig, AudioManager
from poeta.voice.tts import SpeechPipeline, TtsConfig, TtsError, make_engine

SAMPLE = ("Hola, soy POETA. Hoy es domingo y son las cinco y media de la tarde. "
          "Tienes tres mensajes nuevos: uno de Córdoba, otro del equipo y uno de soporte. "
          "Si quieres, los leo uno por uno. ¿Por dónde empiezo?")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("text", nargs="?", default=SAMPLE)
    ap.add_argument("--interrupt", type=float, default=None, help="cortar tras N segundos")
    ap.add_argument("--chunk", type=int, default=6, help="caracteres por trozo (simula tokens)")
    ap.add_argument("--chunk-delay", type=float, default=0.02, help="segundos entre trozos")
    a = ap.parse_args()
    cfg = load_config()
    try:
        t0 = time.perf_counter()
        engine = make_engine(TtsConfig.from_config(cfg))
        print(f"Motor cargado en {time.perf_counter() - t0:.2f} s ({engine.sample_rate} Hz)")
        engine.synthesize("Calentamiento.")
        audio = AudioManager(AudioConfig.from_config(cfg))
        audio.start()
    except Exception as e:
        print(f"Error: {e}")
        return 1
    pipe = SpeechPipeline(engine, audio, on_error=lambda e: print(f"  (error de sintesis: {e})"))
    try:
        t0 = time.perf_counter()
        pipe.begin()
        for i in range(0, len(a.text), a.chunk):
            pipe.feed(a.text[i:i + a.chunk])
            time.sleep(a.chunk_delay)
            if a.interrupt and time.perf_counter() - t0 > a.interrupt:
                pipe.stop()
                print("Interrumpido.")
                return 0
        pipe.end()
        if a.interrupt:
            if not pipe.wait(max(0.0, a.interrupt - (time.perf_counter() - t0))):
                pipe.stop()
                print("Interrumpido.")
                return 0
        else:
            pipe.wait()
        print(f"Primer audio a los {pipe.first_audio_s:.2f} s del primer texto; errores: {pipe.errors}")
    except KeyboardInterrupt:
        pipe.stop()
    finally:
        pipe.close()
        audio.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
