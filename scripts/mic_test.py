#!/usr/bin/env python3
"""Prueba en vivo del Audio Manager + STT con el microfono real.

  python scripts/mic_test.py devices          # lista dispositivos de audio
  python scripts/mic_test.py level            # medidor de nivel (calibrar ruido y volumen)
  python scripts/mic_test.py listen -n 5      # detecta enunciados y los transcribe
  python scripts/mic_test.py echo             # repite lo que dices por los parlantes y
                                              # verifica que POETA no se escuche a si misma

Nada se guarda a disco (el audio queda en memoria).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from poeta.config import load_config
from poeta.voice.audio import AudioConfig, AudioError, AudioManager, level_db, to_float32


def cmd_devices(_a, _cfg) -> int:
    import sounddevice as sd

    print(sd.query_devices())
    print(f"\nPredeterminados (entrada, salida): {sd.default.device}")
    return 0


def cmd_level(a, cfg) -> int:
    ac = AudioConfig.from_config(cfg)
    with AudioManager(ac) as am:
        print("Medidor (Ctrl+C para salir). Habla normal y luego calla para ver el ruido de fondo.")
        floor = -120.0
        t_end = time.monotonic() + a.seconds
        try:
            while time.monotonic() < t_end:
                try:
                    f, _ = am._q.get(timeout=0.5)
                except Exception:
                    continue
                db = level_db(f)
                floor = db if floor == -120.0 else min(floor + 0.05, db) if db < floor else floor + 0.002
                bar = "#" * max(0, int((db + 80) / 2))
                print(f"\r{db:6.1f} dBFS  ruido~{floor:6.1f}  {bar:<40s}", end="", flush=True)
        except KeyboardInterrupt:
            pass
    print()
    return 0


def _transcriber(cfg):
    from poeta.voice.stt import SttConfig, Transcriber

    tr = Transcriber(SttConfig.from_config(cfg))
    print(f"Cargando {tr.config.model} ({tr.config.device})...", flush=True)
    tr.load()
    print(f"Calentando ({tr.config.device}, {tr.config.effective_compute_type})...", flush=True)
    t0 = time.perf_counter()
    tr.warmup()
    print(f"Listo: carga {tr.load_seconds:.1f} s, calentamiento {time.perf_counter() - t0:.2f} s.")
    try:
        import ctranslate2
        print(f"  GPUs vistas por CTranslate2: {ctranslate2.get_cuda_device_count()}")
    except Exception as e:  # informativo
        print(f"  (no se pudo consultar CTranslate2: {e})")
    return tr


def cmd_listen(a, cfg) -> int:
    ac = AudioConfig.from_config(cfg)
    tr = None if a.no_stt else _transcriber(cfg)
    with AudioManager(ac) as am:
        print(f"Habla. Esperando {a.n} enunciado(s) (Ctrl+C para salir).")
        for i in range(a.n):
            t_wait = time.monotonic()
            utt = am.listen(timeout=a.timeout)
            if utt is None:
                print("  (sin voz: timeout)")
                continue
            t_cut = time.monotonic()
            dur = len(utt) / ac.sample_rate
            line = f"[{i + 1}] {dur:4.1f} s de audio, nivel {level_db(utt):5.1f} dBFS"
            if tr:
                r = tr.transcribe(to_float32(utt))
                line += f" | STT {r.elapsed:4.2f} s | {r.text!r}"
            print(line)
    return 0


def cmd_echo(a, cfg) -> int:
    """Repite lo dicho y cuenta cuantos enunciados 'fantasma' entran mientras suena."""
    ac = AudioConfig.from_config(cfg)
    with AudioManager(ac) as am:
        print("Di una frase; POETA la repetira. Hace 3 rondas. Si oye su propia voz, lo indicara.")
        for i in range(3):
            utt = am.listen(timeout=30)
            if utt is None:
                print("  sin voz")
                continue
            x = to_float32(utt)
            am.play(x, ac.sample_rate)
            am.wait()
            time.sleep(ac.tail_ms / 1000 + 0.3)
            ghost = am.listen(timeout=1.0)
            # lo capturado durante la reproduccion ya se descarto; aqui solo cuenta lo posterior
            print(f"  ronda {i + 1}: repeticion de {len(x) / ac.sample_rate:.1f} s -> "
                  + ("ECO DETECTADO tras la cola (sube tail_ms)" if ghost is not None else "sin eco, OK"))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("devices")
    p = sub.add_parser("level")
    p.add_argument("--seconds", type=float, default=30)
    p = sub.add_parser("listen")
    p.add_argument("-n", type=int, default=5)
    p.add_argument("--timeout", type=float, default=60)
    p.add_argument("--no-stt", action="store_true")
    sub.add_parser("echo")
    a = ap.parse_args()
    cfg = load_config()
    try:
        return {"devices": cmd_devices, "level": cmd_level, "listen": cmd_listen, "echo": cmd_echo}[a.cmd](a, cfg)
    except (AudioError, OSError, ImportError) as e:
        print(f"Error: {e}")
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
