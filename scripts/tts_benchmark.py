#!/usr/bin/env python3
"""Benchmark A/B de motores TTS para P.O.E.T.A. (plan v1.1, sección 8.5).

Sintetiza las mismas frases con cada motor/voz candidato, guarda los WAV y
mide latencia, tiempo total, factor de tiempo real (RTF) y memoria.

Ejemplos:
  python tts_benchmark.py kokoro --model kokoro.onnx --voices voices.bin --voice em_alex
  python tts_benchmark.py piper --model ~/AI/models/piper/es_MX-xxx.onnx

Notas:
- Los nombres de voces y la API de kokoro-onnx deben verificarse contra la versión instalada.
- Piper se invoca por línea de comandos (binario `piper`), por lo que no depende de su API Python.
- "Latencia al primer audio" aquí es el tiempo hasta tener la frase completa sintetizada,
  porque ambos motores se usan sin streaming en este script. Con streaming real
  (Fase 4) el primer audio llegará antes; este número es una cota superior.
"""

from __future__ import annotations

import argparse
import csv
import resource
import subprocess
import sys
import time
import wave
from pathlib import Path

import numpy as np

# Frases de uso real: cortas, números, fechas, nombres propios, anglicismos, largas.
PHRASES = [
    "Hola, ¿en qué puedo ayudarte?",
    "Listo, abrí el navegador.",
    "Son las cinco y cuarenta y siete de la tarde.",
    "Hoy es viernes, seis de noviembre de dos mil veintiséis.",
    "Tienes tres mensajes nuevos y una reunión a las nueve y media.",
    "La temperatura en Buenos Aires es de veintitrés grados.",
    "El archivo pesa 2,5 gigabytes y tardará unos 15 minutos en descargarse.",
    "Voy a iniciar llama.cpp con el modelo Nemotron en la GPU.",
    "Instalé las dependencias con uv y creé el entorno virtual.",
    "El servidor MCP de Playwright está conectado y listo.",
    "Wayland no permite controlar ventanas como lo hacía X11.",
    "No encontré ese archivo. ¿Quieres que lo busque en otra carpeta?",
    "Necesito tu confirmación antes de enviar esta información a un proveedor externo.",
    "Leonardo, ¿quieres que elimine los doce archivos temporales de la carpeta de descargas?",
    "Según los resultados de la búsqueda, la nueva versión incluye mejoras de rendimiento, "
    "correcciones de seguridad y soporte para más idiomas, aunque todavía está en fase de pruebas.",
    "Perfecto. Cuando termine, te aviso.",
    "Hay un error de conexión: el puerto ocho mil ochenta no responde.",
    "La reunión con el equipo de Córdoba fue reprogramada para el martes a las diez.",
]


def write_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    samples = np.asarray(samples)
    if samples.dtype != np.int16:
        samples = np.clip(samples, -1.0, 1.0)
        samples = (samples * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(samples.tobytes())


def read_wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / float(w.getframerate())


def rss_mb() -> float:
    # En Linux ru_maxrss está en KiB.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


class KokoroRunner:
    def __init__(self, model: str, voices: str, voice: str, speed: float, lang: str):
        from kokoro_onnx import Kokoro  # import perezoso

        self.engine = Kokoro(model, voices)
        self.voice = voice
        self.speed = speed
        self.lang = lang

    def synth(self, text: str, out: Path) -> float:
        """Sintetiza a WAV y devuelve la duración del audio en segundos."""
        samples, sr = self.engine.create(
            text, voice=self.voice, speed=self.speed, lang=self.lang
        )
        write_wav(out, samples, sr)
        return len(samples) / float(sr)


class PiperRunner:
    def __init__(self, model: str, binary: str, length_scale: float):
        self.model = model
        self.binary = binary
        self.length_scale = length_scale

    def synth(self, text: str, out: Path) -> float:
        cmd = [
            self.binary,
            "--model", self.model,
            "--output_file", str(out),
            "--length_scale", str(self.length_scale),
        ]
        subprocess.run(cmd, input=text.encode("utf-8"), check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return read_wav_duration(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="engine", required=True)

    k = sub.add_parser("kokoro", help="Kokoro-82M vía kokoro-onnx (CPU)")
    k.add_argument("--model", required=True, help="Ruta al modelo .onnx")
    k.add_argument("--voices", required=True, help="Ruta al archivo de voces")
    k.add_argument("--voice", required=True, help="Nombre de la voz (verificar disponibles)")
    k.add_argument("--speed", type=float, default=1.0)
    k.add_argument("--lang", default="es")

    p = sub.add_parser("piper", help="Piper vía línea de comandos")
    p.add_argument("--model", required=True, help="Ruta a la voz .onnx de Piper")
    p.add_argument("--binary", default="piper")
    p.add_argument("--length-scale", type=float, default=1.0)

    ap.add_argument("--out", default="~/AI/output/tts_benchmark",
                    help="Directorio de salida de los WAV y del CSV")
    ap.add_argument("--label", default=None,
                    help="Etiqueta para los archivos (por defecto motor+voz)")
    ap.add_argument("--phrases-file", default=None,
                    help="Archivo de texto con una frase por línea (reemplaza las incluidas)")

    args = ap.parse_args()

    phrases = PHRASES
    if args.phrases_file:
        lines = Path(args.phrases_file).expanduser().read_text(encoding="utf-8").splitlines()
        phrases = [ln.strip() for ln in lines if ln.strip()]

    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    t_load = time.perf_counter()
    if args.engine == "kokoro":
        runner = KokoroRunner(args.model, args.voices, args.voice, args.speed, args.lang)
        default_label = f"kokoro_{args.voice}"
    else:
        runner = PiperRunner(args.model, args.binary, args.length_scale)
        default_label = f"piper_{Path(args.model).stem}"
    load_s = time.perf_counter() - t_load
    label = args.label or default_label

    # Calentamiento: la primera síntesis suele ser más lenta y falsea las medias.
    warm = out_dir / f"{label}_warmup.wav"
    runner.synth("Prueba de calentamiento.", warm)
    warm.unlink(missing_ok=True)

    rows = []
    for i, text in enumerate(phrases, start=1):
        wav = out_dir / f"{label}_{i:02d}.wav"
        t0 = time.perf_counter()
        audio_s = runner.synth(text, wav)
        synth_s = time.perf_counter() - t0
        rtf = synth_s / audio_s if audio_s > 0 else float("nan")
        rows.append({
            "n": i, "texto": text, "audio_s": round(audio_s, 3),
            "sintesis_s": round(synth_s, 3), "rtf": round(rtf, 3),
        })
        print(f"[{i:02d}] sintesis={synth_s:5.2f}s audio={audio_s:5.2f}s RTF={rtf:5.2f}  {text[:50]}")

    csv_path = out_dir / f"{label}_resultados.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    synth_times = np.array([r["sintesis_s"] for r in rows])
    rtfs = np.array([r["rtf"] for r in rows])
    print("\n=== Resumen:", label, "===")
    print(f"Carga del motor:          {load_s:.2f} s")
    print(f"Síntesis media por frase: {synth_times.mean():.2f} s (p95 {np.percentile(synth_times, 95):.2f} s)")
    print(f"RTF medio:                {rtfs.mean():.3f} (menor que 1 = más rápido que tiempo real)")
    print(f"Memoria máxima (RSS):     {rss_mb():.0f} MB")
    print(f"WAV y CSV en:             {out_dir}")
    print("\nSiguiente paso: escuchar a ciegas y puntuar naturalidad, acento y pronunciación.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
