#!/usr/bin/env python3
"""Benchmark de STT (faster-whisper) para decidir modelo y dispositivo.

Prueba combinaciones modelo/dispositivo/precision sobre tus propios audios y
mide: carga, latencia por audio, factor de tiempo real, WER (si hay texto de
referencia) y VRAM usada (nvidia-smi). Mide con el servidor del LLM corriendo
para ver el escenario real (el LLM ocupa ~4,5 GB de los 8 GB).

Audios: archivos .wav/.mp3/... Si junto a un audio hay un .txt con el mismo
nombre, se usa como referencia para calcular el WER.

Ejemplos:
  python scripts/stt_benchmark.py --audio-dir ~/AI/output/vozboard/audio \\
      --combo large-v3:cuda:float16 --combo large-v3-turbo:cuda:int8_float16 \\
      --combo large-v3-turbo:cpu:int8 --combo distil-large-v3:cpu:int8
"""

from __future__ import annotations

import argparse
import csv
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

AUDIO_EXT = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus"}
DEFAULT_COMBOS = ["large-v3:cuda:float16", "large-v3-turbo:cuda:int8_float16",
                  "large-v3-turbo:cpu:int8"]


def vram_used_mib() -> int | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True).stdout
        return int(out.strip().splitlines()[0])
    except Exception:
        return None


def parse_combo(text: str) -> tuple[str, str, str]:
    parts = text.split(":")
    if len(parts) != 3 or not all(parts):
        raise argparse.ArgumentTypeError(f"formato invalido {text!r}; usar modelo:dispositivo:precision")
    return parts[0], parts[1], parts[2]


def collect_audio(audio_dir: Path, limit: int) -> list[Path]:
    files = sorted(p for p in audio_dir.iterdir() if p.suffix.lower() in AUDIO_EXT)
    return files[-limit:] if limit else files


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio-dir", default="~/AI/output/vozboard/audio")
    ap.add_argument("--limit", type=int, default=8, help="usar los N audios mas recientes (0 = todos)")
    ap.add_argument("--combo", action="append", type=parse_combo,
                    help="modelo:dispositivo:precision (repetible)")
    ap.add_argument("--language", default="es")
    ap.add_argument("--download-root", default=None)
    ap.add_argument("--out", default="~/AI/output/stt_benchmark")
    args = ap.parse_args()

    from poeta.voice.metrics import wer
    from poeta.voice.stt import SttConfig, SttError, Transcriber

    audio_dir = Path(args.audio_dir).expanduser()
    if not audio_dir.is_dir():
        print(f"No existe {audio_dir}. Usa --audio-dir con una carpeta de grabaciones.")
        return 2
    files = collect_audio(audio_dir, args.limit)
    if not files:
        print(f"No hay audios en {audio_dir}.")
        return 2
    combos = args.combo or [parse_combo(c) for c in DEFAULT_COMBOS]
    root = args.download_root
    if root is None:
        import os
        models = os.environ.get("AI_MODELS") or str(Path("~/AI/models").expanduser())
        root = str(Path(models, "whisper").expanduser())

    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, summary = [], []
    for model, device, ctype in combos:
        label = f"{model}|{device}|{ctype}"
        print(f"\n=== {label} ===")
        base = vram_used_mib()
        cfg = SttConfig(model=model, device=device, compute_type=ctype,
                        language=args.language or None, download_root=root)
        tr = Transcriber(cfg)
        try:
            tr.load()
            tr.transcribe(str(files[0]))  # calentamiento
        except SttError as e:
            print(f"  FALLO: {e}")
            summary.append((label, "fallo", "", "", "", "", ""))
            continue
        after = vram_used_mib()
        vram_delta = (after - base) if (base is not None and after is not None) else None
        lat, rtf, wers = [], [], []
        for f in files:
            r = tr.transcribe(str(f))
            lat.append(r.elapsed)
            if r.duration:
                rtf.append(r.elapsed / r.duration)
            ref = f.with_suffix(".txt")
            w = wer(ref.read_text(encoding="utf-8"), r.text) if ref.exists() else None
            if w is not None:
                wers.append(w)
            rows.append({"combo": label, "audio": f.name, "audio_s": round(r.duration, 2),
                         "latencia_s": round(r.elapsed, 3),
                         "wer": "" if w is None else round(w, 3), "texto": r.text})
            print(f"  {f.name[:32]:32s} {r.duration:5.1f}s -> {r.elapsed:5.2f}s"
                  + ("" if w is None else f"  WER={w:.2f}") + f"  {r.text[:60]}")
        summary.append((label, f"{tr.load_seconds:.1f}", f"{statistics.mean(lat):.2f}",
                        f"{statistics.mean(rtf):.3f}" if rtf else "",
                        f"{statistics.mean(wers):.3f}" if wers else "n/d",
                        "" if vram_delta is None else str(vram_delta), str(len(files))))
        del tr

    csv_path = out_dir / "stt_resultados.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print("\n=== Resumen ===")
    print(f"{'combo':44s} {'carga_s':>8s} {'lat_s':>7s} {'RTF':>7s} {'WER':>7s} {'VRAM_MiB':>9s}")
    for label, load, lat, rtf, werm, vram, _ in summary:
        print(f"{label:44s} {load:>8s} {lat:>7s} {rtf:>7s} {werm:>7s} {vram:>9s}")
    print("\nVRAM = aumento medido tras cargar (aprox.). Con el LLM activo, el total debe "
          "dejar margen bajo 8192 MiB.")
    print(f"Detalle: {csv_path}" if rows else "Sin resultados: todas las combinaciones fallaron.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
