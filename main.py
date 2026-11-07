"""Punto de entrada de P.O.E.T.A. (Fase 0: solo carga la configuracion y registra el arranque).

El Agent Core y el Model Router se implementan en la Fase 1.
"""

from __future__ import annotations

import sys

from poeta.config import PROJECT_ROOT, ConfigError, load_config
from poeta.logging_setup import setup_logging


def main() -> int:
    try:
        cfg = load_config()
    except ConfigError as exc:
        print(f"Error de configuracion: {exc}", file=sys.stderr)
        return 1

    log = setup_logging(cfg, PROJECT_ROOT)
    defaults = cfg["providers"]["defaults"]
    log.info(
        "P.O.E.T.A. iniciado (Fase 0)",
        extra={"provider": defaults["provider"], "cloud_fallback": defaults["cloud_fallback"]},
    )
    print("P.O.E.T.A. Fase 0: configuracion valida. Ejecuta scripts/doctor.py para el diagnostico.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
