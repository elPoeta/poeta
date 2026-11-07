"""Registro de eventos en JSON lineas (un objeto JSON por linea) + consola."""

from __future__ import annotations

import json
import logging
import logging.handlers
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_STD_ATTRS = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _STD_ATTRS:
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def setup_logging(cfg: dict[str, Any], root: Path) -> logging.Logger:
    """Configura el logger 'poeta' segun settings.logging y devuelve el logger."""
    log_cfg = cfg["settings"].get("logging", {})
    log_dir = Path(log_cfg.get("dir", "logs"))
    if not log_dir.is_absolute():
        log_dir = root / log_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("poeta")
    logger.setLevel(getattr(logging, str(log_cfg.get("level", "INFO")).upper(), logging.INFO))
    logger.handlers.clear()
    logger.propagate = False

    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / "poeta.log",
        maxBytes=int(log_cfg.get("max_bytes", 5_242_880)),
        backupCount=int(log_cfg.get("backup_count", 5)),
        encoding="utf-8",
    )
    file_handler.setFormatter(JsonFormatter())
    logger.addHandler(file_handler)

    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(console)
    return logger
