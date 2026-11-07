import json

from poeta.config import load_config
from poeta.logging_setup import setup_logging


def test_logging_writes_json_lines(tmp_path):
    cfg = load_config()
    cfg["settings"]["logging"]["dir"] = str(tmp_path / "logs")
    log = setup_logging(cfg, tmp_path)
    log.info("evento de prueba", extra={"provider": "local"})
    for h in log.handlers:
        h.flush()

    lines = (tmp_path / "logs" / "poeta.log").read_text(encoding="utf-8").strip().splitlines()
    record = json.loads(lines[-1])
    assert record["msg"] == "evento de prueba"
    assert record["provider"] == "local"
    assert record["level"] == "INFO"
