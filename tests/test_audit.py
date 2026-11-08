import json
import os

import pytest

from poeta.security import AuditError, AuditLog, read_tail, redact, verify_chain


def lines(path):
    return path.read_text(encoding="utf-8").splitlines()


def test_chain_is_valid_and_resumes_after_reopen(tmp_path):
    path = tmp_path / "logs" / "audit.jsonl"
    log = AuditLog(path)
    log.record("a", x=1)
    log.record("b", x=2)
    assert verify_chain(path) == (True, 2, None)

    reopened = AuditLog(path)  # nuevo proceso: continua la cadena
    entry = reopened.record("c")
    assert entry["seq"] == 3
    assert verify_chain(path) == (True, 3, None)


def test_modified_entry_is_detected(tmp_path):
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path)
    for i in range(3):
        log.record("evento", i=i)
    rows = lines(path)
    tampered = json.loads(rows[1])
    tampered["i"] = 999
    rows[1] = json.dumps(tampered, sort_keys=True, ensure_ascii=False)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    ok, verified, err = verify_chain(path)
    assert not ok and verified == 1 and "modificado" in err


def test_deleted_entry_is_detected(tmp_path):
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path)
    for i in range(3):
        log.record("evento", i=i)
    rows = lines(path)
    path.write_text("\n".join([rows[0], rows[2]]) + "\n", encoding="utf-8")

    ok, _, err = verify_chain(path)
    assert not ok and "secuencia" in err


def test_corrupt_log_blocks_startup_instead_of_overwriting(tmp_path):
    path = tmp_path / "audit.jsonl"
    AuditLog(path).record("a")
    with path.open("a") as f:
        f.write('{"seq": 2, "event": "cortado')  # escritura interrumpida
    with pytest.raises(AuditError, match="danado"):
        AuditLog(path)


def test_file_is_private(tmp_path):
    path = tmp_path / "audit.jsonl"
    AuditLog(path).record("a")
    assert oct(os.stat(path).st_mode & 0o777) == "0o600"


def test_reserved_fields_are_rejected(tmp_path):
    with pytest.raises(ValueError):
        AuditLog(tmp_path / "a.jsonl").record("a", hash="falso")


def test_redaction():
    data = {
        "api_key": "abc",
        "Authorization": "Bearer abcdefghijkl",
        "nota": "clave nvapi-ABCDEFGH12345678 y sk-abcdefghijklmnopqrstu",
        "largo": "x" * 1000,
        "anidado": {"password": "p", "ok": 1},
        "first_token_s": 0.4,   # contiene "token" pero es una metrica, no un secreto
        "tokens": 123,
    }
    out = redact(data)
    assert out["api_key"] == "***" and out["Authorization"] == "***"
    assert "nvapi" not in out["nota"] and "sk-abc" not in out["nota"]
    assert len(out["largo"]) < 400 and "+700 caracteres" in out["largo"]
    assert out["anidado"] == {"password": "***", "ok": 1}
    assert out["first_token_s"] == 0.4 and out["tokens"] == 123


def test_secrets_never_reach_the_file(tmp_path):
    path = tmp_path / "audit.jsonl"
    AuditLog(path).record("x", detalle={"api_key": "SECRETO-1", "texto": "usa nvapi-ZZZZZZZZZZZZ"})
    raw = path.read_text(encoding="utf-8")
    assert "SECRETO-1" not in raw and "nvapi-ZZZZ" not in raw


def test_read_tail(tmp_path):
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path)
    for i in range(5):
        log.record("e", i=i)
    assert [e["i"] for e in read_tail(path, 2)] == [3, 4]
    assert read_tail(tmp_path / "no-existe.jsonl") == []


def test_unwritable_path_raises_audit_error(tmp_path):
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path)
    path.unlink() if path.exists() else None
    path.mkdir()  # un directorio donde deberia haber un archivo
    with pytest.raises(AuditError):
        log.record("a")
