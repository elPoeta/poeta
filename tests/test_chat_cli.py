"""Comandos del chat y subcomando `memory`, ejecutando main.py de verdad con la entrada simulada."""

import asyncio
import builtins
import json

import pytest

import main
from poeta.config import load_config
from poeta.memory import MemoryStore


@pytest.fixture
def cfg(tmp_path, fake_llm):
    cfg = load_config()
    cfg["providers"]["providers"]["local"]["base_url"] = fake_llm.url
    cfg["memory"]["db_path"] = str(tmp_path / "data" / "memory.db")
    cfg["permissions"]["audit"]["file"] = str(tmp_path / "audit.jsonl")
    cfg["settings"]["logging"]["dir"] = str(tmp_path / "logs")
    return cfg


def script(monkeypatch, *lines):
    it = iter(lines)

    def fake_input(prompt=""):
        try:
            value = next(it)
        except StopIteration:
            raise EOFError
        print(f"{prompt}{value}")
        return value

    monkeypatch.setattr(builtins, "input", fake_input)


def run_chat(cfg, monkeypatch, capsys, *lines, **kw):
    script(monkeypatch, *lines)
    rc = asyncio.run(main.chat(cfg, None, None, **kw))
    return rc, capsys.readouterr().out


def open_store(cfg):
    return MemoryStore(cfg["memory"]["db_path"])


def test_remember_memories_forget_flow(cfg, monkeypatch, capsys):
    rc, out = run_chat(cfg, monkeypatch, capsys,
        "/remember Ana cumple el 4 de marzo", "/memories", "/memories cumple", "/forget 1", "/memories", "/forget 1", "/quit")
    assert rc == 0
    assert "Guardado (#1)." in out and "#1 [semantic] Ana cumple el 4 de marzo" in out
    assert "Olvidado." in out and "No existe la nota #1." in out and "Sin resultados." in out
    s = open_store(cfg)
    assert s.list_memories() == []
    s.close()


def test_preferences_flow(cfg, monkeypatch, capsys):
    _, out = run_chat(cfg, monkeypatch, capsys,
        "/prefer", "/prefer Idioma=español rioplatense", "/prefer", "/prefer sinigual", "/prefer mala clave=x", "/unprefer idioma", "/unprefer idioma", "/quit")
    assert "Sin preferencias." in out and "Preferencia guardada." in out and "idioma = español rioplatense" in out
    assert "Uso: /prefer CLAVE=VALOR" in out and "Error: La clave debe usar" in out
    assert "Preferencia borrada." in out and "No existe esa preferencia." in out


def test_chat_saves_and_resume_command_restores(cfg, monkeypatch, capsys):
    _, out = run_chat(cfg, monkeypatch, capsys, "hola mundo", "/conversations", "/quit")
    assert "Hola, mundo!" in out and "#1" in out and "2 msgs" in out and "hola mundo" in out

    _, out = run_chat(cfg, monkeypatch, capsys, "/reset", "/resume 1", "/resume 99", "/resume abc", "/quit")
    assert "Retomada la conversacion #1 (2 mensajes cargados)." in out
    assert "No existe la conversacion #99." in out and "Uso: /resume ID" in out


def test_continue_flag_resumes_last_conversation(cfg, monkeypatch, capsys, fake_llm):
    run_chat(cfg, monkeypatch, capsys, "primera charla", "/quit")
    _, out = run_chat(cfg, monkeypatch, capsys, "seguimos", "/quit", resume_last=True)
    assert "Retomada la conversacion #1 (2 mensajes cargados)." in out
    sent = fake_llm.posts()[-1]["body"]["messages"]
    assert [m["content"] for m in sent[1:]] == ["primera charla", "Hola, mundo!", "seguimos"]
    s = open_store(cfg)
    assert len(s.list_conversations()) == 1
    s.close()


def test_continue_with_empty_memory_starts_fresh(cfg, monkeypatch, capsys):
    _, out = run_chat(cfg, monkeypatch, capsys, "/quit", resume_last=True)
    assert "No hay conversaciones guardadas" in out


def test_incognito_command(cfg, monkeypatch, capsys):
    _, out = run_chat(cfg, monkeypatch, capsys, "/incognito", "algo privado", "/status", "/incognito", "/quit")
    assert "Incognito ACTIVADO" in out and "incognito (no se guarda)" in out and "Incognito desactivado" in out
    s = open_store(cfg)
    assert s.stats()["counts"]["messages"] == 0
    s.close()


def test_memory_commands_with_memory_disabled(cfg, monkeypatch, capsys):
    cfg["memory"]["enabled"] = False
    _, out = run_chat(cfg, monkeypatch, capsys, "/remember x", "hola", "/status", "/quit")
    assert "La memoria esta deshabilitada" in out and "Hola, mundo!" in out and "'memoria': 'deshabilitada'" in out


def test_chat_continues_without_memory_if_the_database_is_unusable(cfg, monkeypatch, capsys, tmp_path):
    bad = tmp_path / "corrupta.db"
    bad.write_bytes(b"no es sqlite" * 100)
    cfg["memory"]["db_path"] = str(bad)
    _, out = run_chat(cfg, monkeypatch, capsys, "hola", "/quit")
    assert "Hola, mundo!" in out      # el asistente responde igual


def test_unknown_and_help_commands(cfg, monkeypatch, capsys):
    _, out = run_chat(cfg, monkeypatch, capsys, "/loquesea", "/help", "/quit")
    assert "Comando no reconocido" in out and "/remember TEXTO" in out and "/incognito" in out


# -- main.py memory ------------------------------------------------------------------------------------------

def test_memory_stats_export_wipe(cfg, monkeypatch, capsys, tmp_path):
    s = open_store(cfg)
    s.add_memory("semantic", "dato guardado")
    s.close()

    assert main.memory_cmd(cfg, "stats", None) == 0
    assert "memories" in capsys.readouterr().out

    dest = tmp_path / "salida.json"
    assert main.memory_cmd(cfg, "export", str(dest)) == 0
    assert json.loads(dest.read_text(encoding="utf-8"))["memories"][0]["content"] == "dato guardado"
    assert main.memory_cmd(cfg, "export", str(dest)) == 1          # no sobrescribe
    assert main.memory_cmd(cfg, "export", None) == 2

    script(monkeypatch, "no")                                       # respuesta equivocada: no borra
    assert main.memory_cmd(cfg, "wipe", None) == 1
    s = open_store(cfg)
    assert s.stats()["counts"]["memories"] == 1
    s.close()

    script(monkeypatch, "BORRAR TODO")
    assert main.memory_cmd(cfg, "wipe", None) == 0
    s = open_store(cfg)
    assert all(n == 0 for n in s.stats()["counts"].values())
    s.close()


def test_wipe_cancels_on_eof(cfg, monkeypatch, capsys):
    s = open_store(cfg)
    s.add_memory("semantic", "dato")
    s.close()
    script(monkeypatch)                                             # sin respuesta -> EOF
    assert main.memory_cmd(cfg, "wipe", None) == 1
    s = open_store(cfg)
    assert s.stats()["counts"]["memories"] == 1
    s.close()


def test_relaxed_permissions_are_announced_and_audited(tmp_path, capsys):
    import main as poeta_main
    from poeta.config import load_config
    from poeta.security import AuditLog, PermissionManager, read_tail

    cfg = load_config()
    cfg["permissions"]["cloud"]["require_confirmation"] = False
    audit = AuditLog(tmp_path / "a.jsonl")
    pm = PermissionManager(cfg, audit=audit, root=tmp_path)
    poeta_main.announce_permission_warnings(pm, audit)
    err = capsys.readouterr().err
    assert "Aviso de permisos" in err and "Recomendado: true" in err
    ev = [e for e in read_tail(tmp_path / "a.jsonl", 5) if e["event"] == "permissions_relaxed"]
    assert ev and ev[0]["warnings"]


def test_default_permissions_start_silently(tmp_path, capsys):
    import main as poeta_main
    from poeta.config import load_config
    from poeta.security import PermissionManager

    poeta_main.announce_permission_warnings(PermissionManager(load_config(), root=tmp_path), None)
    assert capsys.readouterr().err == ""
