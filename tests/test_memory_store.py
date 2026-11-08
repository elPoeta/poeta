import json
import os
import re
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from poeta.memory import MemoryStore, MemoryStoreError, fts_query
from poeta.security import AuditLog


@pytest.fixture
def store(tmp_path):
    s = MemoryStore(tmp_path / "data" / "memory.db")
    yield s
    s.close()


def raw_bytes(path):
    """Contenido crudo de la base y de sus archivos auxiliares (WAL, SHM)."""
    data = b""
    for suffix in ("", "-wal", "-shm"):
        p = type(path)(f"{path}{suffix}")
        if p.exists():
            data += p.read_bytes()
    return data


# -- esquema y archivo ----------------------------------------------------------------------

def test_schema_is_created_once_and_reopens(tmp_path):
    path = tmp_path / "m.db"
    s1 = MemoryStore(path)
    s1.add_memory("semantic", "dato persistente")
    s1.close()
    s2 = MemoryStore(path)
    assert [m["content"] for m in s2.list_memories()] == ["dato persistente"]
    assert s2._db.execute("PRAGMA user_version").fetchone()[0] == 1
    s2.close()


def test_database_file_is_private(tmp_path):
    path = tmp_path / "m.db"
    s = MemoryStore(path)
    s.add_memory("semantic", "x")
    assert oct(os.stat(path).st_mode & 0o777) == "0o600"
    for suffix in ("-wal", "-shm"):
        aux = type(path)(f"{path}{suffix}")
        if aux.exists():
            assert oct(os.stat(aux).st_mode & 0o777) == "0o600", suffix
    s.close()


def test_newer_schema_is_refused(tmp_path):
    path = tmp_path / "m.db"
    MemoryStore(path).close()
    db = sqlite3.connect(path)
    db.execute("PRAGMA user_version = 99")
    db.commit()
    db.close()
    with pytest.raises(MemoryStoreError, match="mas nueva"):
        MemoryStore(path)


def test_corrupt_file_raises_store_error(tmp_path):
    path = tmp_path / "m.db"
    path.write_bytes(b"esto no es sqlite" * 100)
    with pytest.raises(MemoryStoreError):
        MemoryStore(path)


# -- conversaciones: independientes del modelo y persistentes ------------------------------------

def test_conversation_survives_restart_and_model_changes(tmp_path):
    path = tmp_path / "m.db"
    s = MemoryStore(path)
    cid = s.start_conversation()
    s.add_exchange(cid, "hola", "hola, ¿que tal?", provider="local", model="nemotron")
    s.add_exchange(cid, "sigamos", "dale", provider="nvidia", model="otro-modelo")
    s.close()

    s = MemoryStore(path)  # reinicio
    msgs = s.get_messages(cid)
    assert [(m["role"], m["content"]) for m in msgs] == [
        ("user", "hola"), ("assistant", "hola, ¿que tal?"), ("user", "sigamos"), ("assistant", "dale"),
    ]
    assert [m["model"] for m in msgs] == ["nemotron", "nemotron", "otro-modelo", "otro-modelo"]  # una conversacion, dos modelos
    assert s.last_conversation_id() == cid
    s.close()


def test_get_messages_limit_returns_last_in_order(store):
    cid = store.start_conversation()
    for i in range(5):
        store.add_exchange(cid, f"u{i}", f"a{i}")
    assert [m["content"] for m in store.get_messages(cid, limit=3)] == ["a3", "u4", "a4"]
    assert [m["content"] for m in store.get_messages(cid, limit=4)] == ["u3", "a3", "u4", "a4"]
    assert len(store.get_messages(cid)) == 10


def test_title_comes_from_first_message_only(store):
    cid = store.start_conversation()
    store.add_exchange(cid, "  primera   pregunta  ", "r")
    store.add_exchange(cid, "segunda", "r")
    assert store.list_conversations()[0]["title"] == "primera pregunta"


def test_partial_flag_and_empty_conversations(store):
    cid = store.start_conversation()
    assert store.last_conversation_id() is None  # sin mensajes no cuenta como "ultima"
    store.add_exchange(cid, "pregunta", "respuesta cort", partial=True)
    assert [m["partial"] for m in store.get_messages(cid)] == [0, 1]
    assert store.last_conversation_id() == cid


def test_exchange_is_atomic(store):
    cid = store.start_conversation()
    with pytest.raises(sqlite3.IntegrityError):
        store.add_exchange(cid, "pregunta", None)      # la 2.a fila falla cuando la 1.a ya se inserto
    assert store.stats()["counts"]["messages"] == 0    # la pregunta tampoco quedo
    assert store.search_messages("pregunta") == []
    store.add_exchange(cid, "otra", "ok")              # y la base sigue usable (la transaccion se cerro)
    assert store.stats()["counts"]["messages"] == 2


# -- busqueda ------------------------------------------------------------------------------------------

def test_fts_query_sanitizes_everything():
    assert fts_query("de la y el") is None
    assert fts_query("a") is None
    assert fts_query("") is None
    q = fts_query('cumpleaños "de" Ana OR NEAR(x) * -foo col:bar ")(; DROP')
    # Solo terminos entre comillas (letras/digitos) unidos por OR: ningun operador del usuario sobrevive.
    assert q is not None and re.fullmatch(r'"\w+"\*( OR "\w+"\*)*', q), q
    assert fts_query("uno dos tres cuatro cinco seis siete ocho nueve diez").count(" OR ") == 7   # tope de terminos


def test_search_ignores_accents_and_case(store):
    store.add_memory("semantic", "El CAMIÓN rojo de Ana está en el taller")
    assert [m["content"] for m in store.search_memories("camion")][0].startswith("El CAMIÓN")
    assert store.search_memories("CAMIÓN")


def test_search_matches_word_variants_by_prefix(store):
    store.add_memory("semantic", "Reunion de planificacion con Marta el lunes")
    assert store.search_memories("planificar la reunion")        # 'reunion' coincide
    assert store.search_memories("planificaciones")


def test_search_ranks_better_matches_first(store):
    store.add_memory("semantic", "Me gusta el cafe")
    best = store.add_memory("semantic", "El cafe de especialidad colombiano es mi cafe favorito")
    assert store.search_memories("cafe colombiano")[0]["id"] == best


@pytest.mark.parametrize("hostile", ['"', "'", "OR", "AND NOT", "NEAR(a b)", "*", "col:x", "a -b", '"unclosed', ")(", "; DROP TABLE memories;--"])
def test_hostile_queries_never_raise(store, hostile):
    store.add_memory("semantic", "contenido normal")
    store.search_memories(hostile)
    store.search_messages(hostile)
    assert store.stats()["counts"]["memories"] == 1


def test_search_messages_excludes_current_conversation(store):
    old = store.start_conversation()
    store.add_exchange(old, "mi perro se llama Rocco", "Rocco es un buen nombre")
    now = store.start_conversation()
    store.add_exchange(now, "recuerdas a Rocco", "claro")
    found = store.search_messages("Rocco", exclude_conversation=now)
    assert found and {m["conversation_id"] for m in found} == {old}


def test_search_filters_by_kind_and_hides_expired(store):
    store.add_memory("semantic", "turno en el dentista", conversation_id=None)
    store.add_memory("episodic", "fui al dentista ayer")
    expired = store.add_memory("semantic", "dentista vencido", expires_in_days=1)
    assert {m["kind"] for m in store.search_memories("dentista")} == {"semantic", "episodic"}
    assert [m["kind"] for m in store.search_memories("dentista", kinds=("episodic",))] == ["episodic"]
    store._db.execute("UPDATE memories SET expires_at = '2000-01-01T00:00:00+00:00' WHERE id = ?", (expired,))
    assert all(m["id"] != expired for m in store.search_memories("dentista"))


# -- memorias, preferencias, herramientas -------------------------------------------------------------------

def test_memory_validation(store):
    with pytest.raises(ValueError):
        store.add_memory("inventada", "x")
    with pytest.raises(ValueError):
        store.add_memory("semantic", "   ")
    with pytest.raises(ValueError):
        store.add_memory("semantic", "x" * 4001)


def test_forget_removes_from_index_too(store):
    mid = store.add_memory("semantic", "clave del wifi: manzana")
    assert store.search_memories("wifi")
    assert store.forget(mid) is True
    assert store.search_memories("wifi") == [] and store.forget(mid) is False
    # el indice FTS quedo consistente
    assert store._db.execute("INSERT INTO memories_fts(memories_fts) VALUES ('integrity-check')").rowcount >= 0


def test_preferences(store):
    store.set_preference("Idioma", "espanol")
    store.set_preference("idioma", "español rioplatense")   # sobrescribe
    store.set_preference("tono.formal", "no")
    assert store.get_preferences() == {"idioma": "español rioplatense", "tono.formal": "no"}
    assert store.delete_preference("IDIOMA") is True and store.delete_preference("idioma") is False
    for bad_key in ("con espacio", "", "x" * 65, "ñandu", "a;b"):
        with pytest.raises(ValueError):
            store.set_preference(bad_key, "v")
    with pytest.raises(ValueError):
        store.set_preference("k", "")
    with pytest.raises(ValueError):
        store.set_preference("k", "v" * 501)


def test_tool_log_redacts_secrets_and_truncates(store):
    store.log_tool("buscar", {"q": "hola", "api_key": "SECRETO-123", "nota": "usa nvapi-ABCDEFGH12345"}, "x" * 1000, "ok", request_id="r1")
    row = store.tool_history()[0]
    assert "SECRETO-123" not in row["args"] and "nvapi-ABCD" not in row["args"] and "hola" in row["args"]
    assert len(row["result"]) < 400 and row["status"] == "ok" and row["request_id"] == "r1"


# -- retencion ----------------------------------------------------------------------------------------------------

def test_expired_memories_are_purged(store):
    keep = store.add_memory("semantic", "vigente", expires_in_days=30)
    gone = store.add_memory("semantic", "vencida", expires_in_days=1)
    forever = store.add_memory("semantic", "siempre")
    later = datetime.now(timezone.utc) + timedelta(days=2)
    assert store.purge_expired(now=later)["memories"] == 1
    assert {m["id"] for m in store.list_memories()} == {keep, forever} and gone not in {m["id"] for m in store.list_memories()}


def test_message_retention_removes_old_messages_and_empty_conversations(tmp_path):
    s = MemoryStore(tmp_path / "m.db", retention_days={"messages": 30, "episodic": 10, "tool_log": 5})
    old, recent = s.start_conversation(), s.start_conversation()
    s.add_exchange(old, "mensaje viejo", "r")
    s.add_exchange(recent, "mensaje reciente", "r")
    long_ago = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat(timespec="seconds")
    s._db.execute("UPDATE messages SET ts = ? WHERE conversation_id = ?", (long_ago, old))
    s._db.execute("UPDATE conversations SET started_at = ? WHERE id = ?", (long_ago, old))
    s.add_memory("episodic", "evento viejo")
    s._db.execute("UPDATE memories SET created_at = ?", (long_ago,))
    s.add_memory("semantic", "nota vieja pero semantica")
    s._db.execute("UPDATE memories SET created_at = ? WHERE kind = 'semantic'", (long_ago,))
    s.log_tool("t", {}, "r", "ok")
    s._db.execute("UPDATE tool_log SET ts = ?", (long_ago,))

    removed = s.purge_expired()
    assert removed == {"memories": 1, "messages": 2, "conversations": 1, "tool_log": 1}
    assert [c["id"] for c in s.list_conversations()] == [recent]
    assert [m["kind"] for m in s.list_memories()] == ["semantic"]    # la semantica no caduca por retencion episodica
    assert s.search_messages("viejo") == []                          # tampoco queda en el indice
    s.close()


def test_no_retention_means_keep_forever(store):
    cid = store.start_conversation()
    store.add_exchange(cid, "viejo", "r")
    store._db.execute("UPDATE messages SET ts = '2000-01-01T00:00:00+00:00'")
    assert store.purge_expired()["messages"] == 0


# -- exportar y borrar todo --------------------------------------------------------------------------------------------

def test_export_is_complete_private_and_does_not_overwrite(store, tmp_path):
    cid = store.start_conversation()
    store.add_exchange(cid, "hola", "chau")
    store.add_memory("semantic", "nota")
    store.set_preference("idioma", "es")
    store.log_tool("t", {}, "r", "ok")
    dest = tmp_path / "out.json"
    counts = store.export(dest)
    assert counts == {"messages": 2, "memories": 1, "preferences": 1, "tool_log": 1, "conversations": 1}
    data = json.loads(dest.read_text(encoding="utf-8"))
    assert [m["content"] for m in data["messages"]] == ["hola", "chau"] and data["preferences"][0]["key"] == "idioma"
    assert oct(os.stat(dest).st_mode & 0o777) == "0o600"
    with pytest.raises(FileExistsError):
        store.export(dest)


def test_export_does_not_follow_a_symlink(store, tmp_path):
    """Un enlace roto en el destino no debe hacer que se cree (o pise) el archivo al que apunta."""
    target = tmp_path / "otro_lugar.json"
    link = tmp_path / "salida.json"
    os.symlink(target, link)
    with pytest.raises(FileExistsError):
        store.export(link)
    assert not target.exists()


def test_forget_and_unprefer_leave_no_remnants_in_db_or_wal(tmp_path):
    """Regresion: sin truncar el WAL, el texto olvidado seguia legible en el archivo -wal."""
    path = tmp_path / "m.db"
    s = MemoryStore(path)
    note = s.add_memory("semantic", "ZXQ-nota-que-hay-que-olvidar")
    s.add_memory("semantic", "ZXQ-nota-que-se-queda")
    s.set_preference("pin", "ZXQ-pin-1234")
    assert b"ZXQ-nota-que-hay-que-olvidar" in raw_bytes(path) and b"ZXQ-pin-1234" in raw_bytes(path)

    s.forget(note)
    s.delete_preference("pin")
    on_disk = raw_bytes(path)
    assert b"ZXQ-nota-que-hay-que-olvidar" not in on_disk
    assert b"ZXQ-pin-1234" not in on_disk
    assert b"ZXQ-nota-que-se-queda" in on_disk          # lo demas no se toca
    assert [m["content"] for m in s.search_memories("queda")] == ["ZXQ-nota-que-se-queda"]
    s.close()


def test_purge_leaves_no_remnants_in_db_or_wal(tmp_path):
    path = tmp_path / "m.db"
    s = MemoryStore(path)
    s.add_memory("semantic", "ZXQ-vencida", expires_in_days=1)
    assert b"ZXQ-vencida" in raw_bytes(path)
    s.purge_expired(now=datetime.now(timezone.utc) + timedelta(days=2))
    assert b"ZXQ-vencida" not in raw_bytes(path)
    s.close()


def test_wipe_removes_everything_including_raw_bytes(tmp_path):
    path = tmp_path / "m.db"
    s = MemoryStore(path)
    cid = s.start_conversation()
    secret_msg, secret_note = "ZXQ-mensaje-ultrasecreto", "ZXQ-nota-ultrasecreta"
    s.add_exchange(cid, secret_msg, "respuesta " + secret_msg)
    s.add_memory("semantic", secret_note)
    s.set_preference("pin", "ZXQ-9999")
    s.log_tool("t", {"x": "ZXQ-arg"}, "ZXQ-res", "ok")
    assert b"ZXQ-" in raw_bytes(path)            # el centinela esta en disco antes de borrar

    counts = s.wipe()
    assert counts["messages"] == 2 and counts["memories"] == 1
    assert all(n == 0 for n in s.stats()["counts"].values())
    assert s.search_messages("ultrasecreto") == [] and s.search_memories("ultrasecreta") == []
    assert b"ZXQ-" not in raw_bytes(path), "quedaron restos del contenido en el archivo"
    s.add_memory("semantic", "se puede seguir usando")   # la base sigue operativa
    s.close()


def test_wipe_reclaims_disk_space(tmp_path):
    path = tmp_path / "m.db"
    s = MemoryStore(path)
    cid = s.start_conversation()
    for i in range(300):
        s.add_exchange(cid, f"pregunta {i} " + "x" * 1000, f"respuesta {i} " + "y" * 1000)
    s._db.execute("PRAGMA wal_checkpoint(TRUNCATE)")   # los datos pasan del WAL al archivo principal
    before = path.stat().st_size
    assert before > 400_000
    s.wipe()
    assert s.stats()["bytes"] < before / 5             # sin VACUUM el archivo conservaria las paginas liberadas
    s.close()


def test_wipe_also_clears_orphan_entries_in_the_search_index(tmp_path):
    path = tmp_path / "m.db"
    s = MemoryStore(path)
    s.add_memory("semantic", "nota normal")
    # entrada huerfana en el indice (sin fila en memories), como la dejaria un fallo anterior
    # (el indice guarda tokens en minuscula, no el texto original: por eso un unico token sin guiones)
    s._db.execute("INSERT INTO memories_fts(rowid, content) VALUES (999, 'zxqfantasmahuerfano')")
    assert b"zxqfantasmahuerfano" in raw_bytes(path)
    s.wipe()
    assert b"zxqfantasmahuerfano" not in raw_bytes(path)
    s.close()


# -- auditoria ------------------------------------------------------------------------------------------------------------------

def test_forget_export_and_wipe_are_audited_without_content(tmp_path):
    audit_file = tmp_path / "audit.jsonl"
    s = MemoryStore(tmp_path / "m.db", audit=AuditLog(audit_file))
    mid = s.add_memory("semantic", "CONTENIDO-PRIVADO-777")
    s.forget(mid)
    s.add_memory("semantic", "otro CONTENIDO-PRIVADO-777")
    s.export(tmp_path / "e.json")
    s.wipe()
    raw = audit_file.read_text(encoding="utf-8")
    events = [json.loads(line)["event"] for line in raw.splitlines()]
    assert events == ["memory_forgotten", "memory_exported", "memory_wiped"]
    assert "CONTENIDO-PRIVADO-777" not in raw
    s.close()
