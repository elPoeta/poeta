"""Memoria persistente en SQLite con busqueda de texto FTS5 (plan secc. 11).

Tipos de memoria:
- conversacional: tablas conversations / messages
- episodica y semantica: tabla memories (kind = 'episodic' | 'semantic')
- preferencias: tabla preferences (solo por accion explicita del usuario)
- operativa: tabla tool_log (historial de herramientas, con argumentos redactados)

Garantias:
- Independiente del modelo y del proveedor: cada mensaje anota que proveedor/modelo lo produjo, pero
  la conversacion es una sola aunque se cambie de modelo.
- Archivo privado (0600) y borrado seguro (PRAGMA secure_delete) al olvidar o borrar todo.
- Las consultas nunca arman sintaxis FTS5 con texto del usuario sin sanearlo (ver fts_query).
- Nada de aqui se envia a la nube: lo decide quien arma el contexto (ver context.py).
"""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

from poeta.security.audit import AuditLog, redact

log = logging.getLogger("poeta.memory")

SCHEMA_VERSION = 1
KINDS = ("episodic", "semantic")
MAX_MEMORY_CHARS = 4000
MAX_PREF_VALUE_CHARS = 500
_PREF_KEY = re.compile(r"^[a-z0-9_.-]{1,64}$")
_WORD = re.compile(r"[^\W_]+", re.UNICODE)
_STOPWORDS = frozenset(
    """para como pero por con sin sobre entre desde hasta hacia que cual cuales quien donde cuando
    cuanto esto esta este estos estas eso esa ese esos esas aqui alli asi muy mas menos tambien
    ademas porque pues aunque cada todo toda todos todas otro otra otros otras mismo misma uno una
    unos unas del los las les nos mis tus sus mio mia tuyo tuya suyo suya hay fue era son ser soy
    eres estoy estas esta estan tengo tiene tienen hacer hace haces puedo puedes puede quiero quieres
    quiere dime dame hola gracias favor""".split()
)

_SCHEMA_V1 = """
BEGIN;
CREATE TABLE conversations (
    id INTEGER PRIMARY KEY,
    started_at TEXT NOT NULL,
    title TEXT
);
CREATE TABLE messages (
    id INTEGER PRIMARY KEY,
    conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    ts TEXT NOT NULL,
    provider TEXT,
    model TEXT,
    partial INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX idx_messages_conversation ON messages(conversation_id, id);
CREATE INDEX idx_messages_ts ON messages(ts);
CREATE TABLE memories (
    id INTEGER PRIMARY KEY,
    kind TEXT NOT NULL CHECK (kind IN ('episodic', 'semantic')),
    content TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'user',
    conversation_id INTEGER REFERENCES conversations(id) ON DELETE SET NULL,
    created_at TEXT NOT NULL,
    expires_at TEXT
);
CREATE TABLE preferences (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'user',
    updated_at TEXT NOT NULL
);
CREATE TABLE tool_log (
    id INTEGER PRIMARY KEY,
    ts TEXT NOT NULL,
    conversation_id INTEGER REFERENCES conversations(id) ON DELETE SET NULL,
    tool TEXT NOT NULL,
    args TEXT NOT NULL,
    result TEXT,
    status TEXT NOT NULL,
    request_id TEXT
);
CREATE VIRTUAL TABLE messages_fts USING fts5(
    content, content='messages', content_rowid='id', tokenize='unicode61 remove_diacritics 2'
);
CREATE TRIGGER messages_ai AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts(rowid, content) VALUES (new.id, new.content);
END;
CREATE TRIGGER messages_ad AFTER DELETE ON messages BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, content) VALUES ('delete', old.id, old.content);
END;
CREATE VIRTUAL TABLE memories_fts USING fts5(
    content, content='memories', content_rowid='id', tokenize='unicode61 remove_diacritics 2'
);
CREATE TRIGGER memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, content) VALUES (new.id, new.content);
END;
CREATE TRIGGER memories_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content) VALUES ('delete', old.id, old.content);
END;
PRAGMA user_version = 1;
COMMIT;
"""

_TABLES = ("messages", "memories", "preferences", "tool_log", "conversations")


class MemoryStoreError(Exception):
    """La base de memoria no se pudo abrir, es de una version mas nueva o no soporta FTS5."""


def fts_query(text: str, max_terms: int = 8) -> str | None:
    """Convierte texto libre en una consulta FTS5 segura, o None si no queda nada util.

    Solo pasan palabras (letras y digitos): operadores, comillas, '*', ':', 'NEAR', '-' y similares
    no llegan nunca a la sintaxis de FTS5. Las palabras largas se acortan para emparejar variantes
    (un stemming muy simple para espanol) y se buscan como prefijo.
    """
    terms: list[str] = []
    for word in _WORD.findall(text.lower()):
        if len(word) < 3 or word in _STOPWORDS or word in terms:
            continue
        terms.append(word)
    if not terms:
        return None
    parts = []
    for word in terms[:max_terms]:
        stem = word if len(word) < 6 else word[: max(4, len(word) - 2)]
        parts.append(f'"{stem}"*')
    return " OR ".join(parts)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds")


def memory_path(cfg: dict[str, Any], root: Path | str) -> Path:
    p = Path(cfg["memory"]["db_path"])
    return p if p.is_absolute() else Path(root) / p


class MemoryStore:
    def __init__(
        self,
        path: Path | str,
        audit: AuditLog | None = None,
        retention_days: dict[str, int | None] | None = None,
    ) -> None:
        self.path = Path(path)
        self._audit = audit
        self._retention = retention_days or {}
        self._lock = threading.RLock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            os.close(os.open(self.path, os.O_CREAT | os.O_WRONLY, 0o600))
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass
        try:
            self._db = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
            self._db.row_factory = sqlite3.Row
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("PRAGMA foreign_keys=ON")
            self._db.execute("PRAGMA secure_delete=ON")
            self._db.execute("PRAGMA synchronous=NORMAL")
            self._migrate()
        except sqlite3.Error as exc:
            raise MemoryStoreError(f"No se pudo abrir la memoria en {self.path}: {exc}") from exc

    @classmethod
    def from_config(cls, cfg: dict[str, Any], root: Path | str, audit: AuditLog | None = None) -> "MemoryStore":
        return cls(memory_path(cfg, root), audit=audit, retention_days=cfg["memory"].get("retention_days"))

    def close(self) -> None:
        with self._lock:
            self._db.close()

    # -- infraestructura ----------------------------------------------------

    def _migrate(self) -> None:
        version = self._db.execute("PRAGMA user_version").fetchone()[0]
        if version > SCHEMA_VERSION:
            raise MemoryStoreError(
                f"La base fue creada por una version mas nueva de POETA (esquema {version}, este soporta {SCHEMA_VERSION})."
            )
        if version == 0:
            try:
                self._db.executescript(_SCHEMA_V1)
            except sqlite3.Error as exc:
                try:
                    self._db.execute("ROLLBACK")
                except sqlite3.Error:
                    pass
                if "fts5" in str(exc).lower():
                    raise MemoryStoreError("Este SQLite no incluye FTS5, necesario para la memoria.") from exc
                raise

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                yield self._db
            except BaseException:
                self._db.execute("ROLLBACK")
                raise
            self._db.execute("COMMIT")

    def _scrub(self) -> None:
        """Tras un borrado, trunca el WAL: sin esto las versiones anteriores de las paginas (con el texto
        que se acaba de olvidar) siguen legibles en el archivo -wal hasta el proximo checkpoint."""
        with self._lock:
            try:
                busy = self._db.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()[0]
            except sqlite3.Error:
                busy = 1
        if busy:
            log.warning("no se pudo truncar el WAL tras un borrado; pueden quedar restos hasta el proximo checkpoint")

    def _audit_event(self, event: str, **fields: Any) -> None:
        if self._audit is None:
            return
        try:
            self._audit.record(event, **fields)
        except Exception:  # noqa: BLE001 - la accion del usuario ya ocurrio; se informa y se sigue
            log.exception("no se pudo registrar %s en la auditoria", event)

    @staticmethod
    def _rows(cursor: sqlite3.Cursor) -> list[dict[str, Any]]:
        return [dict(r) for r in cursor.fetchall()]

    # -- conversaciones ---------------------------------------------------------

    def start_conversation(self, title: str | None = None) -> int:
        with self._tx() as db:
            return db.execute(
                "INSERT INTO conversations(started_at, title) VALUES (?, ?)", (_iso(_now()), title)
            ).lastrowid

    def add_exchange(
        self,
        conversation_id: int,
        user_text: str,
        assistant_text: str,
        provider: str | None = None,
        model: str | None = None,
        partial: bool = False,
    ) -> None:
        """Guarda la pregunta y la respuesta como una unidad (o ninguna)."""
        ts = _iso(_now())
        with self._tx() as db:
            db.execute(
                "INSERT INTO messages(conversation_id, role, content, ts, provider, model, partial) "
                "VALUES (?, 'user', ?, ?, ?, ?, 0)",
                (conversation_id, user_text, ts, provider, model),
            )
            db.execute(
                "INSERT INTO messages(conversation_id, role, content, ts, provider, model, partial) "
                "VALUES (?, 'assistant', ?, ?, ?, ?, ?)",
                (conversation_id, assistant_text, ts, provider, model, int(partial)),
            )
            title = " ".join(user_text.split())[:60]
            db.execute("UPDATE conversations SET title = ? WHERE id = ? AND title IS NULL", (title, conversation_id))

    def get_messages(self, conversation_id: int, limit: int | None = None) -> list[dict[str, Any]]:
        """Mensajes en orden cronologico; con `limit`, los ultimos N."""
        with self._lock:
            if limit is None:
                cur = self._db.execute("SELECT * FROM messages WHERE conversation_id = ? ORDER BY id", (conversation_id,))
                return self._rows(cur)
            cur = self._db.execute(
                "SELECT * FROM messages WHERE conversation_id = ? ORDER BY id DESC LIMIT ?", (conversation_id, limit)
            )
            return list(reversed(self._rows(cur)))

    def list_conversations(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            cur = self._db.execute(
                "SELECT c.id, c.started_at, c.title, COUNT(m.id) AS messages FROM conversations c "
                "LEFT JOIN messages m ON m.conversation_id = c.id GROUP BY c.id ORDER BY c.id DESC LIMIT ?",
                (limit,),
            )
            return self._rows(cur)

    def last_conversation_id(self) -> int | None:
        with self._lock:
            row = self._db.execute(
                "SELECT id FROM conversations c WHERE EXISTS (SELECT 1 FROM messages m WHERE m.conversation_id = c.id) "
                "ORDER BY id DESC LIMIT 1"
            ).fetchone()
            return row["id"] if row else None

    def conversation_exists(self, conversation_id: int) -> bool:
        with self._lock:
            return self._db.execute("SELECT 1 FROM conversations WHERE id = ?", (conversation_id,)).fetchone() is not None

    def search_messages(self, text: str, exclude_conversation: int | None = None, limit: int = 5) -> list[dict[str, Any]]:
        query = fts_query(text)
        if query is None or limit <= 0:
            return []
        sql = (
            "SELECT m.id, m.conversation_id, m.role, m.content, m.ts FROM messages_fts "
            "JOIN messages m ON m.id = messages_fts.rowid WHERE messages_fts MATCH ?"
        )
        params: list[Any] = [query]
        if exclude_conversation is not None:
            sql += " AND m.conversation_id != ?"
            params.append(exclude_conversation)
        sql += " ORDER BY bm25(messages_fts) LIMIT ?"
        params.append(limit)
        with self._lock:
            return self._rows(self._db.execute(sql, params))

    # -- memorias (episodica y semantica) ---------------------------------------------

    def add_memory(
        self,
        kind: str,
        content: str,
        source: str = "user",
        conversation_id: int | None = None,
        expires_in_days: int | None = None,
    ) -> int:
        if kind not in KINDS:
            raise ValueError(f"kind debe ser uno de {KINDS}")
        content = content.strip()
        if not content:
            raise ValueError("El contenido no puede estar vacio.")
        if len(content) > MAX_MEMORY_CHARS:
            raise ValueError(f"El contenido supera {MAX_MEMORY_CHARS} caracteres.")
        expires = _iso(_now() + timedelta(days=expires_in_days)) if expires_in_days else None
        with self._tx() as db:
            return db.execute(
                "INSERT INTO memories(kind, content, source, conversation_id, created_at, expires_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (kind, content, source, conversation_id, _iso(_now()), expires),
            ).lastrowid

    def search_memories(self, text: str, kinds: tuple[str, ...] | None = None, limit: int = 5) -> list[dict[str, Any]]:
        query = fts_query(text)
        if query is None or limit <= 0:
            return []
        sql = (
            "SELECT m.id, m.kind, m.content, m.created_at FROM memories_fts "
            "JOIN memories m ON m.id = memories_fts.rowid WHERE memories_fts MATCH ? "
            "AND (m.expires_at IS NULL OR m.expires_at > ?)"
        )
        params: list[Any] = [query, _iso(_now())]
        if kinds:
            sql += f" AND m.kind IN ({','.join('?' * len(kinds))})"
            params.extend(kinds)
        sql += " ORDER BY bm25(memories_fts) LIMIT ?"
        params.append(limit)
        with self._lock:
            return self._rows(self._db.execute(sql, params))

    def list_memories(self, kind: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        sql = "SELECT id, kind, content, source, created_at, expires_at FROM memories"
        params: list[Any] = []
        if kind:
            sql += " WHERE kind = ?"
            params.append(kind)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        with self._lock:
            return self._rows(self._db.execute(sql, params))

    def forget(self, memory_id: int) -> bool:
        with self._tx() as db:
            row = db.execute("SELECT kind FROM memories WHERE id = ?", (memory_id,)).fetchone()
            if row is None:
                return False
            db.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        self._scrub()
        self._audit_event("memory_forgotten", memory_id=memory_id, kind=row["kind"])
        return True

    # -- preferencias -----------------------------------------------------------------------

    def set_preference(self, key: str, value: str, source: str = "user") -> None:
        key = key.strip().lower()
        value = value.strip()
        if not _PREF_KEY.match(key):
            raise ValueError("La clave debe usar minusculas, digitos, '.', '_' o '-' (maximo 64).")
        if not value or len(value) > MAX_PREF_VALUE_CHARS:
            raise ValueError(f"El valor debe tener entre 1 y {MAX_PREF_VALUE_CHARS} caracteres.")
        with self._tx() as db:
            db.execute(
                "INSERT INTO preferences(key, value, source, updated_at) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value, source = excluded.source, "
                "updated_at = excluded.updated_at",
                (key, value, source, _iso(_now())),
            )

    def get_preferences(self) -> dict[str, str]:
        with self._lock:
            return {r["key"]: r["value"] for r in self._db.execute("SELECT key, value FROM preferences ORDER BY key")}

    def delete_preference(self, key: str) -> bool:
        with self._tx() as db:
            deleted = db.execute("DELETE FROM preferences WHERE key = ?", (key.strip().lower(),)).rowcount > 0
        if deleted:
            self._scrub()
        return deleted

    # -- memoria operativa ----------------------------------------------------------------------

    def log_tool(
        self,
        tool: str,
        args: dict[str, Any],
        result: str | None,
        status: str,
        conversation_id: int | None = None,
        request_id: str | None = None,
    ) -> int:
        """Registra el uso de una herramienta. Argumentos y resultado se redactan y se acotan."""
        safe_args = json.dumps(redact(args), ensure_ascii=False, sort_keys=True)
        safe_result = redact(result) if result is not None else None
        with self._tx() as db:
            return db.execute(
                "INSERT INTO tool_log(ts, conversation_id, tool, args, result, status, request_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (_iso(_now()), conversation_id, tool, safe_args, safe_result, status, request_id),
            ).lastrowid

    def tool_history(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            return self._rows(self._db.execute("SELECT * FROM tool_log ORDER BY id DESC LIMIT ?", (limit,)))

    # -- mantenimiento -----------------------------------------------------------------------------

    def purge_expired(self, now: datetime | None = None) -> dict[str, int]:
        """Aplica expiraciones y retencion configurada. Devuelve cuantas filas se borraron por tabla."""
        now = now or _now()
        removed = {"memories": 0, "messages": 0, "conversations": 0, "tool_log": 0}

        def cutoff(days: int) -> str:
            return _iso(now - timedelta(days=days))

        with self._tx() as db:
            removed["memories"] += db.execute(
                "DELETE FROM memories WHERE expires_at IS NOT NULL AND expires_at <= ?", (_iso(now),)
            ).rowcount
            if self._retention.get("episodic"):
                removed["memories"] += db.execute(
                    "DELETE FROM memories WHERE kind = 'episodic' AND created_at < ?", (cutoff(self._retention["episodic"]),)
                ).rowcount
            if self._retention.get("messages"):
                removed["messages"] = db.execute(
                    "DELETE FROM messages WHERE ts < ?", (cutoff(self._retention["messages"]),)
                ).rowcount
                removed["conversations"] = db.execute(
                    "DELETE FROM conversations WHERE NOT EXISTS (SELECT 1 FROM messages m WHERE m.conversation_id = conversations.id) "
                    "AND started_at < ?",
                    (cutoff(self._retention["messages"]),),
                ).rowcount
            if self._retention.get("tool_log"):
                removed["tool_log"] = db.execute(
                    "DELETE FROM tool_log WHERE ts < ?", (cutoff(self._retention["tool_log"]),)
                ).rowcount
        if any(removed.values()):
            self._scrub()
        return removed

    def stats(self) -> dict[str, Any]:
        with self._lock:
            counts = {t: self._db.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in _TABLES}
        size = sum(p.stat().st_size for p in (self.path, Path(f"{self.path}-wal")) if p.exists())
        return {"path": str(self.path), "counts": counts, "bytes": size}

    def integrity_check(self) -> str:
        with self._lock:
            return self._db.execute("PRAGMA quick_check").fetchone()[0]

    def export(self, dest: Path | str) -> dict[str, int]:
        """Vuelca todo a un JSON privado (0600). Es texto plano: tratalo como los datos que contiene."""
        dest = Path(dest)
        if dest.exists():
            raise FileExistsError(f"{dest} ya existe; elige otro nombre.")
        with self._lock:
            data = {
                t: self._rows(self._db.execute(f"SELECT * FROM {t} ORDER BY {'key' if t == 'preferences' else 'id'}"))
                for t in _TABLES
            }
        fd = os.open(dest, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({"schema": SCHEMA_VERSION, "exported_at": _iso(_now()), **data}, f, ensure_ascii=False, indent=2)
        counts = {t: len(rows) for t, rows in data.items()}
        self._audit_event("memory_exported", counts=counts)
        return counts

    def wipe(self) -> dict[str, int]:
        """Borra TODO. Con secure_delete activo las paginas se sobrescriben; en un SSD no hay garantia fisica."""
        counts = self.stats()["counts"]
        with self._tx() as db:
            for table in _TABLES:  # orden: hijos antes que conversations
                db.execute(f"DELETE FROM {table}")
        with self._lock:
            for fts in ("messages_fts", "memories_fts"):
                self._db.execute(f"INSERT INTO {fts}({fts}) VALUES ('rebuild')")
            self._db.execute("VACUUM")
            self._db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        self._audit_event("memory_wiped", counts=counts)
        return counts
