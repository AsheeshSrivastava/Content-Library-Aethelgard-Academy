"""The index file (spec 17 section 2, ADR 0011): one SQLite file holding sources, passages (with their text) and vectors.

One writer (a file lock), many read-only readers (write-ahead mode). The file records the exact setup it was built with
and refuses a different one. A source is replaced in one transaction, so a crash leaves a valid index."""
from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ragcore.contracts import content_sha256

SCHEMA_VERSION = 1
UNIT_TOLERANCE = 1e-3
LOOKUP_BATCH = 500

# from_version -> a function that brings an open file from that version to the next one (inside one transaction)
_MIGRATIONS: dict[int, Callable[[sqlite3.Connection], None]] = {}

_SCHEMA = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE sources (
    source_id TEXT PRIMARY KEY, version TEXT NOT NULL, content_sha256 TEXT NOT NULL, chunker_version TEXT NOT NULL,
    url TEXT NOT NULL, licence TEXT NOT NULL, loaded_at TEXT NOT NULL);
CREATE TABLE passages (
    id INTEGER PRIMARY KEY, passage_id TEXT NOT NULL UNIQUE,
    source_id TEXT NOT NULL REFERENCES sources(source_id) ON DELETE CASCADE,
    start_pos INTEGER NOT NULL, end_pos INTEGER NOT NULL, heading_path TEXT NOT NULL, text TEXT NOT NULL,
    text_hash TEXT NOT NULL);
CREATE INDEX passages_source ON passages(source_id);
CREATE INDEX passages_text_hash ON passages(text_hash);
CREATE TABLE vectors (
    passage_id TEXT PRIMARY KEY REFERENCES passages(passage_id) ON DELETE CASCADE,
    embedder_id TEXT NOT NULL, dim INTEGER NOT NULL, vector BLOB NOT NULL);
"""


class IndexStoreError(Exception):
    """The index file cannot be opened or changed as asked; nothing was changed."""


@dataclass(frozen=True)
class IndexSetup:
    """What an index was built with. A file only opens with exactly this setup."""
    embedder_id: str
    embedder_revision: str
    chunker: str
    chunker_version: str
    chunk_config: str
    llama_index_version: str
    reranker_id: str
    reranker_revision: str


@dataclass(frozen=True)
class SourceRow:
    source_id: str
    version: str
    content_sha256: str
    chunker_version: str
    url: str
    licence: str
    loaded_at: str


@dataclass(frozen=True)
class PassageRow:
    passage_id: str
    source_id: str
    start: int
    end: int
    heading_path: tuple[str, ...]
    text: str
    text_hash: str


class _FileLock:
    """An operating-system lock on `<index>.lock`: released by the system if the process dies."""

    def __init__(self, path: Path) -> None:
        self._path, self._file = path, None

    def acquire(self) -> None:
        handle = open(self._path, "a+b")
        try:
            if handle.seek(0, os.SEEK_END) == 0:
                handle.write(b"x")
                handle.flush()
            handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            handle.close()
            raise IndexStoreError("another sync is writing to this index; wait for it to finish") from None
        self._file = handle

    def release(self) -> None:
        if self._file is None:
            return
        try:
            if os.name == "nt":
                import msvcrt
                self._file.seek(0)
                msvcrt.locking(self._file.fileno(), msvcrt.LK_UNLCK, 1)
        finally:
            self._file.close()
            self._file = None


def _read_meta(conn: sqlite3.Connection) -> dict[str, str]:
    return dict(conn.execute("SELECT key, value FROM meta"))


def _check_setup(path: Path, meta: dict[str, str], setup: IndexSetup) -> None:
    different = [f.name for f in fields(setup) if meta.get(f.name) != getattr(setup, f.name)]
    if different:
        raise IndexStoreError(f"{path} was built with a different setup ({', '.join(different)}); "
                              "rebuild the index with the new setup instead of mixing models")


def _upgrade(conn: sqlite3.Connection, path: Path, found: int) -> None:
    backup = path.with_name(f"{path.name}.bak-{found}")
    if backup.exists():
        backup.unlink()
    copy = sqlite3.connect(backup)
    try:
        conn.backup(copy)
    finally:
        copy.close()
    try:
        conn.execute("BEGIN IMMEDIATE")
        for version in range(found, SCHEMA_VERSION):
            _MIGRATIONS[version](conn)
            conn.execute("UPDATE meta SET value = ? WHERE key = 'schema_version'", (str(version + 1),))
        conn.execute("COMMIT")
    except Exception as exc:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise IndexStoreError(f"could not upgrade {path} from schema {found} to {SCHEMA_VERSION} ({exc}); "
                              f"the old file is unchanged and a copy is at {backup}") from exc


def open_index(path: Path | str, setup: IndexSetup, *, read_only: bool = False, create: bool = True) -> IndexStore:
    """Open (or create) the index file. A writer takes the lock; a read-only open never does."""
    path = Path(path)
    if not path.exists() and (read_only or not create):
        raise IndexStoreError(f"{path} does not exist")
    lock = None
    if read_only:
        conn = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True, isolation_level=None)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        lock = _FileLock(path.with_name(path.name + ".lock"))
        lock.acquire()
        conn = sqlite3.connect(path, isolation_level=None)
    try:
        _prepare(conn, path, setup, read_only)
    except BaseException:
        conn.close()
        if lock:
            lock.release()
        raise
    return IndexStore(conn, path, lock, read_only)


def _prepare(conn: sqlite3.Connection, path: Path, setup: IndexSetup, read_only: bool) -> None:
    has_meta = conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'meta'").fetchone()
    if not has_meta:
        if read_only or conn.execute("SELECT 1 FROM sqlite_master").fetchone():
            raise IndexStoreError(f"{path} is not an index file")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("BEGIN IMMEDIATE")
        for statement in _SCHEMA.split(";"):
            if statement.strip():
                conn.execute(statement)
        meta = {"schema_version": str(SCHEMA_VERSION), "created_at": datetime.now(timezone.utc).isoformat(),
                **{f.name: getattr(setup, f.name) for f in fields(setup)}}
        conn.executemany("INSERT INTO meta (key, value) VALUES (?, ?)", list(meta.items()))
        conn.execute("COMMIT")
        return
    meta = _read_meta(conn)
    found = int(meta.get("schema_version", "0"))
    if found > SCHEMA_VERSION:
        raise IndexStoreError(f"{path} has schema {found}, newer than this code understands ({SCHEMA_VERSION}); "
                              "update the library")
    _check_setup(path, meta, setup)
    if not read_only:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA foreign_keys = ON")
    if found < SCHEMA_VERSION:
        if read_only:
            raise IndexStoreError(f"{path} has an older schema ({found}); run a sync to upgrade it")
        _upgrade(conn, path, found)


class IndexStore:
    def __init__(self, conn: sqlite3.Connection, path: Path, lock: _FileLock | None, read_only: bool) -> None:
        self._conn, self.path, self._lock, self.read_only = conn, path, lock, read_only

    def __enter__(self) -> IndexStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        if self._conn is None:
            return
        self._conn.close()
        self._conn = None
        if self._lock:
            self._lock.release()

    def _db(self) -> sqlite3.Connection:
        if self._conn is None:
            raise IndexStoreError("the index is closed")
        return self._conn

    def _writable(self) -> sqlite3.Connection:
        if self.read_only:
            raise IndexStoreError("this index was opened read-only")
        return self._db()

    def meta(self) -> dict[str, str]:
        return _read_meta(self._db())

    def counts(self) -> dict[str, int]:
        db = self._db()
        return {name: db.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
                for name in ("sources", "passages", "vectors")}

    def sources(self) -> dict[str, SourceRow]:
        query = ("SELECT source_id, version, content_sha256, chunker_version, url, licence, loaded_at "
                 "FROM sources ORDER BY source_id")
        return {row[0]: SourceRow(*row) for row in self._db().execute(query)}

    def load_all(self) -> tuple[list[PassageRow], np.ndarray]:
        """Every passage in the order they were added, and their vectors as one float32 array (one row each)."""
        db = self._db()
        query = ("SELECT p.passage_id, p.source_id, p.start_pos, p.end_pos, p.heading_path, p.text, p.text_hash, v.vector "
                 "FROM passages p JOIN vectors v ON v.passage_id = p.passage_id ORDER BY p.id")
        found = db.execute(query).fetchall()
        if len(found) != db.execute("SELECT COUNT(*) FROM passages").fetchone()[0]:
            raise IndexStoreError("a passage has no vector; the index file is damaged, rebuild it")
        rows = [PassageRow(r[0], r[1], r[2], r[3], tuple(json.loads(r[4])), r[5], r[6]) for r in found]
        dim = int(_read_meta(db).get("dim", "0"))
        if not rows:
            return rows, np.zeros((0, dim), dtype=np.float32)
        return rows, np.frombuffer(b"".join(r[7] for r in found), dtype=np.float32).reshape(len(rows), dim)

    def find_vectors(self, text_hashes: list[str]) -> dict[str, np.ndarray]:
        """The stored vector of each text hash that is in the index (the first one added, if several)."""
        found: dict[str, np.ndarray] = {}
        wanted = list(dict.fromkeys(text_hashes))
        for i in range(0, len(wanted), LOOKUP_BATCH):
            batch = wanted[i:i + LOOKUP_BATCH]
            marks = ", ".join("?" * len(batch))
            query = ("SELECT p.text_hash, v.vector FROM passages p JOIN vectors v ON v.passage_id = p.passage_id "
                     f"WHERE p.text_hash IN ({marks}) ORDER BY p.id DESC")
            for text_hash, blob in self._db().execute(query, batch):
                found[text_hash] = np.frombuffer(blob, dtype=np.float32)
        return found

    def delete_source(self, source_id: str) -> bool:
        db = self._writable()
        db.execute("BEGIN IMMEDIATE")
        try:
            removed = db.execute("DELETE FROM sources WHERE source_id = ?", (source_id,)).rowcount
            db.execute("COMMIT")
        except BaseException:
            db.execute("ROLLBACK")
            raise
        return removed > 0

    def replace_source(self, source: SourceRow, items: list[tuple[PassageRow, np.ndarray]]) -> None:
        """Replace one source with its new passages and vectors, all or nothing."""
        db = self._writable()
        dim = int(_read_meta(db).get("dim", "0")) or (len(items[0][1]) if items and items[0][1].ndim == 1 else 0)
        seen: set[str] = set()
        for passage, vector in items:
            self._check_item(source, passage, vector, dim, seen)
            for other in db.execute("SELECT source_id FROM passages WHERE passage_id = ?", (passage.passage_id,)):
                if other[0] != source.source_id:
                    raise IndexStoreError(f"passage {passage.passage_id!r} is already in the index under another source")
        embedder_id = _read_meta(db)["embedder_id"]
        db.execute("BEGIN IMMEDIATE")
        try:
            db.execute("DELETE FROM sources WHERE source_id = ?", (source.source_id,))
            db.execute("INSERT INTO sources VALUES (?, ?, ?, ?, ?, ?, ?)",
                       (source.source_id, source.version, source.content_sha256, source.chunker_version, source.url,
                        source.licence, source.loaded_at))
            for passage, vector in items:
                db.execute("INSERT INTO passages (passage_id, source_id, start_pos, end_pos, heading_path, text, text_hash) "
                           "VALUES (?, ?, ?, ?, ?, ?, ?)",
                           (passage.passage_id, passage.source_id, passage.start, passage.end,
                            json.dumps(list(passage.heading_path)), passage.text, passage.text_hash))
                db.execute("INSERT INTO vectors VALUES (?, ?, ?, ?)",
                           (passage.passage_id, embedder_id, len(vector), vector.tobytes()))
            if items:
                db.execute("INSERT OR IGNORE INTO meta (key, value) VALUES ('dim', ?)", (str(dim),))
            db.execute("COMMIT")
        except BaseException as exc:
            db.execute("ROLLBACK")
            if isinstance(exc, sqlite3.Error):
                raise IndexStoreError(f"could not store {source.source_id!r}: {exc}") from exc
            raise

    @staticmethod
    def _check_item(source: SourceRow, passage: PassageRow, vector: np.ndarray, dim: int, seen: set[str]) -> None:
        name = repr(passage.passage_id)
        if passage.source_id != source.source_id:
            raise IndexStoreError(f"passage {name} belongs to {passage.source_id!r}, not {source.source_id!r}")
        if passage.passage_id in seen:
            raise IndexStoreError(f"passage {name} is given twice")
        seen.add(passage.passage_id)
        if not passage.text.strip():
            raise IndexStoreError(f"passage {name} has empty text")
        if not 0 <= passage.start < passage.end:
            raise IndexStoreError(f"passage {name} has a bad span ({passage.start}, {passage.end})")
        if passage.text_hash != content_sha256(passage.text):
            raise IndexStoreError(f"passage {name}: text_hash does not match the text")
        if vector.dtype != np.float32:
            raise IndexStoreError(f"the vector of {name} must be float32, not {vector.dtype}")
        if vector.ndim != 1:
            raise IndexStoreError(f"the vector of {name} must be one-dimensional")
        if len(vector) != dim:
            raise IndexStoreError(f"the vector of {name} has dimension {len(vector)}, the index uses {dim}")
        if not abs(float(np.linalg.norm(vector)) - 1.0) <= UNIT_TOLERANCE:
            raise IndexStoreError(f"the vector of {name} must have unit length")
