"""Unit 17a (spec 17 section 2, ADR 0011): the SQLite index file."""
import dataclasses
import hashlib
import sqlite3

import numpy as np
import pytest

from ragcore import index_store
from ragcore.contracts import content_sha256
from ragcore.index_store import (SCHEMA_VERSION, IndexSetup, IndexStoreError, PassageRow, SourceRow, open_index)

SETUP = IndexSetup(
    embedder_id="jina-v2-base-en", embedder_revision="322d4d7e2f35", chunker="llamaindex",
    chunker_version="llamaindex-1/375/1500/6000", chunk_config="375/1500/6000", llama_index_version="0.14.25",
    reranker_id="bge-reranker-v2-m3", reranker_revision="953dc6f6f85a")
DIM = 4


def unit(*values: float) -> np.ndarray:
    vector = np.array(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


def source(source_id: str = "docs/a.md", content: str = "c1") -> SourceRow:
    return SourceRow(source_id, "v1", content_sha256(content), SETUP.chunker_version, f"https://example.org/{source_id}",
                     "MIT", "2026-10-30T10:00:00+00:00")


def passage(source_id: str, n: int, text: str | None = None) -> PassageRow:
    text = text if text is not None else f"passage {n} of {source_id}"
    return PassageRow(f"{source_id}#{n}", source_id, n * 100, n * 100 + len(text), ("Guide", f"Part {n}"), text,
                      content_sha256(text))


def rows(source_id: str, count: int = 2) -> list[tuple[PassageRow, np.ndarray]]:
    return [(passage(source_id, n), unit(1, n + 1, 0, 1)) for n in range(count)]


@pytest.fixture
def path(tmp_path):
    return tmp_path / "index" / "index.db"


@pytest.fixture
def store(path):
    with open_index(path, SETUP) as opened:
        yield opened


def file_hash(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --- creating and reopening -------------------------------------------------------------------------------------

def test_new_file_records_schema_and_setup(path):
    with open_index(path, SETUP) as store:
        meta = store.meta()
    assert meta["schema_version"] == str(SCHEMA_VERSION)
    for field in dataclasses.fields(SETUP):
        assert meta[field.name] == getattr(SETUP, field.name)
    assert "created_at" in meta


def test_connection_uses_write_ahead_mode_and_foreign_keys(store):
    assert store._conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert store._conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_reopen_with_same_setup_keeps_everything(path):
    with open_index(path, SETUP) as store:
        created = store.meta()["created_at"]
        store.replace_source(source(), rows("docs/a.md"))
    with open_index(path, SETUP) as store:
        assert store.meta()["created_at"] == created
        assert store.counts() == {"sources": 1, "passages": 2, "vectors": 2}


def test_missing_file_is_not_created_when_create_is_false(path):
    with pytest.raises(IndexStoreError, match="does not exist"):
        open_index(path, SETUP, create=False)
    assert not path.exists()


# --- one pinned setup per file ----------------------------------------------------------------------------------

@pytest.mark.parametrize("field", [f.name for f in dataclasses.fields(SETUP)])
def test_a_different_setup_is_refused_and_the_file_is_unchanged(path, field):
    with open_index(path, SETUP) as store:
        store.replace_source(source(), rows("docs/a.md"))
    before = file_hash(path)
    other = dataclasses.replace(SETUP, **{field: "something-else"})
    with pytest.raises(IndexStoreError, match=field) as error:
        open_index(path, other)
    assert "rebuild" in str(error.value)
    assert file_hash(path) == before


def test_read_only_open_also_checks_the_setup(path):
    with open_index(path, SETUP):
        pass
    other = dataclasses.replace(SETUP, embedder_revision="other")
    with pytest.raises(IndexStoreError, match="embedder_revision"):
        open_index(path, other, read_only=True)


# --- schema version -----------------------------------------------------------------------------------------------

def test_a_newer_schema_is_refused(path):
    with open_index(path, SETUP):
        pass
    raw = sqlite3.connect(path)
    raw.execute("UPDATE meta SET value = '99' WHERE key = 'schema_version'")
    raw.commit()
    raw.close()
    with pytest.raises(IndexStoreError, match="newer"):
        open_index(path, SETUP)


def test_an_older_schema_is_backed_up_then_upgraded(path, monkeypatch):
    with open_index(path, SETUP) as store:
        store.replace_source(source(), rows("docs/a.md"))
    calls = []

    def upgrade(conn):
        calls.append(1)
        conn.execute("CREATE TABLE added_in_two (x INTEGER)")

    monkeypatch.setattr(index_store, "SCHEMA_VERSION", 2)
    monkeypatch.setattr(index_store, "_MIGRATIONS", {1: upgrade})
    with open_index(path, SETUP) as store:
        assert store.meta()["schema_version"] == "2"
        assert store.counts()["passages"] == 2
    backup = path.with_name("index.db.bak-1")
    assert backup.exists()
    old = sqlite3.connect(backup)
    assert old.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()[0] == "1"
    assert old.execute("SELECT name FROM sqlite_master WHERE name = 'added_in_two'").fetchone() is None
    old.close()
    with open_index(path, SETUP):
        pass
    assert calls == [1]


def test_a_failed_upgrade_leaves_the_old_version_and_the_backup(path, monkeypatch):
    with open_index(path, SETUP):
        pass

    def broken(conn):
        conn.execute("CREATE TABLE half_done (x INTEGER)")
        raise RuntimeError("boom")

    monkeypatch.setattr(index_store, "SCHEMA_VERSION", 2)
    monkeypatch.setattr(index_store, "_MIGRATIONS", {1: broken})
    with pytest.raises(IndexStoreError, match="upgrade"):
        open_index(path, SETUP)
    raw = sqlite3.connect(path)
    assert raw.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()[0] == "1"
    assert raw.execute("SELECT name FROM sqlite_master WHERE name = 'half_done'").fetchone() is None
    raw.close()
    assert path.with_name("index.db.bak-1").exists()


# --- writing and reading sources ----------------------------------------------------------------------------------

def test_replace_source_stores_source_passages_and_vectors(store):
    given = rows("docs/a.md")
    store.replace_source(source(), given)
    assert store.sources() == {"docs/a.md": source()}
    assert store.counts() == {"sources": 1, "passages": 2, "vectors": 2}
    loaded_rows, matrix = store.load_all()
    assert loaded_rows == [p for p, _ in given]
    assert matrix.dtype == np.float32 and matrix.shape == (2, DIM)
    assert np.array_equal(matrix, np.stack([v for _, v in given]))


def test_heading_path_round_trips_as_a_tuple(store):
    store.replace_source(source(), rows("docs/a.md"))
    loaded_rows, _ = store.load_all()
    assert loaded_rows[0].heading_path == ("Guide", "Part 0")
    assert isinstance(loaded_rows[0].heading_path, tuple)


def test_rows_come_back_in_the_order_they_were_added(store):
    store.replace_source(source("docs/a.md"), rows("docs/a.md", 3))
    store.replace_source(source("docs/b.md"), rows("docs/b.md", 2))
    loaded_rows, _ = store.load_all()
    assert [p.passage_id for p in loaded_rows] == [
        "docs/a.md#0", "docs/a.md#1", "docs/a.md#2", "docs/b.md#0", "docs/b.md#1"]


def test_replace_source_replaces_only_that_source(store):
    store.replace_source(source("docs/a.md"), rows("docs/a.md", 3))
    store.replace_source(source("docs/b.md"), rows("docs/b.md", 2))
    newer = SourceRow("docs/a.md", "v2", content_sha256("c2"), SETUP.chunker_version, "https://example.org/a2", "MIT",
                      "2026-10-31T10:00:00+00:00")
    store.replace_source(newer, [(passage("docs/a.md", 7, "only one new passage"), unit(0, 0, 1, 1))])
    assert store.sources()["docs/a.md"] == newer
    loaded_rows, matrix = store.load_all()
    assert sorted(p.passage_id for p in loaded_rows) == ["docs/a.md#7", "docs/b.md#0", "docs/b.md#1"]
    assert store.counts() == {"sources": 2, "passages": 3, "vectors": 3}
    assert matrix.shape == (3, DIM)


def test_a_failed_replace_changes_nothing(store):
    store.replace_source(source(), rows("docs/a.md"))
    bad = [(passage("docs/a.md", 5), unit(1, 2, 3, 4)), (passage("docs/a.md", 6), np.zeros(DIM, dtype=np.float32))]
    with pytest.raises(IndexStoreError):
        store.replace_source(source(content="c9"), bad)
    assert store.sources() == {"docs/a.md": source()}
    assert [p.passage_id for p in store.load_all()[0]] == ["docs/a.md#0", "docs/a.md#1"]


def test_empty_passage_list_is_allowed_and_leaves_the_source_with_no_passages(store):
    store.replace_source(source(), rows("docs/a.md"))
    store.replace_source(source(content="c2"), [])
    assert store.counts() == {"sources": 1, "passages": 0, "vectors": 0}


def test_delete_source_removes_its_passages_and_vectors(store):
    store.replace_source(source("docs/a.md"), rows("docs/a.md"))
    store.replace_source(source("docs/b.md"), rows("docs/b.md"))
    assert store.delete_source("docs/a.md") is True
    assert list(store.sources()) == ["docs/b.md"]
    assert store.counts() == {"sources": 1, "passages": 2, "vectors": 2}
    assert store.delete_source("docs/a.md") is False


def test_find_vectors_by_text_hash(store):
    store.replace_source(source(), rows("docs/a.md"))
    wanted = [content_sha256("passage 0 of docs/a.md"), content_sha256("never stored")]
    found = store.find_vectors(wanted)
    assert list(found) == [wanted[0]]
    assert np.array_equal(found[wanted[0]], unit(1, 1, 0, 1))
    assert store.find_vectors([]) == {}


def test_empty_index_loads_as_nothing(store):
    loaded_rows, matrix = store.load_all()
    assert loaded_rows == [] and matrix.shape[0] == 0
    assert store.counts() == {"sources": 0, "passages": 0, "vectors": 0}


# --- checks on what goes in ---------------------------------------------------------------------------------------

def test_vector_must_be_float32(store):
    with pytest.raises(IndexStoreError, match="float32"):
        store.replace_source(source(), [(passage("docs/a.md", 0), unit(1, 2, 3, 4).astype(np.float64))])


def test_vector_must_be_one_dimensional(store):
    with pytest.raises(IndexStoreError, match="one-dimensional"):
        store.replace_source(source(), [(passage("docs/a.md", 0), unit(1, 2, 3, 4).reshape(2, 2))])


def test_vector_must_have_unit_length(store):
    with pytest.raises(IndexStoreError, match="unit length"):
        store.replace_source(source(), [(passage("docs/a.md", 0), np.array([1, 2, 3, 4], dtype=np.float32))])


def test_vector_length_close_to_one_is_accepted(store):
    store.replace_source(source(), [(passage("docs/a.md", 0), unit(1, 2, 3, 4) * np.float32(1.0005))])
    assert store.counts()["vectors"] == 1


def test_all_vectors_must_have_the_same_dimension(store):
    store.replace_source(source("docs/a.md"), rows("docs/a.md"))
    with pytest.raises(IndexStoreError, match="dimension"):
        store.replace_source(source("docs/b.md"), [(passage("docs/b.md", 0), unit(1, 2, 3))])
    with pytest.raises(IndexStoreError, match="dimension"):
        store.replace_source(source("docs/c.md"), [(passage("docs/c.md", 0), unit(1, 2, 3, 4)),
                                                   (passage("docs/c.md", 1), unit(1, 2, 3))])


def test_dimension_is_kept_in_meta_and_survives_reopen(path):
    with open_index(path, SETUP) as store:
        store.replace_source(source(), rows("docs/a.md"))
        assert store.meta()["dim"] == str(DIM)
    with open_index(path, SETUP) as store:
        with pytest.raises(IndexStoreError, match="dimension"):
            store.replace_source(source("docs/b.md"), [(passage("docs/b.md", 0), unit(1, 2, 3))])


def test_text_hash_must_match_the_text(store):
    bad = dataclasses.replace(passage("docs/a.md", 0), text_hash=content_sha256("something else"))
    with pytest.raises(IndexStoreError, match="text_hash"):
        store.replace_source(source(), [(bad, unit(1, 2, 3, 4))])


def test_passage_must_have_text_and_a_forward_span(store):
    empty = dataclasses.replace(passage("docs/a.md", 0), text="", text_hash=content_sha256(""))
    with pytest.raises(IndexStoreError, match="empty"):
        store.replace_source(source(), [(empty, unit(1, 2, 3, 4))])
    backwards = dataclasses.replace(passage("docs/a.md", 0), start=50, end=50)
    with pytest.raises(IndexStoreError, match="span"):
        store.replace_source(source(), [(backwards, unit(1, 2, 3, 4))])
    negative = dataclasses.replace(passage("docs/a.md", 0), start=-1)
    with pytest.raises(IndexStoreError, match="span"):
        store.replace_source(source(), [(negative, unit(1, 2, 3, 4))])


def test_passage_must_belong_to_the_source_being_replaced(store):
    with pytest.raises(IndexStoreError, match="belongs to"):
        store.replace_source(source("docs/a.md"), [(passage("docs/b.md", 0), unit(1, 2, 3, 4))])


def test_passage_ids_must_be_unique_inside_the_call_and_in_the_index(store):
    twice = [(passage("docs/a.md", 0), unit(1, 2, 3, 4)), (passage("docs/a.md", 0), unit(1, 2, 3, 4))]
    with pytest.raises(IndexStoreError, match="twice"):
        store.replace_source(source("docs/a.md"), twice)
    store.replace_source(source("docs/a.md"), rows("docs/a.md"))
    clash = dataclasses.replace(passage("docs/b.md", 0), passage_id="docs/a.md#0")
    with pytest.raises(IndexStoreError, match="already"):
        store.replace_source(source("docs/b.md"), [(clash, unit(1, 2, 3, 4))])
    assert store.counts()["sources"] == 1


# --- one writer, many readers, read-only retriever ------------------------------------------------------------

def test_a_second_writer_is_refused_until_the_first_closes(path):
    first = open_index(path, SETUP)
    with pytest.raises(IndexStoreError, match="another"):
        open_index(path, SETUP)
    first.close()
    open_index(path, SETUP).close()


def test_the_lock_is_released_when_the_with_block_ends_with_an_error(path):
    with pytest.raises(ValueError):
        with open_index(path, SETUP):
            raise ValueError("fail")
    open_index(path, SETUP).close()


def test_read_only_open_works_while_a_writer_is_open_and_sees_only_committed_data(path):
    writer = open_index(path, SETUP)
    writer.replace_source(source("docs/a.md"), rows("docs/a.md"))
    reader = open_index(path, SETUP, read_only=True)
    assert reader.counts()["passages"] == 2
    writer._conn.execute("BEGIN IMMEDIATE")
    writer._conn.execute("DELETE FROM sources")
    assert reader.counts()["passages"] == 2  # not blocked, and the open change is not visible
    writer._conn.execute("ROLLBACK")
    reader.close()
    writer.close()


def test_read_only_connection_cannot_write(path):
    with open_index(path, SETUP) as store:
        store.replace_source(source(), rows("docs/a.md"))
    with open_index(path, SETUP, read_only=True) as reader:
        with pytest.raises(sqlite3.OperationalError):
            reader._conn.execute("DELETE FROM sources")
        with pytest.raises(IndexStoreError, match="read-only"):
            reader.replace_source(source(), rows("docs/a.md"))
        with pytest.raises(IndexStoreError, match="read-only"):
            reader.delete_source("docs/a.md")
        assert reader.counts()["passages"] == 2
        assert reader.load_all()[1].shape == (2, DIM)


def test_read_only_open_of_a_missing_file_is_refused(path):
    with pytest.raises(IndexStoreError, match="does not exist"):
        open_index(path, SETUP, read_only=True)


def test_read_only_open_does_not_take_the_write_lock(path):
    with open_index(path, SETUP):
        pass
    first = open_index(path, SETUP, read_only=True)
    second = open_index(path, SETUP, read_only=True)
    open_index(path, SETUP).close()
    first.close()
    second.close()


def test_a_closed_store_refuses_use(path):
    store = open_index(path, SETUP)
    store.close()
    with pytest.raises(IndexStoreError, match="closed"):
        store.counts()
    store.close()  # closing twice is fine


# --- gaps found by the mutation check ---------------------------------------------------------------------------

def test_a_vector_that_is_one_percent_too_long_is_refused(store):
    with pytest.raises(IndexStoreError, match="unit length"):
        store.replace_source(source(), [(passage("docs/a.md", 0), unit(1, 2, 3, 4) * np.float32(1.01))])


def test_text_of_only_spaces_is_empty(store):
    blank = dataclasses.replace(passage("docs/a.md", 0), text="   ", text_hash=content_sha256("   "))
    with pytest.raises(IndexStoreError, match="empty"):
        store.replace_source(source(), [(blank, unit(1, 2, 3, 4))])


def test_the_same_source_can_be_replaced_with_the_same_passage_ids(store):
    store.replace_source(source(), rows("docs/a.md"))
    store.replace_source(source(content="c2"), rows("docs/a.md"))
    assert store.counts() == {"sources": 1, "passages": 2, "vectors": 2}


def test_find_vectors_gives_the_first_added_when_a_text_is_stored_twice(store):
    same = "the very same sentence in two sources"
    first, second = unit(1, 0, 0, 1), unit(0, 1, 1, 0)
    store.replace_source(source("docs/a.md"), [(passage("docs/a.md", 0, same), first)])
    store.replace_source(source("docs/b.md"), [(passage("docs/b.md", 0, same), second)])
    found = store.find_vectors([content_sha256(same)])
    assert np.array_equal(found[content_sha256(same)], first)


def test_a_passage_without_a_vector_is_reported_as_damage(store):
    store.replace_source(source(), rows("docs/a.md"))
    store._conn.execute("DELETE FROM vectors WHERE passage_id = 'docs/a.md#0'")
    with pytest.raises(IndexStoreError, match="damaged"):
        store.load_all()


def test_an_older_schema_cannot_be_opened_read_only(path, monkeypatch):
    with open_index(path, SETUP):
        pass
    monkeypatch.setattr(index_store, "SCHEMA_VERSION", 2)
    monkeypatch.setattr(index_store, "_MIGRATIONS", {1: lambda conn: None})
    with pytest.raises(IndexStoreError, match="older schema"):
        open_index(path, SETUP, read_only=True)


def test_some_other_database_file_is_not_an_index(path):
    path.parent.mkdir(parents=True)
    other = sqlite3.connect(path)
    other.execute("CREATE TABLE notes (x TEXT)")
    other.commit()
    other.close()
    with pytest.raises(IndexStoreError, match="not an index"):
        open_index(path, SETUP)
    with pytest.raises(IndexStoreError, match="not an index"):
        open_index(path, SETUP, read_only=True)


def test_an_empty_database_file_is_not_an_index_when_read_only(path):
    path.parent.mkdir(parents=True)
    empty = sqlite3.connect(path)
    empty.execute("PRAGMA user_version = 1")
    empty.close()
    with pytest.raises(IndexStoreError, match="not an index"):
        open_index(path, SETUP, read_only=True)

