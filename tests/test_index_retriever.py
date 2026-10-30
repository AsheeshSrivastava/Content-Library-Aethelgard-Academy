"""Unit 17c (spec 17 section 4): the retriever adapter over the index file."""
import dataclasses
import hashlib
import sqlite3

import numpy as np
import pytest

from ragcore.chunking import ChunkConfig, chunk_source
from ragcore.contracts import Source
from ragcore.index_retriever import Result, Retriever
from ragcore.index_store import IndexSetup, IndexStoreError, open_index
from ragcore.index_sync import sync
from ragcore.rerank import FakeReranker
from ragcore.retrieve import make_passage_pipeline
from ragcore.runs import Run

CONFIG = ChunkConfig(max_chars=70, hard_limit=400)
SETUP = IndexSetup(
    embedder_id="fake", embedder_revision="r1", chunker="heading-pack", chunker_version=CONFIG.version,
    chunk_config="70/400", llama_index_version="0.14.25", reranker_id="fake-reranker", reranker_revision="r1")
URLS = {"alpha": "https://alpha.example/", "beta": "https://beta.example/"}


def make_source(source_id: str, text: str, licence: str = "MIT") -> Source:
    return Source(source_id, "v1", text, hashlib.sha256(text.encode()).hexdigest(), "loader-1", licence, "declared")


def doc(topic: str, parts: int = 4) -> str:
    body = "\n\n".join(f"Paragraph {n} about {topic}. It says something specific to {topic} number {n}." for n in range(parts))
    return f"# {topic.title()}\n\n{body}\n"


SOURCES = [make_source("alpha/a.md", doc("apples")), make_source("alpha/b.md", doc("bananas")),
           make_source("beta/c.md", doc("cherries"), "Apache-2.0")]


class FakeEmbedder:
    def __init__(self):
        self.calls: list[tuple[list[str], str]] = []

    def embed(self, texts, kind="passage"):
        self.calls.append((list(texts), kind))
        out = []
        for text in texts:
            vector = np.array(list(hashlib.sha256(text.encode()).digest()[:8]), dtype=np.float32) + 1
            out.append(vector / np.linalg.norm(vector))
        return np.stack(out).astype(np.float32)


def fill(path, sources=SOURCES):
    with open_index(path, SETUP) as store:
        sync(sources, store, lambda s: chunk_source(s, CONFIG), FakeEmbedder(), urls=URLS)


@pytest.fixture
def path(tmp_path):
    target = tmp_path / "index.db"
    fill(target)
    return target


@pytest.fixture
def retriever(path):
    embedder = FakeEmbedder()
    with Retriever.open(path, SETUP, embedder, FakeReranker()) as opened:
        opened.embedder = embedder
        yield opened


# --- answers -------------------------------------------------------------------------------------------------------

def test_results_carry_everything_a_caller_needs_to_cite(retriever):
    results = retriever.search("apples paragraph", k=5)
    assert 0 < len(results) <= 5 and all(isinstance(r, Result) for r in results)
    assert [r.rank for r in results] == list(range(1, len(results) + 1))
    text_of = {s.source_id: s.text for s in SOURCES}
    licence_of = {s.source_id: s.licence for s in SOURCES}
    for r in results:
        assert r.text == text_of[r.source_id][r.start:r.end]
        assert r.licence == licence_of[r.source_id]
        assert r.url == URLS[r.source_id.split("/")[0]]
        assert isinstance(r.heading_path, tuple) and r.heading_path
        assert r.passage_id.startswith(r.source_id)


def test_the_best_passage_for_a_topic_comes_from_that_topic(retriever):
    assert retriever.search("cherries", k=1)[0].source_id == "beta/c.md"
    assert retriever.search("bananas", k=1)[0].source_id == "alpha/b.md"


def test_scores_are_the_rerankers_and_come_best_first(retriever):
    results = retriever.search("apples number 2", k=5)
    reranker = FakeReranker()
    for r in results:
        assert r.score == reranker.score("apples number 2", [r.text])[0]
    assert [r.score for r in results] == sorted((r.score for r in results), reverse=True)


def test_the_order_equals_the_stage_3_hr_pipeline(path):
    embedder, reranker = FakeEmbedder(), FakeReranker()
    with open_index(path, SETUP, read_only=True) as store:
        passages, matrix = store.load_all()
    rows = [{"passage_id": p.passage_id, "source_id": p.source_id, "start": p.start, "end": p.end,
             "headings": list(p.heading_path), "parent": None} for p in passages]
    run = Run("x", {}, rows, matrix)
    pipeline = make_passage_pipeline("HR", run, [p.text for p in passages], embedder=embedder, reranker=reranker)
    with Retriever.open(path, SETUP, embedder, reranker) as retriever:
        for query in ("apples", "number 3 specific", "paragraph about bananas", "zebra"):
            expected = [(h.source_id, h.start, h.end, h.score) for h in pipeline(query, 8)]
            got = [(r.source_id, r.start, r.end, r.score) for r in retriever.search(query, k=8)]
            assert got == expected, query


def test_the_same_query_gives_the_same_list(retriever):
    assert retriever.search("paragraph", k=6) == retriever.search("paragraph", k=6)


def test_queries_are_embedded_in_query_mode(retriever):
    retriever.search("apples")
    assert retriever.embedder.calls == [(["apples"], "query")]


def test_k_limits_the_list_and_depth_limits_how_far_the_reranker_looks(path):
    with Retriever.open(path, SETUP, FakeEmbedder(), FakeReranker(), depth=3) as shallow:
        assert len(shallow.search("paragraph", k=10)) == 3
        assert len(shallow.search("paragraph", k=2)) == 2


# --- one source at most N times -------------------------------------------------------------------------------------

def test_max_per_file_limits_each_source_and_keeps_the_order(path):
    with Retriever.open(path, SETUP, FakeEmbedder(), FakeReranker()) as retriever:
        plain = retriever.search("paragraph about", k=30)
        limited = retriever.search("paragraph about", k=30, max_per_file=1)
    assert len({r.source_id for r in limited}) == len(limited) == 3
    assert [r.rank for r in limited] == [1, 2, 3]
    first_of = {}
    for r in plain:
        first_of.setdefault(r.source_id, r.passage_id)
    assert [r.passage_id for r in limited] == list(first_of.values())


def test_max_per_file_of_two_allows_two(retriever):
    results = retriever.search("paragraph about", k=30, max_per_file=2)
    counts = {}
    for r in results:
        counts[r.source_id] = counts.get(r.source_id, 0) + 1
    assert max(counts.values()) == 2


# --- nothing to answer, wrong calls ---------------------------------------------------------------------------------

@pytest.mark.parametrize("query", ["", "   ", "\n\t"])
def test_an_empty_query_returns_nothing_and_embeds_nothing(retriever, query):
    assert retriever.search(query) == []
    assert retriever.embedder.calls == []


def test_an_empty_index_returns_nothing(tmp_path):
    empty = tmp_path / "empty.db"
    fill(empty, [])
    with Retriever.open(empty, SETUP, FakeEmbedder(), FakeReranker()) as retriever:
        assert retriever.search("anything") == []


@pytest.mark.parametrize("k, per_file", [(0, None), (-1, None), (3, 0), (3, -2)])
def test_bad_limits_are_refused(retriever, k, per_file):
    with pytest.raises(ValueError):
        retriever.search("apples", k=k, max_per_file=per_file)


def test_a_missing_file_is_refused_and_not_created(tmp_path):
    with pytest.raises(IndexStoreError, match="does not exist"):
        Retriever.open(tmp_path / "none.db", SETUP, FakeEmbedder(), FakeReranker())
    assert not (tmp_path / "none.db").exists()


def test_a_different_setup_is_refused(path):
    with pytest.raises(IndexStoreError, match="different setup"):
        Retriever.open(path, dataclasses.replace(SETUP, embedder_id="other"), FakeEmbedder(), FakeReranker())


def test_a_closed_retriever_refuses_searches(path):
    retriever = Retriever.open(path, SETUP, FakeEmbedder(), FakeReranker())
    retriever.close()
    with pytest.raises(IndexStoreError, match="closed"):
        retriever.search("apples")


# --- read-only, and alongside a sync --------------------------------------------------------------------------------

def test_the_retriever_cannot_write_to_the_index(retriever):
    with pytest.raises(IndexStoreError, match="read-only"):
        retriever.store.delete_source("alpha/a.md")
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        retriever.store._conn.execute("DELETE FROM sources")
    assert retriever.store.counts()["sources"] == 3


def test_searching_works_while_a_sync_holds_a_write_transaction(path):
    with open_index(path, SETUP) as writer:
        writer._conn.execute("BEGIN IMMEDIATE")
        writer._conn.execute("DELETE FROM sources WHERE source_id = 'alpha/a.md'")
        try:
            with Retriever.open(path, SETUP, FakeEmbedder(), FakeReranker()) as retriever:
                assert retriever.search("apples", k=3)[0].source_id == "alpha/a.md"
        finally:
            writer._conn.execute("ROLLBACK")


def test_a_retriever_opened_after_a_sync_sees_the_new_passages(path):
    extra = SOURCES + [make_source("beta/d.md", doc("dates"))]
    fill(path, extra)
    with Retriever.open(path, SETUP, FakeEmbedder(), FakeReranker()) as retriever:
        assert retriever.search("dates", k=1)[0].source_id == "beta/d.md"


# --- added after the mutation check ------------------------------------------------------------------------------------

def test_n_raw_limits_how_many_candidates_each_search_list_gives(path):
    with Retriever.open(path, SETUP, FakeEmbedder(), FakeReranker(), n_raw=1) as retriever:
        assert 1 <= len(retriever.search("paragraph", k=10)) <= 2


def test_a_writable_index_is_not_accepted(path):
    with open_index(path, SETUP) as writable:
        with pytest.raises(IndexStoreError, match="read-only"):
            Retriever(writable, FakeEmbedder(), FakeReranker())


def test_closing_the_retriever_closes_the_index_file(path):
    retriever = Retriever.open(path, SETUP, FakeEmbedder(), FakeReranker())
    retriever.close()
    with pytest.raises(IndexStoreError, match="closed"):
        retriever.store.counts()
