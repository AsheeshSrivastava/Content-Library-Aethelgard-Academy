"""Unit 17b (spec 17 section 3): filling and updating the index from loaded sources."""
import dataclasses
import hashlib

import numpy as np
import pytest

from ragcore.chunking import ChunkConfig, chunk_source
from ragcore.contracts import Source
from ragcore.index_store import IndexSetup, IndexStoreError, open_index
from ragcore.index_sync import SyncError, SyncReport, sync

CONFIG = ChunkConfig(max_chars=70, hard_limit=400)
SETUP = IndexSetup(
    embedder_id="fake", embedder_revision="r1", chunker="heading-pack", chunker_version=CONFIG.version,
    chunk_config="70/400", llama_index_version="0.14.25", reranker_id="fake-reranker", reranker_revision="r1")
LOADED_AT = "2026-10-30T10:00:00+00:00"


def make_source(source_id: str, text: str, version: str = "v1") -> Source:
    return Source(source_id, version, text, hashlib.sha256(text.encode()).hexdigest(), "loader-1", "MIT", "declared")


def doc(topic: str, parts: int = 3) -> str:
    body = "\n\n".join(f"Paragraph {n} about {topic}. It says something specific to {topic} number {n}." for n in range(parts))
    return f"# {topic.title()}\n\n{body}\n"


def chunk(source: Source):
    return chunk_source(source, CONFIG)


class FakeEmbedder:
    def __init__(self, fail_on: str | None = None):
        self.calls: list[list[str]] = []
        self.kinds: list[str] = []
        self.fail_on = fail_on

    @property
    def texts(self) -> list[str]:
        return [t for call in self.calls for t in call]

    def embed(self, texts, kind="passage"):
        if self.fail_on and any(self.fail_on in t for t in texts):
            raise RuntimeError("embedder failed")
        self.calls.append(list(texts))
        self.kinds.append(kind)
        out = []
        for text in texts:
            digest = hashlib.sha256(text.encode()).digest()
            vector = np.array(list(digest[:8]), dtype=np.float32) + 1
            out.append(vector / np.linalg.norm(vector))
        return np.stack(out).astype(np.float32)


@pytest.fixture
def store(tmp_path):
    with open_index(tmp_path / "index.db", SETUP) as opened:
        yield opened


def run(store, sources, embedder=None, **options):
    embedder = embedder or FakeEmbedder()
    options.setdefault("loaded_at", LOADED_AT)
    return sync(sources, store, chunk, embedder, **options)


def three():
    return [make_source("alpha/a.md", doc("apples")), make_source("alpha/b.md", doc("bananas")),
            make_source("beta/c.md", doc("cherries"))]


def snapshot(store):
    rows, matrix = store.load_all()
    return [dataclasses.astuple(r) for r in rows], matrix.copy()


# --- filling an empty index --------------------------------------------------------------------------------------

def test_a_fresh_sync_stores_every_passage_as_an_exact_slice(store):
    sources = three()
    report = run(store, sources)
    expected = sum(len(chunk(s).passages) for s in sources)
    assert expected > 6
    assert report.sources_added == 3 and report.sources_replaced == 0 and report.sources_unchanged == 0
    assert report.passages_added == expected and report.passages_reused == 0
    assert store.counts() == {"sources": 3, "passages": expected, "vectors": expected}
    text_of = {s.source_id: s.text for s in sources}
    rows, _ = store.load_all()
    for row in rows:
        assert row.text == text_of[row.source_id][row.start:row.end]


def test_source_rows_keep_version_hash_licence_url_and_time(store):
    sources = three()
    run(store, sources, urls={"alpha": "https://alpha.example/", "beta": "https://beta.example/"})
    stored = store.sources()
    first = stored["alpha/a.md"]
    assert (first.version, first.content_sha256, first.licence) == ("v1", sources[0].text_sha256, "MIT")
    assert first.chunker_version == CONFIG.version and first.loaded_at == LOADED_AT
    assert first.url == "https://alpha.example/" and stored["beta/c.md"].url == "https://beta.example/"


def test_a_source_without_a_url_gets_an_empty_one(store):
    run(store, three(), urls={"alpha": "https://alpha.example/"})
    assert store.sources()["beta/c.md"].url == ""


def test_headings_are_kept_with_each_passage(store):
    run(store, [make_source("alpha/a.md", doc("apples"))])
    rows, _ = store.load_all()
    assert all(row.heading_path == ("Apples",) for row in rows)


def test_passages_are_embedded_as_plain_text_in_passage_mode(store):
    embedder = FakeEmbedder()
    run(store, three(), embedder)
    rows, _ = store.load_all()
    assert sorted(embedder.texts) == sorted({r.text for r in rows})
    assert set(embedder.kinds) == {"passage"}


def test_batches_are_no_larger_than_the_batch_size(store):
    embedder = FakeEmbedder()
    run(store, [make_source("alpha/big.md", doc("melons", parts=12))], embedder, batch=4)
    assert len(embedder.calls) > 1
    assert max(len(call) for call in embedder.calls) == 4


def test_the_same_text_is_embedded_once_across_and_inside_sources(store):
    twice = "# Same\n\nExactly this sentence appears in two files.\n\nExactly this sentence appears in two files.\n"
    embedder = FakeEmbedder()
    sources = [make_source("alpha/a.md", twice), make_source("beta/c.md", twice)]
    report = run(store, sources, embedder)
    assert embedder.texts.count("Exactly this sentence appears in two files.") == 1
    assert report.vectors_embedded == len(set(embedder.texts))
    assert report.passages_reused > 0
    assert store.counts()["vectors"] == store.counts()["passages"]


# --- second run, edits, removals ------------------------------------------------------------------------------------

def test_a_second_sync_changes_nothing_and_embeds_nothing(store):
    sources = three()
    run(store, sources)
    before = snapshot(store)
    embedder = FakeEmbedder()
    report = run(store, sources, embedder)
    assert report.sources_unchanged == 3 and report.sources_added == report.sources_replaced == 0
    assert report.passages_added == report.vectors_embedded == 0 and embedder.calls == []
    after = snapshot(store)
    assert after[0] == before[0] and np.array_equal(after[1], before[1])


def test_an_edit_replaces_only_that_source_and_embeds_only_new_text(store):
    sources = three()
    run(store, sources)
    edited = make_source("alpha/a.md", doc("apples") + "\nA brand new closing paragraph that was not there.\n", "v2")
    embedder = FakeEmbedder()
    report = run(store, [edited, sources[1], sources[2]], embedder)
    assert report.sources_replaced == 1 and report.sources_unchanged == 2
    assert embedder.texts == ["A brand new closing paragraph that was not there."]
    assert report.vectors_embedded == 1 and report.passages_reused == report.passages_added - 1
    assert store.sources()["alpha/a.md"].version == "v2"
    rows, _ = store.load_all()
    assert any(r.text == "A brand new closing paragraph that was not there." for r in rows)
    assert {r.version if hasattr(r, "version") else None for r in rows} == {None}


def test_a_new_version_with_the_same_text_costs_no_embedding(store):
    old = make_source("alpha/a.md", doc("apples"), "v1")
    run(store, [old])
    before_ids = {r.passage_id for r in store.load_all()[0]}
    embedder = FakeEmbedder()
    report = run(store, [make_source("alpha/a.md", doc("apples"), "v2")], embedder, allow_mass_delete=True)
    assert report.sources_replaced == 1 and embedder.calls == [] and report.vectors_embedded == 0
    after_ids = {r.passage_id for r in store.load_all()[0]}
    assert after_ids and after_ids.isdisjoint(before_ids)  # passage ids include the source version


def test_a_removed_source_disappears_with_its_passages(store):
    sources = three()
    run(store, sources)
    gone_passages = store.count_passages("beta/c.md")
    report = run(store, sources[:2])
    assert report.sources_deleted == 1 and report.passages_deleted == gone_passages
    assert "beta/c.md" not in store.sources() and store.count_passages("beta/c.md") == 0
    assert store.counts()["vectors"] == store.counts()["passages"]


def test_replaced_sources_count_their_old_passages_as_deleted(store):
    sources = three()
    run(store, sources)
    old_count = store.count_passages("alpha/a.md")
    edited = make_source("alpha/a.md", doc("apples", parts=1), "v2")
    report = run(store, [edited, sources[1], sources[2]])
    assert report.passages_deleted == old_count


def test_removing_most_of_the_index_is_refused_unless_allowed(store):
    sources = three()
    run(store, sources)
    before = snapshot(store)
    with pytest.raises(SyncError, match="allow_mass_delete"):
        run(store, sources[:1])
    after = snapshot(store)
    assert after[0] == before[0]
    report = run(store, sources[:1], allow_mass_delete=True)
    assert report.sources_deleted == 2 and list(store.sources()) == ["alpha/a.md"]


def test_removing_less_than_half_is_allowed(store):
    sources = three() + [make_source("beta/d.md", doc("dates"))]
    run(store, sources)
    assert run(store, sources[:3]).sources_deleted == 1


def test_an_empty_source_list_on_an_empty_index_is_fine(store):
    report = run(store, [])
    assert report == SyncReport(0, 0, 0, 0, 0, 0, 0, 0, report.seconds, False)


# --- power cuts, dry runs, mistakes ---------------------------------------------------------------------------------

def test_an_interrupted_sync_leaves_a_valid_index_and_the_rerun_equals_an_uninterrupted_one(store, tmp_path):
    sources = three()
    with pytest.raises(RuntimeError):
        run(store, sources, FakeEmbedder(fail_on="cherries"))
    assert list(store.sources()) == ["alpha/a.md", "alpha/b.md"]
    assert store.counts()["vectors"] == store.counts()["passages"] > 0
    run(store, sources)
    with open_index(tmp_path / "clean.db", SETUP) as clean:
        run(clean, sources)
        assert sorted(snapshot(store)[0]) == sorted(snapshot(clean)[0])
        order = lambda s: np.array(sorted(map(tuple, snapshot(s)[1].tolist())))  # noqa: E731
        assert np.array_equal(order(store), order(clean))


def test_a_dry_run_writes_nothing_and_embeds_nothing(store):
    sources = three()
    run(store, sources[:2])
    before = snapshot(store)
    embedder = FakeEmbedder()
    report = run(store, sources, embedder, dry_run=True)
    assert report.dry_run is True and embedder.calls == []
    assert report.sources_added == 1 and report.vectors_embedded > 0 and report.passages_added > 0
    after = snapshot(store)
    assert after[0] == before[0] and np.array_equal(after[1], before[1])
    assert len(store.sources()) == 2


def test_a_dry_run_matches_the_real_run(store, tmp_path):
    sources = three()
    run(store, sources[:2])
    planned = run(store, sources, dry_run=True)
    real = run(store, sources)
    assert dataclasses.replace(planned, seconds=0, dry_run=False) == dataclasses.replace(real, seconds=0)


def test_the_chunker_must_match_the_version_recorded_in_the_index(store):
    def other(source):
        return chunk_source(source, ChunkConfig(max_chars=80, hard_limit=400))

    with pytest.raises(SyncError, match="chunker"):
        sync(three(), store, other, FakeEmbedder(), loaded_at=LOADED_AT)
    assert store.counts()["sources"] == 0


def test_the_embedder_must_return_one_vector_per_text(store):
    class Short(FakeEmbedder):
        def embed(self, texts, kind="passage"):
            return super().embed(texts[:-1] or texts, kind)

    with pytest.raises(SyncError, match="vectors"):
        run(store, [make_source("alpha/a.md", doc("apples"))], Short())
    assert store.counts()["sources"] == 0


def test_vectors_that_are_not_unit_length_are_refused_by_the_store(store):
    class Loud(FakeEmbedder):
        def embed(self, texts, kind="passage"):
            return super().embed(texts, kind) * 3

    with pytest.raises(IndexStoreError, match="unit length"):
        run(store, [make_source("alpha/a.md", doc("apples"))], Loud())


def test_a_read_only_index_cannot_be_synced(tmp_path):
    with open_index(tmp_path / "i.db", SETUP):
        pass
    with open_index(tmp_path / "i.db", SETUP, read_only=True) as readonly:
        with pytest.raises(IndexStoreError, match="read-only"):
            run(readonly, three())


# --- progress and report ---------------------------------------------------------------------------------------------

def test_progress_is_reported_per_source_in_sorted_order(store):
    seen = []
    run(store, list(reversed(three())), progress=lambda done, total, source_id: seen.append((done, total, source_id)))
    assert seen == [(1, 3, "alpha/a.md"), (2, 3, "alpha/b.md"), (3, 3, "beta/c.md")]


def test_the_report_reads_in_plain_words(store):
    sources = three()
    text = run(store, sources).render()
    assert "3 sources added" in text and "0 sources unchanged" in text
    assert "embedded" in text
    assert "dry run" not in text.lower()
    planned = run(store, sources, dry_run=True).render()
    assert "dry run" in planned.lower() and "would need embedding" in planned and "texts embedded" not in planned
    assert "texts embedded" in text


def test_seconds_come_from_the_clock(store):
    ticks = iter([10.0, 14.5])
    report = run(store, three(), clock=lambda: next(ticks))
    assert report.seconds == 4.5


# --- added after the mutation check ------------------------------------------------------------------------------------

def test_removing_exactly_half_is_refused(store):
    sources = three() + [make_source("beta/d.md", doc("dates"))]
    run(store, sources)
    with pytest.raises(SyncError, match="allow_mass_delete"):
        run(store, sources[:2])


def test_a_changed_url_or_licence_replaces_the_source_without_embedding(store):
    sources = three()
    run(store, sources, urls={"alpha": "https://old.example/"})
    embedder = FakeEmbedder()
    assert run(store, sources, embedder, urls={"alpha": "https://new.example/"}).sources_replaced == 2
    assert store.sources()["alpha/a.md"].url == "https://new.example/" and embedder.calls == []
    relicensed = [dataclasses.replace(sources[0], licence="Apache-2.0"), *sources[1:]]
    assert run(store, relicensed, urls={"alpha": "https://new.example/"}).sources_replaced == 1
    assert store.sources()["alpha/a.md"].licence == "Apache-2.0"


def test_two_sources_with_the_same_id_are_refused(store):
    twin = [make_source("alpha/a.md", doc("apples")), make_source("alpha/a.md", doc("pears"))]
    with pytest.raises(SyncError, match="same source id"):
        run(store, twin)
    assert store.counts()["sources"] == 0
