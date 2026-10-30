"""Unit 17d (spec 17 section 5): the sync, status and search commands behind `scripts/index.ps1`."""
import dataclasses
import hashlib
import re

import numpy as np
import pytest

from ragcore.chunking import ChunkConfig, chunk_source
from ragcore.contracts import Source
from ragcore.index_cli import Services, main
from ragcore.index_store import IndexSetup, open_index
from ragcore.rerank import FakeReranker

CONFIG = ChunkConfig(max_chars=70, hard_limit=400)
SETUP = IndexSetup(
    embedder_id="fake", embedder_revision="r1", chunker="heading-pack", chunker_version=CONFIG.version,
    chunk_config="70/400", llama_index_version="0.14.25", reranker_id="fake-reranker", reranker_revision="r1")
URLS = {"alpha": "https://alpha.example/", "beta": "https://beta.example/"}


def make_source(source_id: str, text: str) -> Source:
    return Source(source_id, "v1", text, hashlib.sha256(text.encode()).hexdigest(), "loader-1", "MIT", "declared")


def doc(topic: str, parts: int = 4) -> str:
    body = "\n\n".join(f"Paragraph {n} about {topic}. It says something specific to {topic} number {n}." for n in range(parts))
    return f"# {topic.title()}\n\n{body}\n"


SOURCES = [make_source("alpha/a.md", doc("apples")), make_source("alpha/b.md", doc("bananas")),
           make_source("beta/c.md", doc("cherries"))]


class FakeEmbedder:
    def __init__(self, calls=None):
        self.calls = calls if calls is not None else []

    def embed(self, texts, kind="passage"):
        self.calls.append(len(texts))
        out = []
        for text in texts:
            vector = np.array(list(hashlib.sha256(text.encode()).digest()[:8]), dtype=np.float32) + 1
            out.append(vector / np.linalg.norm(vector))
        return np.stack(out).astype(np.float32)


class BoundChunker:
    def __init__(self, fail_verify: bool = False):
        self.fail_verify, self.verified = fail_verify, False

    def __call__(self, source):
        return chunk_source(source, CONFIG)

    def verify(self):
        self.verified = True
        if self.fail_verify:
            raise RuntimeError("3 of 10 LlamaIndex nodes could not be found in the source text")


class Fakes:
    """Counts how often each model or loader is built, so tests can see what a command really needed."""

    def __init__(self, sources=None, notes=(), setup=SETUP, fail_verify=False):
        self.sources, self.notes, self.setup, self.fail_verify = list(SOURCES if sources is None else sources), list(notes), setup, fail_verify
        self.built = {"embedder": 0, "reranker": 0, "chunker": 0}
        self.chunker = None
        self.batches: list[int] = []

    def services(self) -> Services:
        def chunker(args):
            self.built["chunker"] += 1
            self.chunker = BoundChunker(self.fail_verify)
            return self.chunker

        def embedder(args):
            self.built["embedder"] += 1
            return FakeEmbedder(self.batches)

        def reranker(args):
            self.built["reranker"] += 1
            return FakeReranker()

        return Services(load_sources=lambda args: (list(self.sources), URLS, list(self.notes)),
                        setup=lambda args: self.setup, make_chunker=chunker, make_embedder=embedder,
                        make_reranker=reranker)


@pytest.fixture
def index(tmp_path):
    return str(tmp_path / "index.db")


def run(capsys, fakes, *argv, index_path):
    code = main([*argv, "--index", index_path, "--register", "register", "--snapshots", "snapshots"], fakes.services())
    captured = capsys.readouterr()
    return code, captured.out + captured.err


# --- sync ----------------------------------------------------------------------------------------------------------

def test_a_first_sync_fills_the_index_and_says_so(capsys, index):
    code, out = run(capsys, Fakes(), "sync", index_path=index)
    assert code == 0
    assert "3 sources added" in out and "Sync finished" in out


def test_a_second_sync_loads_no_model_and_no_chunker(capsys, index):
    run(capsys, Fakes(), "sync", index_path=index)
    again = Fakes()
    code, out = run(capsys, again, "sync", index_path=index)
    assert code == 0 and "3 sources unchanged" in out
    assert again.built == {"embedder": 0, "reranker": 0, "chunker": 0}


def test_the_chunker_is_checked_after_a_sync(capsys, index):
    fakes = Fakes()
    run(capsys, fakes, "sync", index_path=index)
    assert fakes.chunker.verified is True


def test_a_failed_chunker_check_is_reported_and_fails_the_command(capsys, index):
    code, out = run(capsys, Fakes(fail_verify=True), "sync", index_path=index)
    assert code == 1 and "could not be found" in out and "LlamaIndex" in out


def test_notes_from_loading_are_printed(capsys, index):
    code, out = run(capsys, Fakes(notes=["exclusions: 5 files left out of ruff"]), "sync", index_path=index)
    assert code == 0 and "exclusions: 5 files left out of ruff" in out


def test_a_dry_run_changes_nothing_and_does_not_create_the_file(capsys, tmp_path, index):
    fakes = Fakes()
    code, out = run(capsys, fakes, "sync", "--dry-run", index_path=index)
    assert code == 0 and "Dry run" in out and "3 sources added" in out
    assert not (tmp_path / "index.db").exists()
    assert fakes.built["embedder"] == 0


def test_a_dry_run_on_an_existing_index_changes_nothing(capsys, index):
    run(capsys, Fakes(), "sync", index_path=index)
    code, out = run(capsys, Fakes(sources=SOURCES[:1]), "sync", "--dry-run", "--allow-mass-delete", index_path=index)
    assert code == 0 and "2 removed" in out
    code, out = run(capsys, Fakes(), "status", index_path=index)
    assert "3 sources" in out


def test_removing_most_sources_needs_the_allow_flag(capsys, index):
    run(capsys, Fakes(), "sync", index_path=index)
    code, out = run(capsys, Fakes(sources=SOURCES[:1]), "sync", index_path=index)
    assert code == 1 and "allow_mass_delete" not in out and "--allow-mass-delete" in out
    code, out = run(capsys, Fakes(sources=SOURCES[:1]), "sync", "--allow-mass-delete", index_path=index)
    assert code == 0 and "2 removed" in out


def test_loading_no_sources_at_all_is_refused(capsys, index):
    code, out = run(capsys, Fakes(sources=[], notes=["ruff: not on disk"]), "sync", index_path=index)
    assert code == 2 and "no sources loaded" in out and "ruff: not on disk" in out


def test_a_different_setup_is_refused_with_a_plain_message(capsys, index):
    run(capsys, Fakes(), "sync", index_path=index)
    other = Fakes(setup=dataclasses.replace(SETUP, embedder_id="another"))
    code, out = run(capsys, other, "sync", index_path=index)
    assert code == 1 and "different setup" in out and "rebuild" in out


def test_progress_is_printed_for_the_last_source(capsys, index):
    code, out = run(capsys, Fakes(), "sync", index_path=index)
    assert "3 of 3 sources" in out


# --- status --------------------------------------------------------------------------------------------------------

def test_status_of_a_missing_index_says_to_run_sync(capsys, index):
    code, out = run(capsys, Fakes(), "status", index_path=index)
    assert code == 1 and "does not exist" in out and "sync" in out


def test_status_shows_counts_setup_and_last_sync(capsys, index):
    run(capsys, Fakes(), "sync", index_path=index)
    code, out = run(capsys, Fakes(), "status", index_path=index)
    assert code == 0
    assert "3 sources" in out and "passages" in out and "vectors" in out
    assert "embedder_id" in out and "fake" in out and "llama_index_version" in out and "0.14.25" in out
    assert "last sync" in out.lower() and "schema" in out.lower()


def test_status_loads_no_model(capsys, index):
    run(capsys, Fakes(), "sync", index_path=index)
    fakes = Fakes()
    run(capsys, fakes, "status", index_path=index)
    assert fakes.built == {"embedder": 0, "reranker": 0, "chunker": 0}


# --- search --------------------------------------------------------------------------------------------------------

def test_search_prints_rank_heading_link_and_start_of_the_text(capsys, index):
    run(capsys, Fakes(), "sync", index_path=index)
    code, out = run(capsys, Fakes(), "search", "cherries", "-k", "2", index_path=index)
    assert code == 0
    assert out.count("https://beta.example/") >= 1
    assert "Cherries" in out and "Paragraph" in out
    assert re.findall(r"(?m)^(\d+)\. ", out) == ["1", "2"]


def test_search_cuts_long_text_to_200_characters(capsys, index):
    long = make_source("alpha/long.md", "# Long\n\n" + "word " * 200 + "\n")
    run(capsys, Fakes(sources=[long]), "sync", index_path=index)
    code, out = run(capsys, Fakes(), "search", "word", "-k", "1", index_path=index)
    assert code == 0 and "word " * 40 in out and "word " * 41 not in out


def test_search_max_per_file_is_passed_on(capsys, index):
    run(capsys, Fakes(), "sync", index_path=index)
    code, out = run(capsys, Fakes(), "search", "paragraph about", "-k", "10", "--max-per-file", "1", index_path=index)
    assert code == 0 and out.count("https://") == 3


def test_search_without_a_question_is_a_usage_error(capsys, index):
    code, out = run(capsys, Fakes(), "search", index_path=index)
    assert code == 2 and "question" in out


def test_search_of_a_missing_index_says_to_run_sync(capsys, index):
    code, out = run(capsys, Fakes(), "search", "apples", index_path=index)
    assert code == 1 and "does not exist" in out and "sync" in out


def test_search_with_no_hits_says_so(capsys, index):
    run(capsys, Fakes(), "sync", index_path=index)
    code, out = run(capsys, Fakes(), "search", "   ", index_path=index)
    assert code == 0 and "No passages found" in out


# --- usage ---------------------------------------------------------------------------------------------------------

def test_sync_needs_the_register_and_snapshot_folders(capsys, index):
    code = main(["sync", "--index", index], Fakes().services())
    assert code == 2 and "--register" in capsys.readouterr().err


@pytest.mark.parametrize("flags", [["-k", "0"], ["--max-per-file", "0"], ["--batch", "0"]])
def test_limits_below_one_are_refused(capsys, index, flags):
    code, out = run(capsys, Fakes(), "search", "apples", *flags, index_path=index)
    assert code == 2 and "1 or more" in out


def test_search_builds_each_model_once(capsys, index):
    run(capsys, Fakes(), "sync", index_path=index)
    fakes = Fakes()
    run(capsys, fakes, "search", "apples", index_path=index)
    assert fakes.built == {"embedder": 1, "reranker": 1, "chunker": 0}


def test_the_setup_comes_from_the_services_not_from_the_file(capsys, index):
    run(capsys, Fakes(), "sync", index_path=index)
    other = Fakes(setup=dataclasses.replace(SETUP, reranker_revision="r2"))
    code, out = run(capsys, other, "search", "apples", index_path=index)
    assert code == 1 and "reranker_revision" in out


def test_the_batch_option_reaches_the_embedder(capsys, index):
    fakes = Fakes()
    run(capsys, fakes, "sync", "--batch", "2", index_path=index)
    assert len(fakes.batches) > 1 and max(fakes.batches) == 2


def test_status_works_while_a_sync_holds_the_index(capsys, index):
    run(capsys, Fakes(), "sync", index_path=index)
    with open_index(index, SETUP):  # a running sync holds the write lock
        code, out = run(capsys, Fakes(), "status", index_path=index)
    assert code == 0 and "3 sources" in out


def test_a_search_snippet_is_exactly_200_characters(capsys, index):
    long = make_source("alpha/long.md", "# Long\n\n" + "word " * 200 + "\n")
    run(capsys, Fakes(sources=[long]), "sync", index_path=index)
    code, out = run(capsys, Fakes(), "search", "word", "-k", "1", index_path=index)
    assert [len(line) for line in out.splitlines() if line.startswith("   word")] == [203]


def test_the_real_setup_reads_the_pins_from_the_config_files(capsys):
    pytest.importorskip("llama_index.core")
    import argparse
    from ragcore.index_cli import DEFAULT_EMBEDDER, DEFAULT_RERANKER, _parser, real_services

    args = argparse.Namespace(**vars(_parser().parse_args(["status"])))
    setup = real_services().setup(args)
    assert (setup.embedder_id, setup.reranker_id) == (DEFAULT_EMBEDDER, DEFAULT_RERANKER)
    assert setup.chunker == "llamaindex" and setup.chunker_version == "llamaindex-1/375/1500/6000"
    assert setup.chunk_config == "1500/6000" and setup.llama_index_version == "0.14.25"
    assert len(setup.embedder_revision.split("+")[0]) == 40 and "+" in setup.embedder_revision  # model + its code pin
    assert len(setup.reranker_revision) == 40
