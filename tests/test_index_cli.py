"""Unit 17d (spec 17 section 5): the sync, status and search commands behind `scripts/index.ps1`."""
import dataclasses
import hashlib
import re

import numpy as np
import pytest

from ragcore.chunking import ChunkConfig, chunk_source
from ragcore.contracts import Source
from ragcore.index_cli import Services, main
from ragcore.index_retriever import Retriever
from ragcore.index_store import IndexSetup, open_index
from ragcore.passage_eval import QuoteSpan
from ragcore.passage_questions import PassageQuestion
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


class FlakyReranker(FakeReranker):
    """Gives good scores for the first three searches and bad ones after, so a second run differs from the first."""

    def __init__(self):
        super().__init__()
        self.calls = 0

    def score(self, query, passages):
        self.calls += 1
        scores = super().score(query, passages)
        return [-s for s in scores] if self.calls > 3 else scores


class Fakes:
    """Counts how often each model or loader is built, so tests can see what a command really needed."""

    def __init__(self, sources=None, notes=(), setup=SETUP, fail_verify=False, flaky=False, questions=None):
        self.sources, self.notes, self.setup, self.fail_verify = list(SOURCES if sources is None else sources), list(notes), setup, fail_verify
        self.flaky, self.questions = flaky, questions
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
            return FlakyReranker() if self.flaky else FakeReranker()

        def questions(args):
            if self.questions is not None:
                return self.questions
            by_id = {s.source_id: s.text for s in SOURCES}
            items, spans = [], []
            for source_id, topic, n in (("alpha/a.md", "apples", 1), ("alpha/b.md", "bananas", 2), ("beta/c.md", "cherries", 3)):
                quote = f"Paragraph {n} about {topic}. It says something specific to {topic} number {n}."
                items.append(PassageQuestion(f"{topic} number {n}", source_id, quote, "fact"))
                start = by_id[source_id].index(quote)
                spans.append(QuoteSpan(source_id, start, start + len(quote)))
            return items, spans

        return Services(load_sources=lambda args: (list(self.sources), URLS, list(self.notes)),
                        setup=lambda args: self.setup, make_chunker=chunker, make_embedder=embedder,
                        make_reranker=reranker, load_questions=questions)


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


# --- check (17e) ---------------------------------------------------------------------------------------------------

def check(capsys, fakes, *extra, index_path):
    return run(capsys, fakes, "check", "--expect-first", "3", "--expect-top3", "3", *extra, index_path=index_path)


def test_check_scores_the_questions_through_the_search_and_says_they_match(capsys, index):
    run(capsys, Fakes(), "sync", index_path=index)
    code, out = check(capsys, Fakes(), index_path=index)
    assert code == 0
    assert "right first: 3 of 3" in out and "right in the first 3: 3 of 3" in out
    assert "same result on a second run: yes" in out and "matches" in out and "mean reciprocal rank" in out


def test_check_fails_and_says_so_when_the_numbers_differ(capsys, index):
    run(capsys, Fakes(), "sync", index_path=index)
    code, out = run(capsys, Fakes(), "check", "--expect-first", "2", "--expect-top3", "3", index_path=index)
    assert code == 1 and "differs" in out and "expected 2" in out


def test_check_fails_when_the_search_is_not_repeatable(capsys, index):
    run(capsys, Fakes(), "sync", index_path=index)
    code, out = check(capsys, Fakes(flaky=True), index_path=index)
    assert code == 1 and "same result on a second run: no" in out


def test_check_of_a_missing_index_says_to_run_sync(capsys, index):
    code, out = check(capsys, Fakes(), index_path=index)
    assert code == 1 and "does not exist" in out and "sync" in out


def test_check_loads_the_models_once_and_no_chunker(capsys, index):
    run(capsys, Fakes(), "sync", index_path=index)
    fakes = Fakes()
    check(capsys, fakes, index_path=index)
    assert fakes.built == {"embedder": 1, "reranker": 1, "chunker": 0}


def test_check_with_no_questions_is_refused(capsys, index):
    run(capsys, Fakes(), "sync", index_path=index)
    code, out = check(capsys, Fakes(questions=([], [])), index_path=index)
    assert code == 2 and "no questions" in out


def test_check_counts_a_wrong_first_answer_as_wrong(capsys, index):
    run(capsys, Fakes(), "sync", index_path=index)
    spans_wrong = Fakes().services().load_questions(None)
    questions, spans = spans_wrong
    swapped = [spans[1], spans[0], spans[2]]  # the first two questions now point at the other source
    code, out = run(capsys, Fakes(questions=(questions, swapped)), "check", "--expect-first", "1", "--expect-top3", "1",
                    index_path=index)
    assert "right first: 1 of 3" in out


def test_check_defaults_are_the_stage_4_numbers(capsys, index):
    run(capsys, Fakes(), "sync", index_path=index)
    code, out = run(capsys, Fakes(), "check", index_path=index)
    assert code == 1 and "expected 100 first and 121 in the first 3" in out


def test_check_counts_first_top_3_and_ten_separately_and_computes_the_reciprocal_rank(capsys, index):
    run(capsys, Fakes(), "sync", index_path=index)
    questions, spans = Fakes().services().load_questions(None)
    by_id = {s.source_id: s.text for s in SOURCES}
    # "apples number 1" has one passage with every word (paragraph 1) and three with two words: they come 2nd, 3rd, 4th
    items, others = [], []
    for n in (0, 2, 3):
        quote = f"Paragraph {n} about apples. It says something specific to apples number {n}."
        items.append(PassageQuestion("apples number 1", "alpha/a.md", quote, "fact"))
        start = by_id["alpha/a.md"].index(quote)
        others.append(QuoteSpan("alpha/a.md", start, start + len(quote)))
    with Retriever.open(index, SETUP, FakeEmbedder(), FakeReranker()) as retriever:  # ranks worked out without the scorer
        found = retriever.search("apples number 1", k=10)
    ranks = [next(i for i, x in enumerate(found, 1) if x.source_id == s.source_id and x.start <= s.start and s.end <= x.end)
             for s in others]
    first, top3, ten = (sum(r == 1 for r in ranks), sum(r <= 3 for r in ranks), sum(r <= 10 for r in ranks))
    assert len({first, top3, ten}) == 3  # the three counts differ, so a mix-up between them shows
    code, out = run(capsys, Fakes(questions=(items, others)), "check", "--expect-first", str(first),
                    "--expect-top3", str(top3), index_path=index)
    assert code == 0, out
    assert f"right first: {first} of 3" in out and f"right in the first 3: {top3} of 3" in out
    assert f"right within 10: {ten} of 3" in out
    assert f"mean reciprocal rank: {sum(1 / r for r in ranks) / 3:.3f}" in out


def test_check_needs_the_register_and_snapshot_folders(capsys, index):
    code = main(["check", "--index", index], Fakes().services())
    assert code == 2 and "--register" in capsys.readouterr().err
