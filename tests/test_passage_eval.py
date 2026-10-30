"""Passage-level pipelines and scoring (spec 16 section 5)."""
import math

import numpy as np
import pytest

from ragcore.compare import QResult
from ragcore.passage_eval import QuoteSpan, cut_flags, holds, quote_span, score_passage_questions
from ragcore.passage_questions import PassageQuestion
from ragcore.rerank import FakeReranker
from ragcore.retrieve import PIPELINES, PassageHit, make_passage_pipeline, make_pipeline
from ragcore.runs import Run

QUOTE = "The retry decorator runs the wrapped function again when it raises a timeout error."
SOURCES = {"lib/guide.md": f"# Guide\n\n{QUOTE}\n\nMore text.\n", "lib/other.md": "# Other\n\nNothing to see here, move on please.\n"}


def question(q="What happens after a call fails?", source="guide", quote=QUOTE):
    return PassageQuestion(q, source, quote, "fact")


# --- the quote's place -----------------------------------------------------------------------------

def test_quote_span_is_the_place_of_the_quote():
    span = quote_span(question(), SOURCES)
    assert span == QuoteSpan("lib/guide.md", 9, 9 + len(QUOTE))
    assert SOURCES["lib/guide.md"][span.start:span.end] == QUOTE


@pytest.mark.parametrize("q, message", [
    (question(source="nope"), "no source"),
    (question(quote="this exact sentence is certainly not in the guide text"), "not found"),
])
def test_quote_span_refuses_what_the_checks_refuse(q, message):
    with pytest.raises(ValueError, match=message):
        quote_span(q, SOURCES)


def test_quote_span_refuses_a_quote_that_is_there_twice():
    twice = {"lib/guide.md": f"{QUOTE}\n\n{QUOTE}\n"}
    with pytest.raises(ValueError, match="2 times"):
        quote_span(question(), twice)


def test_an_exact_source_id_wins_over_longer_ids_that_contain_it():
    sources = {"lib/guide.md.old": "# Old\n\nSomething else entirely, with no quote in it at all.\n", **SOURCES}
    assert quote_span(question(source="lib/guide.md"), sources).source_id == "lib/guide.md"

def test_a_passage_holds_a_quote_only_when_it_covers_all_of_it():
    span = QuoteSpan("a", 100, 200)
    assert holds(PassageHit("a", 100, 200, 0.0), span)
    assert holds(PassageHit("a", 0, 1000, 0.0), span)
    assert not holds(PassageHit("a", 101, 300, 0.0), span)
    assert not holds(PassageHit("a", 0, 199, 0.0), span)
    assert not holds(PassageHit("b", 0, 1000, 0.0), span)


# --- scoring ---------------------------------------------------------------------------------------

def fixed(*hits_per_query):
    answers = iter(hits_per_query)
    return lambda query, n: list(next(answers))[:n]


def hit(source, start, end):
    return PassageHit(source, start, end, 1.0)


SPAN = QuoteSpan("a", 100, 200)
RIGHT, WRONG = hit("a", 90, 210), hit("a", 300, 400)


def test_first_top_three_and_rank():
    qs = [question(q=f"q{i}") for i in range(5)]
    found = fixed([RIGHT, WRONG], [WRONG, RIGHT], [WRONG, WRONG, RIGHT], [WRONG, WRONG, WRONG, RIGHT], [WRONG] * 10)
    out = score_passage_questions(qs, [SPAN] * 5, found)
    assert [(r.first, r.top_k, r.rr, r.recall10) for r in out] == [
        (True, True, 1.0, True), (False, True, 0.5, True), (False, True, 1 / 3, True), (False, False, 0.25, True),
        (False, False, 0.0, False)]
    assert all(isinstance(r, QResult) for r in out) and [r.q for r in out] == ["q0", "q1", "q2", "q3", "q4"]


def test_only_the_first_ten_passages_count():
    out = score_passage_questions([question()], [SPAN], fixed([WRONG] * 10 + [RIGHT]))
    assert out[0].recall10 is False and out[0].rr == 0.0
    out = score_passage_questions([question()], [SPAN], fixed([WRONG] * 9 + [RIGHT]))
    assert out[0].recall10 is True and out[0].rr == pytest.approx(0.1)


def test_a_cut_quote_is_not_held_by_either_half():
    halves = [hit("a", 0, 150), hit("a", 150, 400)]
    out = score_passage_questions([question()], [SPAN], fixed(halves))
    assert (out[0].first, out[0].top_k, out[0].recall10) == (False, False, False)


def test_scoring_asks_for_ten_passages_and_checks_lengths():
    asked = []
    score_passage_questions([question()], [SPAN], lambda q, n: asked.append(n) or [])
    assert asked == [10]
    with pytest.raises(ValueError, match="one place per question"):
        score_passage_questions([question()], [], fixed([]))


# --- quote cut by the chunking ------------------------------------------------------------------------

def run_of(spans):
    rows = [{"passage_id": f"p{i}", "source_id": s, "start": a, "end": b, "headings": [], "parent": None}
            for i, (s, a, b) in enumerate(spans)]
    return Run("r", {"inputs": {}}, rows, np.zeros((len(rows), 4), dtype=np.float32))


def test_cut_flags_mark_quotes_that_no_passage_holds():
    run = run_of([("a", 0, 150), ("a", 150, 400), ("b", 0, 1000)])
    assert cut_flags(run, [SPAN, QuoteSpan("a", 10, 140), QuoteSpan("b", 5, 500), QuoteSpan("c", 0, 5)]) == [
        True, False, False, True]


# --- passage pipelines ---------------------------------------------------------------------------------

class MapEmbedder:
    id = "map"
    fingerprint = "map"
    dim = 4

    def embed(self, texts, *, kind):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        out[:, 0] = 1.0
        return out


def vec(cosine):
    return [cosine, math.sqrt(1 - cosine * cosine), 0.0, 0.0]


def setup(spec):
    rows = [{"passage_id": f"p{i}", "source_id": s, "start": a, "end": a + len(t), "headings": [], "parent": None}
            for i, (s, a, t, _) in enumerate(spec)]
    return Run("r", {"inputs": {}}, rows, np.array([vec(c) for *_, c in spec], dtype=np.float32)), [t for _, _, t, _ in spec]


SPEC = [("A", 0, "unrelated words here", 0.9), ("B", 0, "retry timeout handling", 0.8),
        ("B", 50, "retry only here", 0.7), ("C", 0, "nothing", 0.1)]
QUERY = "retry timeout"


def passages(variant, **kw):
    run, texts = setup(SPEC)
    return make_passage_pipeline(variant, run, texts, embedder=MapEmbedder(), reranker=FakeReranker(), **kw)


def test_passage_answers_keep_every_passage_of_a_file():
    hits = passages("V")(QUERY, 10)
    assert [(h.source_id, h.start) for h in hits] == [("A", 0), ("B", 0), ("B", 50), ("C", 0)]
    assert hits[0].end == len(SPEC[0][2]) and isinstance(hits[0], PassageHit)


def test_passage_answers_follow_each_variant():
    assert [(h.source_id, h.start) for h in passages("L")(QUERY, 10)] == [("B", 0), ("B", 50)]
    assert [(h.source_id, h.start) for h in passages("H")(QUERY, 10)] == [("B", 0), ("B", 50), ("A", 0), ("C", 0)]
    assert [(h.source_id, h.start) for h in passages("VR", depth=3)(QUERY, 10)] == [("B", 0), ("B", 50), ("A", 0)]
    assert [(h.source_id, h.start) for h in passages("HR", depth=2)(QUERY, 10)] == [("B", 0), ("B", 50)]


def test_n_limits_the_passages():
    assert len(passages("V")(QUERY, 2)) == 2


def test_file_answers_are_the_same_as_before():
    run, texts = setup(SPEC)
    for variant in PIPELINES:
        files = make_pipeline(variant, run, texts, embedder=MapEmbedder(), reranker=FakeReranker())(QUERY, 10)
        seen = []
        for h in make_passage_pipeline(variant, run, texts, embedder=MapEmbedder(), reranker=FakeReranker())(QUERY, 10):
            if h.source_id not in seen:
                seen.append(h.source_id)
        assert [h.source_id for h in files] == seen


def test_passage_pipelines_refuse_what_the_file_pipelines_refuse():
    run, texts = setup(SPEC)
    with pytest.raises(ValueError, match="reranker"):
        make_passage_pipeline("VR", run, texts, embedder=MapEmbedder())
    with pytest.raises(ValueError, match="unknown variant"):
        make_passage_pipeline("X", run, texts, embedder=MapEmbedder())
