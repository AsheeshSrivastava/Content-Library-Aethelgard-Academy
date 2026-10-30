"""Late chunking (spec 13 section 4.3, unit 13h): windows, pooling, fallbacks. Fake embedders only."""
import re

import numpy as np
import pytest

from ragcore.chunking import ChunkConfig
from ragcore.contracts import Source
from ragcore.embed.base import FakeEmbedder
from ragcore.embed.late import MARGIN, embed_passages_late, late_vectors, plan_windows
from ragcore.stage import build_run, chunk_corpus, run_inputs, corpus_fingerprint
from ragcore.runs import compute_run_id, load_run

CONFIG = ChunkConfig(max_chars=400, hard_limit=2000)


def words(n):
    return " ".join(f"w{i}" for i in range(n))


def spans_of(text, *pieces):
    out = []
    for piece in pieces:
        start = text.index(piece)
        out.append((start, start + len(piece)))
    return out


class WindowFake(FakeEmbedder):
    """Every token vector says which window it came from: one-hot at (first word number // 10)."""

    def __init__(self, max_tokens):
        super().__init__(dim=32)
        self.max_tokens = max_tokens
        self.window_texts = []

    def token_states(self, text):
        self.window_texts.append(text)
        states = super().token_states(text)
        first = int(re.search(r"w(\d+)", text).group(1))
        vectors = np.zeros_like(states.vectors)
        vectors[:, first // 10] = 1.0
        return type(states)(vectors, states.offsets)


# --- plan_windows -------------------------------------------------------------------------------

def test_text_that_fits_is_one_window():
    assert plan_windows(10, 40, 10) == [(0, 10)]
    assert plan_windows(40, 40, 10) == [(0, 40)]


def test_empty_text_has_no_window():
    assert plan_windows(0, 40, 10) == []


def test_windows_slide_with_the_overlap_and_end_exactly_at_the_end():
    wins = plan_windows(100, 40, 10)
    assert wins == [(0, 40), (30, 70), (60, 100)]


def test_windows_cover_everything_and_share_at_least_the_overlap():
    for n in (41, 77, 100, 1001):
        wins = plan_windows(n, 40, 10)
        assert wins[0][0] == 0 and wins[-1][1] == n
        assert all(b - a <= 40 for a, b in wins)
        for (a1, b1), (a2, b2) in zip(wins, wins[1:]):
            assert b1 - a2 >= 10 and a2 > a1


def test_last_window_is_pulled_back_to_full_size():
    assert plan_windows(77, 40, 10) == [(0, 40), (30, 70), (37, 77)]


def test_bad_window_settings_are_refused():
    for args in ((100, 0, 0), (100, 40, 40), (100, 40, 50), (100, 40, -1)):
        with pytest.raises(ValueError):
            plan_windows(*args)


# --- late_vectors -------------------------------------------------------------------------------

def test_one_passage_source_matches_normal_embedding():
    emb = FakeEmbedder()
    text = "Alpha explains how retrieval ranks passages by similarity."
    result = late_vectors(emb, text, [(0, len(text))])
    normal = emb.embed([text], kind="passage")
    assert result.fallback_long == [] and result.fallback_empty == []
    assert np.allclose(result.vectors, normal, atol=1e-4)


def test_context_changes_the_vector_but_it_stays_unit_length():
    emb = FakeEmbedder()
    text = "Intro about volcano glass. Obsidian knives cut sharply. Closing remarks about zebras."
    (span,) = spans_of(text, "Obsidian knives cut sharply.")
    late = late_vectors(emb, text, [span]).vectors[0]
    normal = emb.embed([text[span[0]:span[1]]], kind="passage")[0]
    assert abs(np.linalg.norm(late) - 1.0) < 1e-5
    assert not np.allclose(late, normal, atol=1e-3)
    assert float(late @ normal) > 0.5  # Still about the same words.


def test_each_passage_takes_its_vector_from_the_best_window():
    emb = WindowFake(max_tokens=40 + MARGIN)  # Effective window 40 tokens.
    text = words(100)
    spans = spans_of(text, "w5 w6 w7", "w35 w36", "w95 w96 w97")
    result = late_vectors(emb, text, spans, overlap=10)
    hot = [int(np.argmax(v)) for v in result.vectors]
    # w5..w7: only window 0 holds them. w35..w36: window 0 leaves 3 tokens after, window 1 (from w30) leaves 5
    # before, so window 1. w95..w97: only the last window (from w60).
    assert hot == [0, 3, 6]
    assert result.windows == 3
    assert all(abs(np.linalg.norm(v) - 1.0) < 1e-5 for v in result.vectors)


def test_a_passage_touching_the_document_edge_counts_that_edge_as_far_away():
    emb = WindowFake(max_tokens=40 + MARGIN)
    text = words(100)
    (span,) = spans_of(text, "w0 w1 w2")
    assert int(np.argmax(late_vectors(emb, text, [span], overlap=10).vectors[0])) == 0


def test_only_windows_that_some_passage_needs_are_run():
    emb = WindowFake(max_tokens=40 + MARGIN)
    text = words(100)
    late_vectors(emb, text, spans_of(text, "w5 w6"), overlap=10)
    assert len(emb.window_texts) == 1


def test_a_tie_goes_to_the_earlier_window():
    emb = WindowFake(max_tokens=40 + MARGIN)
    text = words(100)
    (span,) = spans_of(text, "w34 w35")  # Window 0 leaves 4 after; window 1 (from w30) leaves 4 before.
    assert int(np.argmax(late_vectors(emb, text, [span], overlap=10).vectors[0])) == 0


def test_passage_longer_than_a_window_falls_back_and_leaves_a_zero_row():
    emb = WindowFake(max_tokens=40 + MARGIN)
    text = words(100)
    spans = spans_of(text, "w5 w6") + [(0, len(text))]
    result = late_vectors(emb, text, spans, overlap=10)
    assert result.fallback_long == [1] and result.fallback_empty == []
    assert not result.vectors[1].any() and result.vectors[0].any()


def test_passage_without_tokens_falls_back():
    emb = FakeEmbedder()
    text = "Words here.\n\n...\n\nMore words."
    (span,) = spans_of(text, "...")
    result = late_vectors(emb, text, [span])
    assert result.fallback_empty == [0] and not result.vectors[0].any()


def test_token_cut_by_a_span_edge_is_left_out():
    emb = FakeEmbedder()
    text = "alphabet soup tastes good"
    # The span ends in the middle of "soup": only "alphabet" lies fully inside.
    only = late_vectors(emb, text, [(0, 11)]).vectors[0]
    whole_word = late_vectors(emb, text, [(0, 8)]).vectors[0]
    assert np.allclose(only, whole_word, atol=1e-6)


def test_vectors_do_not_depend_on_the_order_of_the_passages():
    emb = FakeEmbedder()
    text = "One alpha sentence here. Two beta sentence there. Three gamma sentence last."
    spans = spans_of(text, "One alpha sentence here.", "Two beta sentence there.", "Three gamma sentence last.")
    forward = late_vectors(emb, text, spans).vectors
    backward = late_vectors(emb, text, spans[::-1]).vectors[::-1]
    assert np.allclose(forward, backward, atol=1e-6)


def test_bad_spans_and_settings_are_refused():
    emb = FakeEmbedder()
    with pytest.raises(ValueError, match="span"):
        late_vectors(emb, "abc def", [(3, 3)])
    with pytest.raises(ValueError, match="span"):
        late_vectors(emb, "abc def", [(0, 99)])
    with pytest.raises(ValueError, match="overlap"):
        late_vectors(emb, "abc def", [(0, 3)], window=100, overlap=100)


def test_a_passage_of_exactly_one_window_is_pooled_and_one_token_more_falls_back():
    emb = WindowFake(max_tokens=40 + MARGIN)
    text = words(100)
    fits = spans_of(text, " ".join(f"w{i}" for i in range(0, 40)))  # 40 tokens
    too_long = spans_of(text, " ".join(f"w{i}" for i in range(0, 41)))  # 41 tokens
    result = late_vectors(emb, text, fits + too_long, overlap=10)
    assert result.fallback_long == [1] and result.vectors[0].any()


def test_a_passage_cut_by_every_window_boundary_falls_back_too():
    emb = WindowFake(max_tokens=40 + MARGIN)
    text = words(100)
    # 20 tokens is shorter than a window, but windows (0,40) and (30,70) each cut it (w25..w44).
    result = late_vectors(emb, text, spans_of(text, " ".join(f"w{i}" for i in range(25, 45))), overlap=10)
    assert result.fallback_long == [0] and not result.vectors[0].any()


def test_default_window_is_the_model_limit_less_the_margin():
    result = late_vectors(FakeEmbedder(), "abc def", [(0, 3)])
    assert result.window_tokens == 8192 - 16


def test_text_without_any_token_sends_every_passage_to_fallback():
    result = late_vectors(FakeEmbedder(), "... ---", [(0, 3), (4, 7)])
    assert result.fallback_empty == [0, 1] and result.windows == 0


def test_a_text_that_fits_goes_through_the_model_whole():
    emb = WindowFake(max_tokens=200)
    text = "  w0 w1 w2 ...  "
    late_vectors(emb, text, [(2, 4)], overlap=10)
    assert emb.window_texts == [text]


def test_window_larger_than_the_model_limit_is_cut_to_the_limit():
    emb = WindowFake(max_tokens=40 + MARGIN)
    text = words(100)
    result = late_vectors(emb, text, spans_of(text, "w5 w6"), window=8192, overlap=10)
    assert result.window_tokens == 40


def test_window_below_the_margin_is_refused():
    emb = FakeEmbedder()
    emb.max_tokens = MARGIN
    with pytest.raises(ValueError, match="window"):
        late_vectors(emb, "abc def", [(0, 3)])


# --- embed_passages_late ------------------------------------------------------------------------

def source(name, text):
    return Source(f"s1/{name}", "v1", text, "0" * 64, "md-1", "MIT", "declared")


def corpus():
    para = "Retrieval finds passages that answer a question, and ranking orders them by cosine similarity. "
    return [source("a.md", f"# Alpha\n\n{para * 3}\n\n## Beta\n\nBeta passages talk about zebras and quartz.\n"),
            source("b.md", f"# Gamma\n\n{para * 2}\n\nGamma explains volcano glass and obsidian knives.\n")]


def test_every_passage_gets_a_vector_in_passage_order():
    emb = FakeEmbedder()
    srcs = corpus()
    passages = chunk_corpus(srcs, "rules", CONFIG)[0]
    vectors, stats = embed_passages_late(emb, srcs, passages)
    assert vectors.shape == (len(passages), emb.dim) and vectors.dtype == np.float32
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)
    assert stats["late_fallback_too_long"] == 0 and stats["late_fallback_no_tokens"] == 0
    assert stats["late_windows"] == len(srcs)  # Both files fit in one window.
    for p, v in zip(passages, vectors):  # Passage i is about passage i's words.
        assert float(v @ emb.embed([p.text], kind="passage")[0]) > 0.5


def test_fallback_passages_use_normal_embedding():
    text = "w0 " + "w1 " * 60
    src = source("long.md", text)
    passages = chunk_corpus([src], "rules", ChunkConfig(max_chars=400, hard_limit=2000))[0]
    plain = FakeEmbedder(dim=32)
    plain.max_tokens = 40 + MARGIN
    vectors, stats = embed_passages_late(plain, [src], passages, overlap=10)
    assert stats["late_fallback_too_long"] == len(passages)
    assert np.allclose(vectors, plain.embed([p.text for p in passages], kind="passage"), atol=1e-6)


def test_progress_is_reported_and_ends_at_the_total():
    emb = FakeEmbedder()
    srcs = corpus()
    passages = chunk_corpus(srcs, "rules", CONFIG)[0]
    seen = []
    embed_passages_late(emb, srcs, passages, progress=lambda done, total: seen.append((done, total)))
    assert seen[-1] == (len(passages), len(passages))
    assert [d for d, _ in seen] == sorted(d for d, _ in seen)


def test_passage_of_an_unknown_source_is_refused():
    emb = FakeEmbedder()
    srcs = corpus()
    passages = chunk_corpus(srcs, "rules", CONFIG)[0]
    with pytest.raises(ValueError, match="source"):
        embed_passages_late(emb, srcs[:1], passages)


# --- the stage runner ---------------------------------------------------------------------------

def test_late_run_is_a_different_run_with_its_own_stats(tmp_path):
    emb = FakeEmbedder()
    srcs = corpus()
    normal = build_run(tmp_path, "rules", srcs, emb, CONFIG, questions_fp="q1")
    late = build_run(tmp_path, "rules", srcs, emb, CONFIG, questions_fp="q1", mode="late")
    assert normal.run_id != late.run_id and not late.reused
    run = load_run(tmp_path, late.run_id)
    assert run.manifest["inputs"]["mode"] == "late"
    assert run.manifest["extra"]["late_windows"] == len(srcs)
    assert run.vectors.shape[0] == late.count == normal.count
    assert not np.allclose(run.vectors, load_run(tmp_path, normal.run_id).vectors, atol=1e-3)


def test_late_run_is_reused_the_second_time(tmp_path):
    emb = FakeEmbedder()
    srcs = corpus()
    first = build_run(tmp_path, "rules", srcs, emb, CONFIG, questions_fp="q1", mode="late")
    again = build_run(tmp_path, "rules", srcs, emb, CONFIG, questions_fp="q1", mode="late")
    assert again.reused and again.run_id == first.run_id


def test_late_window_setting_is_part_of_the_run_id(tmp_path):
    emb = FakeEmbedder()
    srcs = corpus()
    a = compute_run_id(run_inputs("rules", CONFIG, emb, corpus_fingerprint(srcs), "q1", "late"))
    b = compute_run_id(run_inputs("rules", CONFIG, emb, corpus_fingerprint(srcs), "q1", "late", late_window=4096))
    assert a != b


def test_unknown_mode_is_refused(tmp_path):
    with pytest.raises(ValueError, match="mode"):
        build_run(tmp_path, "rules", corpus(), FakeEmbedder(), CONFIG, questions_fp="q1", mode="sideways")
