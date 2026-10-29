"""Spec 13, unit 13a: the fixed-window and merge-small chunkers, and the structure switch of the check. Tests first."""
import random

import pytest

from ragcore.chunking import (ChunkConfig, ChunkingError, check_chunking, chunk_fixed, chunk_merge_small,
                              chunk_source)
from ragcore.contracts import Source, Span, make_passages


def source(text, source_id="s1/doc.md", version="v1"):
    return Source(source_id=source_id, version=version, text=text, raw_sha256="0" * 64, loader_version="t-1",
                  licence="MIT", licence_basis="declared")


def texts(result):
    return [p.text for p in result.passages]


# --- the structure switch of the check ------------------------------------------------------

def test_check_allows_a_heading_inside_a_passage_when_structure_is_off():
    src = source("# A\n\nx\n\n# B\n\ny\n")
    passages = make_passages(src, [Span(0, len(src.text.rstrip()), ())], chunker_version="t")
    config = ChunkConfig()
    assert any("heading line" in p for p in check_chunking(src, passages, config))
    assert check_chunking(src, passages, config, structure=False) == []


def test_structure_off_still_checks_everything_else():
    src = source("aaaa bbbb cccc")
    passages = make_passages(src, [Span(0, 4, ()), Span(2, 9, ())], chunker_version="t")
    problems = check_chunking(src, passages, ChunkConfig(), structure=False)
    assert any("overlap" in p for p in problems)
    passages = make_passages(src, [Span(0, 4, ())], chunker_version="t")
    assert any("lost" in p for p in check_chunking(src, passages, ChunkConfig(), structure=False))


def test_chunk_source_passes_the_version_and_structure_through():
    src = source("# A\n\nx\n\n# B\n\ny\n")
    with pytest.raises(ChunkingError, match="heading line"):
        chunk_source(src, chunker=lambda s: [Span(0, len(s.text.rstrip()), ())])
    r = chunk_source(src, chunker=lambda s: [Span(0, len(s.text.rstrip()), ())], version="x-1", structure=False)
    assert r.passages[0].chunker_version == "x-1"


# --- fixed ----------------------------------------------------------------------------------

def test_fixed_short_text_is_one_passage_with_the_heading_in_force():
    r = chunk_fixed(source("# Guide\n\n## Install\n\nRun it.\n"), ChunkConfig(max_chars=100))
    assert texts(r) == ["# Guide\n\n## Install\n\nRun it."]
    assert r.passages[0].headings == ("Guide",)  # The passage starts at the heading line, so its path is in force.


def test_fixed_headings_follow_the_start_of_each_passage():
    para = ["x" * 80, "y" * 80, "z" * 80]
    text = "# Guide\n\n" + para[0] + "\n\n## Install\n\n" + para[1] + "\n\n" + para[2] + "\n"
    r = chunk_fixed(source(text), ChunkConfig(max_chars=100, hard_limit=100))
    assert [p.headings for p in r.passages][0] == ("Guide",)
    assert r.passages[-1].headings == ("Guide", "Install")
    assert len(r.passages) >= 3


def test_fixed_ignores_headings_when_the_window_is_big():
    r = chunk_fixed(source("# A\n\nxxx\n\n# B\n\nyyy\n"), ChunkConfig(max_chars=500))
    assert len(r.passages) == 1 and "# B" in r.passages[0].text


def test_fixed_windows_end_at_the_latest_paragraph_end_that_fits():
    paras = ["p" * 40, "q" * 40, "r" * 40]
    r = chunk_fixed(source("\n\n".join(paras) + "\n"), ChunkConfig(max_chars=100, hard_limit=100))
    assert texts(r) == [paras[0] + "\n\n" + paras[1], paras[2]]


def test_fixed_cuts_a_spaceless_line_at_the_limit():
    r = chunk_fixed(source("a" * 250), ChunkConfig(max_chars=100, hard_limit=100))
    assert [len(t) for t in texts(r)] == [100, 100, 50]


def test_fixed_windows_are_at_most_max_chars_and_never_blank():
    text = "\n\n".join(f"Paragraph {i} " + "word " * (i % 17 + 3) for i in range(80))
    r = chunk_fixed(source(text), ChunkConfig(max_chars=300, hard_limit=300))
    assert all(0 < len(t) <= 300 and t == t.strip() for t in texts(r))
    assert len(r.passages) > 5


def test_fixed_drops_a_tail_that_is_only_a_heading_and_counts_it():
    r = chunk_fixed(source("aaaa\n\n# H\n"), ChunkConfig(max_chars=6, hard_limit=6))
    assert texts(r) == ["aaaa"] and r.dropped_empty == 1


def test_fixed_version_and_determinism():
    src = source("# A\n\n" + "word " * 200)
    a, b = chunk_fixed(src, ChunkConfig(max_chars=200)), chunk_fixed(src, ChunkConfig(max_chars=200))
    assert texts(a) == texts(b) and a.passages[0].chunker_version == "fixed-1/200/6000"


def test_fixed_empty_and_blank_sources():
    assert chunk_fixed(source("\n\n  \n")).passages == []
    assert chunk_fixed(source("# Only a heading\n")).passages == []


def test_fixed_random_documents_pass_the_checks():
    rng = random.Random(11)
    pieces = ["# T\n", "## S\n", "word " * 30, "line\nbreak " * 5, "```\ncode\n\nmore\n```\n", "\n\n", "\n"]
    for _ in range(100):
        text = "".join(rng.choice(pieces) for _ in range(rng.randint(1, 25)))
        if text.strip():
            r = chunk_fixed(source(text), ChunkConfig(max_chars=rng.choice([50, 120, 400]), hard_limit=400))
            assert all(len(p.text) <= 400 for p in r.passages)


# --- merge-small ----------------------------------------------------------------------------

def test_merge_small_joins_tiny_neighbours_and_keeps_the_first_heading_path():
    text = "# A\n\n## B\n\nbb bb\n\n## C\n\ncc cc\n\n## D\n\ndd dd\n"
    rules = chunk_source(source(text), ChunkConfig(max_chars=50))
    r = chunk_merge_small(source(text), ChunkConfig(max_chars=500), min_chars=300)
    assert len(rules.passages) > 1 and len(r.passages) == 1
    assert "## C" in r.passages[0].text and "dd dd" in r.passages[0].text
    assert r.passages[0].headings == ("A", "B")


def test_merge_small_never_crosses_a_top_level_section():
    r = chunk_merge_small(source("# A\n\nsmall one\n\n# B\n\nsmall two\n"), ChunkConfig(), min_chars=300)
    assert texts(r) == ["# A\n\nsmall one", "# B\n\nsmall two"]


def test_merge_small_stops_at_max_chars():
    sections = [f"## S{i}\n\n" + "w" * 100 for i in range(6)]
    text = "# Top\n\n" + "\n\n".join(sections) + "\n"
    r = chunk_merge_small(source(text), ChunkConfig(max_chars=450), min_chars=300)
    assert len(r.passages) > 1 and all(len(t) <= 450 for t in texts(r))
    assert sum(len(t) for t in texts(r)) > 600


def test_merge_small_leaves_a_big_left_neighbour_alone():
    big, small = "b" * 320, "small"
    r = chunk_merge_small(source(f"# A\n\n## X\n\n{big}\n\n## Y\n\n{small}\n"), ChunkConfig(max_chars=2000), min_chars=300)
    assert len(r.passages) == 2
    assert not any(big in t and small in t for t in texts(r))


def test_merge_small_keeps_going_while_the_merged_piece_is_still_small():
    text = "# A\n\n## B\n\n1\n\n## C\n\n2\n\n## D\n\n3\n\n## E\n\n4\n"
    r = chunk_merge_small(source(text), ChunkConfig(), min_chars=300)
    assert len(r.passages) == 1


def test_merge_small_version_and_checks():
    src = source("# A\n\n## B\n\nbb\n\n## C\n\ncc\n")
    r = chunk_merge_small(src, ChunkConfig(max_chars=500), min_chars=250)
    assert r.passages[0].chunker_version == "merge-small-1/500/6000/250"
    assert check_chunking(src, r.passages, ChunkConfig(max_chars=500), structure=False) == []


def test_merge_small_has_no_more_passages_than_rules_and_loses_no_text():
    rng = random.Random(5)
    pieces = ["# T\n\n", "## S\n\n", "short.\n\n", "word " * 80 + "\n\n", "### U\n\nx\n\n"]
    for _ in range(100):
        text = "".join(rng.choice(pieces) for _ in range(rng.randint(1, 20)))
        config = ChunkConfig(max_chars=400, hard_limit=1000)
        rules = chunk_source(source(text), config)
        merged = chunk_merge_small(source(text), config, min_chars=150)
        assert len(merged.passages) <= len(rules.passages)
        assert merged.dropped_empty == rules.dropped_empty


def test_merge_small_rejects_a_bad_threshold():
    with pytest.raises(ValueError):
        chunk_merge_small(source("# A\n\nx\n"), ChunkConfig(), min_chars=0)


# --- found by the mutation check ------------------------------------------------------------

def test_fixed_windows_use_max_chars_not_the_hard_limit():
    r = chunk_fixed(source("a" * 250), ChunkConfig(max_chars=100, hard_limit=300))
    assert [len(t) for t in texts(r)] == [100, 100, 50]


def test_fixed_prefers_the_latest_line_end_when_no_paragraph_end_fits():
    lines = [f"line{i} " + "x" * 20 for i in range(6)]  # 27 characters each, one paragraph
    r = chunk_fixed(source("\n".join(lines) + "\n"), ChunkConfig(max_chars=100, hard_limit=100))
    assert texts(r)[0] == "\n".join(lines[:3])  # 3 lines = 83 characters, 4 lines = 111


def test_fixed_prefers_a_paragraph_end_over_a_later_line_end():
    p1, p2, p3 = "a" * 40, "b" * 30 + "\n" + "c" * 30, "d" * 10 + "\n" + "e" * 50
    r = chunk_fixed(source(f"{p1}\n\n{p2}\n\n{p3}\n"), ChunkConfig(max_chars=120, hard_limit=120))
    assert texts(r)[0] == f"{p1}\n\n{p2}"


MERGE_TEXT = "# A\n\n## X\n\nabc\n\n## Y\n\ndef\n"  # Passage X is "## X\n\nabc" (9 characters); the join is 20.


def test_merge_small_threshold_is_exclusive():
    assert len(chunk_merge_small(source(MERGE_TEXT), ChunkConfig(), min_chars=9).passages) == 2
    assert len(chunk_merge_small(source(MERGE_TEXT), ChunkConfig(), min_chars=10).passages) == 1


def test_merge_small_limit_is_inclusive():
    assert len(chunk_merge_small(source(MERGE_TEXT), ChunkConfig(max_chars=20), min_chars=300).passages) == 1
    assert len(chunk_merge_small(source(MERGE_TEXT), ChunkConfig(max_chars=19), min_chars=300).passages) == 2
