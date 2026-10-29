"""Spec 09: heading-pack chunking and the exact-slice check. Written before the code."""
import random
import re

import pytest

from ragcore.chunking import ChunkConfig, ChunkingError, check_chunking, chunk_source
from ragcore.contracts import Source, Span, make_passages

TICKS = "`" * 3


def source(text, source_id="s1/doc.md", version="v1"):
    return Source(source_id=source_id, version=version, text=text, raw_sha256="0" * 64, loader_version="t-1",
                  licence="MIT", licence_basis="declared")


def chunk(text, **config):
    return chunk_source(source(text), ChunkConfig(**config))


def texts(result):
    return [p.text for p in result.passages]


# --- sections and heading paths -------------------------------------------------------------

def test_sections_and_heading_paths_with_a_skipped_level():
    text = "# Guide\n\nIntro text here.\n\n## Install\n\nRun it.\n\n#### Deep\n\nDeep text.\n"
    r = chunk(text, max_chars=30)
    assert texts(r) == ["# Guide\n\nIntro text here.", "## Install\n\nRun it.", "#### Deep\n\nDeep text."]
    assert [p.headings for p in r.passages] == [("Guide",), ("Guide", "Install"), ("Guide", "Install", "Deep")]


def test_a_sibling_heading_replaces_the_previous_one_at_its_level():
    r = chunk("# A\n\n## B\n\nbody b\n\n## C\n\nbody c\n\n# D\n\nbody d\n", max_chars=20)
    assert [p.headings for p in r.passages] == [("A", "B"), ("A", "C"), ("D",)]


def test_text_before_the_first_heading_is_its_own_section_with_no_path():
    r = chunk("Preface text.\n\n# H\n\nBody.\n")
    assert texts(r) == ["Preface text.", "# H\n\nBody."]
    assert r.passages[0].headings == ()


def test_closing_hashes_and_extra_spaces_in_a_heading():
    r = chunk("##   Title with hashes  ##\n\nBody.\n")
    assert r.passages[0].headings == ("Title with hashes",)


def test_a_hash_line_inside_a_fence_is_not_a_heading():
    text = f"# A\n\n{TICKS}\n# not a heading\n{TICKS}\n\nafter\n"
    r = chunk(text)
    assert len(r.passages) == 1 and r.passages[0].headings == ("A",) and "# not a heading" in r.passages[0].text


def test_a_hash_without_a_space_is_not_a_heading():
    r = chunk("# A\n\n#hashtag text\n\nmore\n")
    assert len(r.passages) == 1


# --- paragraphs and packing -----------------------------------------------------------------

def test_a_fence_with_blank_lines_is_never_split():
    text = f"# A\n\n{TICKS}\nx = 1\n\ny = 2\n{TICKS}\n"
    r = chunk(text, max_chars=20)
    assert len(r.passages) == 1 and "x = 1\n\ny = 2" in r.passages[0].text


def test_an_unclosed_fence_runs_to_the_end():
    text = f"# A\n\nintro\n\n{TICKS}\ncode\n\n# still code\n"
    r = chunk(text, max_chars=10)
    assert "# still code" in r.passages[-1].text and len(r.passages) == 2


def test_paragraphs_pack_up_to_max_chars_and_the_next_one_starts_a_new_passage():
    p = ["a" * 100, "b" * 100, "c" * 100]
    r = chunk("# T\n\n" + "\n\n".join(p) + "\n", max_chars=250)
    assert texts(r) == ["# T\n\n" + p[0] + "\n\n" + p[1], p[2]]
    assert r.passages[1].headings == ("T",)


def test_a_heading_always_goes_with_the_paragraph_after_it():
    r = chunk("# T\n\n" + "w" * 300 + "\n", max_chars=100)
    assert texts(r) == ["# T\n\n" + "w" * 300]


def test_a_passage_never_crosses_a_section():
    r = chunk("# A\n\nsmall\n\n# B\n\nsmall too\n")
    assert len(r.passages) == 2


def test_passages_are_trimmed():
    r = chunk("# T\n\n\n  Body text.   \n\n\n")
    assert texts(r) == ["# T\n\n\n  Body text."]


# --- hard limit -----------------------------------------------------------------------------

def test_a_long_single_line_is_cut_at_a_space():
    words = " ".join(f"w{i}" for i in range(400))
    r = chunk("# T\n\n" + words + "\n", max_chars=100, hard_limit=500)
    assert len(r.passages) > 3 and all(len(t) <= 500 for t in texts(r))
    assert " ".join(texts(r)).split() == ("# T " + words).split()


def test_a_multi_line_paragraph_is_cut_only_at_line_ends():
    lines = [f"line {i:02d} " + "x" * 70 for i in range(20)]
    r = chunk("# T\n\n" + "\n".join(lines) + "\n", max_chars=100, hard_limit=500)
    assert all(len(t) <= 500 for t in texts(r))
    cut_lines = [ln for t in texts(r) for ln in t.split("\n") if ln.startswith("line ")]
    assert cut_lines == lines


def test_a_token_with_no_spaces_is_cut_at_the_limit():
    r = chunk("# T\n\n" + "z" * 1200 + "\n", max_chars=100, hard_limit=500)
    assert all(len(t) <= 500 for t in texts(r)) and len(r.passages) == 3
    assert "".join("".join(texts(r)).split()) == "#T" + "z" * 1200


def test_a_cut_never_leaves_a_heading_alone():
    r = chunk("# T\n\n" + "word " * 300 + "\n", max_chars=100, hard_limit=500)
    assert all(not re.fullmatch(r"#+ .*", t) for t in texts(r)) and texts(r)[0].startswith("# T\n\nword")


def test_config_is_validated():
    with pytest.raises(ValueError):
        ChunkConfig(max_chars=0)
    with pytest.raises(ValueError):
        ChunkConfig(max_chars=100, hard_limit=50)


def test_a_paragraph_over_max_chars_but_under_the_hard_limit_stays_whole():
    r = chunk("# T\n\n" + "word " * 100 + "\n", max_chars=100, hard_limit=1000)
    assert len(r.passages) == 1


# --- dropping -------------------------------------------------------------------------------

def test_a_section_with_only_a_heading_is_dropped_and_counted():
    r = chunk("# A\n\n## B\n\nBody.\n")
    assert texts(r) == ["## B\n\nBody."] and r.dropped_empty == 1


def test_html_comment_only_passages_are_dropped():
    r = chunk("# A\n\n<!-- generated marker -->\n\n## B\n\nBody.\n")
    assert texts(r) == ["## B\n\nBody."] and r.dropped_empty == 1


def test_blank_source_gives_nothing():
    r = chunk("  \n\n")
    assert r.passages == [] and r.dropped_empty == 0


# --- ids and determinism --------------------------------------------------------------------

def test_deterministic_and_ids_follow_the_config():
    text = "# A\n\none\n\n## B\n\ntwo\n"
    a, b = chunk(text), chunk(text)
    assert a.passages == b.passages
    other = chunk(text, max_chars=1000)
    assert [(p.start, p.end) for p in a.passages] == [(p.start, p.end) for p in other.passages]
    assert all(x.passage_id != y.passage_id for x, y in zip(a.passages, other.passages))
    assert a.passages[0].chunker_version == "heading-pack-1/1500/6000"


def test_a_content_id_is_shared_by_equal_text_in_different_places():
    r = chunk("# A\n\nsame text here\n\n# B\n\nsame text here\n")
    assert r.passages[0].content_id != r.passages[1].content_id  # the headings differ, so the text differs
    r = chunk("same text here\n\n```\ncode\n```\n\nsame text here\n", max_chars=15)
    assert r.passages[0].content_id == r.passages[2].content_id


# --- the checks -----------------------------------------------------------------------------

def spans_to_passages(src, spans):
    return make_passages(src, [Span(s, e) for s, e in spans], chunker_version="test-1")


def test_a_sound_chunking_has_no_problems():
    src = source("# A\n\none\n\n## B\n\ntwo\n")
    r = chunk_source(src)
    assert check_chunking(src, r.passages, ChunkConfig()) == []


def test_check_names_overlap_and_order_and_duplicates():
    src = source("0123456789 0123456789 0123456789\n")
    overlap = spans_to_passages(src, [(0, 10), (5, 20), (22, 31)])
    assert any("overlap" in p for p in check_chunking(src, overlap, ChunkConfig()))
    order = spans_to_passages(src, [(12, 20), (0, 10)])
    assert any("order" in p for p in check_chunking(src, order, ChunkConfig()))
    twice = spans_to_passages(src, [(0, 10), (0, 10)])
    assert any("duplicate" in p for p in check_chunking(src, twice, ChunkConfig()))


def test_check_names_a_heading_inside_a_passage():
    src = source("para one\n# Heading\npara two\n")
    bad = spans_to_passages(src, [(0, len(src.text) - 1)])
    assert any("heading" in p for p in check_chunking(src, bad, ChunkConfig()))


def test_check_names_lost_text():
    src = source("first para\n\nmiddle words\n\nlast para\n")
    last = src.text.index("last para")
    bad = spans_to_passages(src, [(0, 10), (last, last + 9)])
    assert any("lost" in p for p in check_chunking(src, bad, ChunkConfig()))
    only_start = spans_to_passages(src, [(last, last + 9)])
    assert any("lost" in p for p in check_chunking(src, only_start, ChunkConfig()))
    only_end = spans_to_passages(src, [(0, 10)])
    assert any("lost" in p for p in check_chunking(src, only_end, ChunkConfig()))


def test_check_allows_lost_whitespace_headings_and_comments_only():
    src = source("# A\n\nfirst para\n\n<!-- c -->\n\n## B\n\nsecond para\n")
    a, b = src.text.index("first para"), src.text.index("second para")
    ok = spans_to_passages(src, [(a, a + 10), (b, b + 11)])
    assert check_chunking(src, ok, ChunkConfig()) == []


def test_check_names_over_the_hard_limit():
    src = source("x" * 50 + "\n")
    bad = spans_to_passages(src, [(0, 50)])
    assert any("hard limit" in p for p in check_chunking(src, bad, ChunkConfig(max_chars=5, hard_limit=10)))


def test_check_names_a_passage_from_another_source_or_version():
    a, b = source("some words here\n"), source("some words here\n", version="v2")
    foreign = spans_to_passages(b, [(0, 15)])
    assert any("slice" in p for p in check_chunking(a, foreign, ChunkConfig()))


def test_chunk_source_refuses_a_chunker_that_breaks_the_checks():
    def overlapping(src):
        return [Span(0, 10), Span(5, 20)]
    with pytest.raises(ChunkingError):
        chunk_source(source("0123456789 0123456789 0123456789\n"), ChunkConfig(), chunker=overlapping)


# --- properties -----------------------------------------------------------------------------

PIECES = ["# One\n", "## Two\n", "### Three\n", "#### Four\n", "Plain words in a paragraph.\n", "\n", "\n\n",
          "A longer paragraph " + "with several words " * 12 + "\n", f"{TICKS}\ncode = 1\n\n# not heading\n{TICKS}\n",
          "<!-- marker -->\n", "line one\nline two\nline three\n", "x" * 130 + "\n", "#nothash\n", "  indented text\n"]


def test_properties_on_random_markdown():
    rng = random.Random(5)
    for _ in range(400):
        text = "".join(rng.choice(PIECES) for _ in range(rng.randint(1, 16)))
        hard = rng.choice([60, 150, 400, 6000])
        config = ChunkConfig(max_chars=min(rng.choice([30, 80, 200, 1500]), hard), hard_limit=hard)
        src = source(text)
        r = chunk_source(src, config)
        assert check_chunking(src, r.passages, config) == [], repr(text)
        assert r.passages == chunk_source(src, config).passages
        assert all(len(p.text) <= config.hard_limit for p in r.passages)
        assert all(p.text == p.text.strip() for p in r.passages)
        for fence in re.findall(rf"{TICKS}\n.*?\n{TICKS}\n", text, re.S):  # a short fence stays in one passage
            if len(fence) <= config.hard_limit - 10:
                assert any(fence.strip() in p.text for p in r.passages), repr(text)


# --- boundaries and rare cases (found by the mutation check) -----------------------------------

def test_max_chars_is_inclusive():
    text = "# T\n\n" + "a" * 100 + "\n\n" + "b" * 100 + "\n"  # first two blocks span exactly 207 characters
    assert len(chunk(text, max_chars=207).passages) == 1
    assert len(chunk(text, max_chars=206).passages) == 2


def test_hard_limit_is_inclusive():
    assert len(chunk("a" * 100, max_chars=100, hard_limit=100).passages) == 1
    assert texts(chunk("a" * 101, max_chars=100, hard_limit=100)) == ["a" * 100, "a"]


def test_a_long_line_is_cut_at_a_space_not_inside_a_word():
    words = " ".join(["word"] * 30)
    r = chunk(words, max_chars=40, hard_limit=40)
    assert len(r.passages) > 1
    assert all(len(t) <= 40 and t == t.strip() and set(t.split()) == {"word"} for t in texts(r))


def test_a_hard_cut_prefers_the_latest_line_end_in_the_window():
    lines = "\n".join(f"line {i:02d} xxxxxxxxxx" for i in range(10))  # 18 characters per line
    r = chunk(lines, max_chars=60, hard_limit=60)
    assert texts(r)[0] == "\n".join(f"line {i:02d} xxxxxxxxxx" for i in range(3))  # 3 lines = 56, 4 lines = 75


def test_a_hash_inside_a_heading_word_stays_in_the_title():
    r = chunk("# C#\n\nbody\n\n## F# notes\n\nmore\n", max_chars=10)
    assert [p.headings for p in r.passages] == [("C#",), ("C#", "F# notes")]


def test_backticks_in_a_fence_info_string_do_not_open_a_fence():
    r = chunk("# A\n\n``` a`b\n\n# H2\n\nz\n", max_chars=10)
    assert r.passages[-1].headings == ("H2",)


def test_a_closing_fence_must_match_and_carry_no_text():
    text = f"# A\n\n{TICKS}\ncode\n{TICKS} not closing\n# still code\n{TICKS}\n\n# B\n\nz\n"
    r = chunk(text, max_chars=10)
    assert [p.headings for p in r.passages] == [("A",), ("B",)]
    assert "# still code" in r.passages[0].text


def test_a_blank_span_is_refused_by_the_contract_before_the_check():
    from ragcore.contracts import ContractError
    with pytest.raises(ContractError, match="blank"):
        chunk_source(source("hello   world"), chunker=lambda s: [Span(0, 5, ()), Span(5, 8, ())])


def test_the_check_reports_a_heading_inside_a_passage_but_allows_one_at_the_start():
    src = source("intro\n\n# B\n\ny\n")
    assert chunk_source(src, chunker=lambda s: [Span(0, 5, ()), Span(7, len(s.text), ("B",))])
    with pytest.raises(ChunkingError, match="heading line"):
        chunk_source(src, chunker=lambda s: [Span(0, len(s.text), ())])


def test_the_check_enforces_the_hard_limit_inclusively():
    src = source("a" * 101)
    assert chunk_source(src, ChunkConfig(max_chars=100, hard_limit=101), chunker=lambda s: [Span(0, 101, ())])
    with pytest.raises(ChunkingError, match="hard limit"):
        chunk_source(src, ChunkConfig(max_chars=100, hard_limit=100), chunker=lambda s: [Span(0, 101, ())])
