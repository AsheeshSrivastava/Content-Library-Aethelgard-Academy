"""Random quote picker for the passage-level questions (spec 16 section 3)."""
import pytest

from ragcore.quote_picker import (Candidate, allocate, blocks, group_of, heading_before, hint, is_usable, pick_candidates, shape_quote)

PARA = ("Retries are attempted again when a call raises a timeout error, and the delay between two attempts grows "
        "after every failure.")


def doc(n, tag):
    return "\n\n".join(f"# Heading {i}\n\n{tag} paragraph {i}. {PARA}" for i in range(n)) + "\n"


# --- allocation -----------------------------------------------------------------------------------

def test_allocation_follows_the_square_root_of_the_size():
    assert allocate({"a": 100, "b": 10000}, 11, cap=100) == {"a": 1, "b": 10}


def test_allocation_adds_up_and_is_repeatable():
    sizes = {f"s{i}": 100 * (i + 1) for i in range(7)}
    out = allocate(sizes, 40, cap=40)
    assert sum(out.values()) == 40 and out == allocate(dict(reversed(list(sizes.items()))), 40, cap=40)


def test_allocation_cap_moves_the_rest_to_other_sources():
    out = allocate({"big": 1_000_000, "a": 10_000, "b": 10_000}, 30, cap=12)
    assert out["big"] == 12 and out["a"] + out["b"] == 18 and sum(out.values()) == 30


def test_allocation_when_everything_is_capped_gives_less():
    assert allocate({"a": 100, "b": 100}, 50, cap=5) == {"a": 5, "b": 5}


def test_allocation_edge_cases():
    assert allocate({}, 10, cap=5) == {}
    assert allocate({"a": 100}, 0, cap=5) == {"a": 0}
    with pytest.raises(ValueError, match="cap"):
        allocate({"a": 1}, 1, cap=0)
    with pytest.raises(ValueError, match="total"):
        allocate({"a": 1}, -1, cap=1)


# --- blocks --------------------------------------------------------------------------------------

def test_blocks_split_on_blank_lines():
    text = "one\ntwo\n\nthree\n\n\nfour\n"
    assert [text[s:e] for s, e in blocks(text)] == ["one\ntwo", "three", "four"]


def test_a_code_fence_is_one_block_even_with_blank_lines():
    text = "intro\n\n```python\nx = 1\n\ny = 2\n```\n\nafter\n"
    assert [text[s:e] for s, e in blocks(text)] == ["intro", "```python\nx = 1\n\ny = 2\n```", "after"]


def test_an_unclosed_fence_runs_to_the_end():
    text = "intro\n\n```\ncode\n\nmore\n"
    assert [text[s:e] for s, e in blocks(text)][-1] == "```\ncode\n\nmore"


# --- what is worth quoting -----------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "# Only a heading\n## and another one here, still just headings",
    "- [Intro](intro.md)\n- [Guide](guide.md)\n- [Reference](reference.md)\n- [Changelog](changelog.md)",
    "Licensed under the Apache License, Version 2.0 (the License); you may not use this file except in compliance.",
    "Permission is hereby granted, free of charge, to any person obtaining a copy of this software and files.",
    "SPDX-License-Identifier: MIT and some more words to be long enough for the length limit",
    "short",
    "| a | b |\n|---|---|\n|---|---|\n|---|---|\n|---|---|\n|---|---|\n|---|---|",
])
def test_unusable_blocks(text):
    assert not is_usable(text)


def test_a_normal_paragraph_and_code_are_usable():
    assert is_usable(PARA)
    assert is_usable("```python\nresult = client.call(timeout=30)\nprint(result.status_code)\n```")


def test_a_short_list_of_real_sentences_is_usable():
    assert is_usable("- Retries stop after three attempts have failed in a row.\n- The delay doubles each time it fails.")


def test_shaping_keeps_a_short_block_and_cuts_a_long_one_at_a_sentence():
    assert shape_quote(PARA) == PARA
    long = " ".join(f"Sentence number {i} says something useful here." for i in range(30))
    cut = shape_quote(long)
    assert 40 <= len(cut) <= 400 and cut.endswith(".") and long.startswith(cut)


def test_shaping_cuts_long_code_at_a_line():
    code = "```\n" + "\n".join(f"value_{i} = compute({i})" for i in range(60)) + "\n```"
    cut = shape_quote(code)
    assert len(cut) <= 400 and code.startswith(cut) and not cut.endswith("(")


def test_shaping_gives_nothing_when_there_is_no_place_to_cut():
    assert shape_quote("x" * 1000) is None


def test_hints():
    assert hint("```python\nx = 1\n```") == "code"
    assert hint("Call `retry()` with the `timeout` option.") == "code"
    assert hint("The default limit is 30 seconds per request.") == "number"
    assert hint("Retries happen again after a failure.") == "text"


# --- picking -------------------------------------------------------------------------------------

SOURCES = {"a/x.md": doc(40, "alpha"), "b/x.md": doc(10, "beta"), "c/x.md": doc(25, "gamma")}


def test_candidates_are_exact_slices_found_once():
    picks = pick_candidates(SOURCES, 12, seed=1, cap=12)
    assert len(picks) == 12
    for c in picks:
        text = SOURCES[c.source_id]
        assert text[c.start:c.end] == c.quote and text.count(c.quote) == 1 and 40 <= len(c.quote) <= 400
        assert c.hint in ("code", "number", "text")


def test_the_same_seed_gives_the_same_picks_and_another_seed_other_ones():
    one = pick_candidates(SOURCES, 12, seed=1, cap=12)
    assert one == pick_candidates(dict(reversed(list(SOURCES.items()))), 12, seed=1, cap=12)
    assert one != pick_candidates(SOURCES, 12, seed=2, cap=12)


def test_picks_for_one_source_do_not_change_when_another_source_changes():
    a = [c for c in pick_candidates(SOURCES, 12, seed=1, cap=12) if c.source_id == "a/x.md"]
    changed = {**SOURCES, "b/x.md": doc(10, "zeta")}
    assert a == [c for c in pick_candidates(changed, 12, seed=1, cap=12) if c.source_id == "a/x.md"]


def test_no_candidate_is_picked_twice_and_none_overlap():
    picks = pick_candidates({"a/x.md": doc(6, "alpha")}, 6, seed=3, cap=6)
    spans = sorted((c.start, c.end) for c in picks)
    assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:])) and len(spans) == len(set(spans))


def test_unusable_text_is_never_picked():
    sources = {"a/x.md": doc(3, "alpha") + "\n\n# Only headings\n## here too\n\n- [a](a.md)\n- [b](b.md)\n- [c](c.md)\n"}
    for c in pick_candidates(sources, 3, seed=1, cap=3):
        assert is_usable(c.quote) and "](" not in c.quote


def test_fewer_come_back_when_the_text_runs_out():
    picks = pick_candidates({"a/x.md": doc(2, "alpha")}, 10, seed=1, cap=10)
    assert 0 < len(picks) <= 2


def test_repeated_paragraphs_are_not_picked_because_they_are_not_unique():
    same = "\n\n".join([PARA] * 6) + "\n"
    assert pick_candidates({"a/x.md": same}, 3, seed=1, cap=3) == []


def test_the_pool_respects_the_per_source_cap():
    picks = pick_candidates({**SOURCES, "d/x.md": doc(200, "delta")}, 30, seed=1, cap=8)
    assert max(sum(c.source_id == s for c in picks) for s in {c.source_id for c in picks}) <= 8  # one file per source here

def test_allocation_gives_the_spare_picks_to_the_largest_fractions():
    assert allocate({"a": 4, "b": 9, "c": 16}, 10, cap=10) == {"a": 2, "b": 3, "c": 5}


def test_link_list_threshold_is_half_the_lines():
    half = "- [a](a.md)\n- [b](b.md)\n- Real sentence about retries stopping after three tries.\n- Another real sentence about delays growing."
    assert not is_usable(half)
    third = "- [a](a.md)\n- Real sentence about retries stopping after three tries.\n- Another real sentence about delays growing."
    assert is_usable(third)


def test_a_block_of_exactly_400_characters_is_kept_whole():
    text = "abcd " * 79 + "abcde"
    assert len(text) == 400 and shape_quote(text) == text


def test_a_sentence_ending_at_the_400th_character_is_used():
    text = "a" * 399 + ". next sentence goes on and on"
    assert shape_quote(text) == "a" * 399 + "."


def test_a_first_sentence_that_is_too_short_gives_nothing():
    assert shape_quote("Short one. " + "x" * 500) is None


def test_hint_edge_cases():
    assert hint("Use `retry` first.") == "code"
    assert hint("The default is 5.") == "number"
    assert hint("The default is chosen by the library.") == "text"


def test_two_sources_with_the_same_text_get_different_places():
    text = doc(40, "alpha")
    picks = pick_candidates({"a/x.md": text, "b/x.md": text}, 20, seed=1, cap=20)
    a = [c.start for c in picks if c.source_id == "a/x.md"]
    b = [c.start for c in picks if c.source_id == "b/x.md"]
    assert a and b and a != b

def test_a_register_source_gets_its_quota_whatever_the_number_of_files():
    files = {"p/one.md": doc(30, "one"), "p/two.md": doc(30, "two"), "q/only.md": doc(20, "only")}
    picks = pick_candidates(files, 20, seed=4, cap=20)
    quota = allocate({"p": len(files["p/one.md"]) + len(files["p/two.md"]), "q": len(files["q/only.md"])}, 20, cap=20)
    assert sum(group_of(c.source_id) == "p" for c in picks) == quota["p"]
    assert sum(group_of(c.source_id) == "q" for c in picks) == quota["q"]


def test_picks_spread_over_the_files_of_a_source_and_over_their_blocks():
    files = {"p/one.md": doc(30, "one"), "p/two.md": doc(30, "two")}
    picks = pick_candidates(files, 12, seed=4, cap=12)
    for name in files:
        starts = {c.start for c in picks if c.source_id == name}
        assert len(starts) >= 3


def test_a_bigger_file_gets_more_picks():
    files = {"p/big.md": doc(80, "bigone"), "p/small.md": doc(5, "small")}
    picks = pick_candidates(files, 10, seed=4, cap=10)
    assert sum(c.source_id == "p/big.md" for c in picks) > sum(c.source_id == "p/small.md" for c in picks)

def test_the_group_of_a_file_is_its_register_source():
    assert group_of("python-peps/pep-0008.rst") == "python-peps" and group_of("loose.md") == "loose.md"

def test_a_bare_number_of_two_digits_or_a_decimal_is_a_number_hint():
    assert hint("Set the limit to 4096 for long inputs.") == "number"
    assert hint("The ratio is 0.75 for most inputs.") == "number"
    assert hint("Version 3 is the one to use.") == "text"

def test_the_heading_above_a_place():
    text = "# Top\n\nintro\n\n## Retry options ##\n\nbody text\n\n### Deep one\n\nmore\n"
    assert heading_before(text, text.index("intro")) == "Top"
    assert heading_before(text, text.index("body")) == "Retry options"
    assert heading_before(text, text.index("more")) == "Deep one"
    assert heading_before("no headings here\n", 5) == ""
    assert heading_before(text, 0) == ""
    assert heading_before("#hashtag is not a heading\n\ntext\n", 30) == ""


def test_candidates_carry_the_heading_above_them():
    picks = pick_candidates({"a/x.md": doc(6, "alpha")}, 4, seed=1, cap=4)
    for c in picks:
        assert c.heading.startswith("Heading ") and c.heading == heading_before(doc(6, "alpha"), c.start)
