"""Spec 13 section 4.2: parent-child chunker. Children are paragraphs or code blocks (up to child_chars); the parent of a
child is the rules passage that holds it. Written before the code."""
import pytest

from ragcore.chunking import ChunkConfig, chunk_parent_child, chunk_source, check_chunking
from ragcore.contracts import Source

WORDS = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu "


def src(text, name="doc.md"):
    return Source(f"s/{name}", "v1", text, "0" * 64, "md-1", "MIT", "declared")


def paragraph(n_chars, seed="p"):
    out = f"{seed} "
    while len(out) < n_chars:
        out += WORDS
    return out[:n_chars].rstrip()


CFG = ChunkConfig(max_chars=1500, hard_limit=6000)


def test_each_short_paragraph_is_a_child_inside_one_parent():
    text = f"# H\n\n{paragraph(150, 'one')}\n\n{paragraph(150, 'two')}\n\n{paragraph(150, 'three')}\n"
    r = chunk_parent_child(src(text), CFG)
    assert [c.text[:3] for c in r.children] == ["one", "two", "thr"]
    assert len(r.parents) == 1
    assert {r.parent_of[c.passage_id] for c in r.children} == {r.parents[0].passage_id}


def test_children_are_exact_slices_in_order_and_pass_the_checks():
    source = src(f"# A\n\n{paragraph(300)}\n\n## B\n\n{paragraph(900)}\n\n```py\nx = 1\n\ny = 2\n```\n")
    r = chunk_parent_child(source, CFG)
    assert check_chunking(source, r.children, CFG) == []
    assert all(source.text[c.start:c.end] == c.text for c in r.children)
    assert [c.start for c in r.children] == sorted(c.start for c in r.children)


def test_children_never_exceed_child_chars_and_long_paragraphs_are_cut():
    r = chunk_parent_child(src(f"# A\n\n{paragraph(1300)}\n"), CFG, child_chars=500)
    assert len(r.children) >= 3 and all(len(c.text) <= 500 for c in r.children)
    assert len(r.parents) == 1 and len({r.parent_of[c.passage_id] for c in r.children}) == 1


def test_a_code_block_stays_one_child_even_with_a_blank_line_inside():
    code = "```py\nx = 1\n\ny = 2\n```"
    r = chunk_parent_child(src(f"# A\n\n{paragraph(100)}\n\n{code}\n"), CFG)
    assert code in [c.text for c in r.children]


def test_every_child_lies_inside_its_parent_and_has_its_headings():
    source = src(f"# A\n\n{paragraph(400)}\n\n## B\n\n{paragraph(400)}\n\n## C\n\n{paragraph(400)}\n")
    r = chunk_parent_child(source, ChunkConfig(max_chars=450, hard_limit=6000), child_chars=200)
    parents = {p.passage_id: p for p in r.parents}
    assert len(parents) == 3
    for c in r.children:
        p = parents[r.parent_of[c.passage_id]]
        assert p.start <= c.start and c.end <= p.end and c.headings == p.headings


def test_children_do_not_straddle_parents_when_a_paragraph_exceeds_the_hard_limit():
    source = src(f"# A\n\n{paragraph(1000)}\n")
    config = ChunkConfig(max_chars=150, hard_limit=300)
    r = chunk_parent_child(source, config, child_chars=100)
    parents = {p.passage_id: p for p in r.parents}
    assert len(parents) >= 3
    for c in r.children:
        p = parents[r.parent_of[c.passage_id]]
        assert p.start <= c.start and c.end <= p.end and len(c.text) <= 100


def test_parents_are_exactly_the_rules_passages():
    source = src(f"# A\n\n{paragraph(300)}\n\n## B\n\n{paragraph(300)}\n")
    r = chunk_parent_child(source, CFG)
    assert [p.passage_id for p in r.parents] == [p.passage_id for p in chunk_source(source, CFG).passages]


def test_versions_name_every_setting():
    r = chunk_parent_child(src(f"# A\n\n{paragraph(100)}\n"), CFG, child_chars=400)
    assert r.children[0].chunker_version == "parent-child-1/1500/6000/400"
    assert r.parents[0].chunker_version == CFG.version


def test_heading_only_blocks_are_dropped_and_counted():
    r = chunk_parent_child(src(f"# A\n\n## B\n\n{paragraph(100)}\n"), CFG)
    assert len(r.children) == 1 and r.dropped_empty >= 1


@pytest.mark.parametrize("bad", [0, -5, 6001])
def test_child_size_must_be_between_one_and_the_hard_limit(bad):
    with pytest.raises(ValueError, match="child_chars"):
        chunk_parent_child(src("# A\n\ntext here\n"), CFG, child_chars=bad)


def test_same_input_same_output():
    source = src(f"# A\n\n{paragraph(1300)}\n\n## B\n\n{paragraph(700)}\n")
    a, b = chunk_parent_child(source, CFG), chunk_parent_child(source, CFG)
    assert [c.passage_id for c in a.children] == [c.passage_id for c in b.children]
    assert a.parent_of == b.parent_of


def test_a_source_with_only_headings_has_no_children():
    r = chunk_parent_child(src("# A\n\n## B\n"), CFG)
    assert r.children == [] and r.parent_of == {}


def test_text_the_rules_chunker_dropped_gives_no_children():
    source = src(f"# A\n\n{paragraph(100, 'keep')}\n\n## B\n\n<!-- hidden -->\n")
    r = chunk_parent_child(source, CFG)
    assert [c.text[:4] for c in r.children] == ["keep"] and len(r.parents) == 1
    assert r.dropped_empty >= 1


def test_a_long_paragraph_in_a_dropped_free_document_loses_nothing():
    source = src(f"# A\n\n{paragraph(1000)}\n")
    r = chunk_parent_child(source, ChunkConfig(max_chars=150, hard_limit=300), child_chars=100)
    joined = " ".join(c.text for c in r.children)
    assert joined.split() == source.text.split("\n\n")[1].split()
