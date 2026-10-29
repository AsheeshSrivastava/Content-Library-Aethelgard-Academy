"""Spec 13 section 4.2: semantic chunker. Blocks are embedded by the steering embedder; a passage is cut where the
distance between neighbouring blocks is above the chosen percentile of that source's distances. Written before the code."""
import numpy as np
import pytest

from ragcore.chunking import ChunkConfig, check_chunking, chunk_source
from ragcore.contracts import Source
from ragcore.embed.base import FakeEmbedder
from ragcore.semantic import SemanticChunker

A = "apple banana cherry grape melon peach plum pear "
B = "quartz basalt granite obsidian marble slate flint chalk "
C = "violin cello oboe flute harp trumpet tuba piano "
CFG = ChunkConfig(max_chars=1500, hard_limit=6000)


def src(text, name="d.md"):
    return Source(f"s/{name}", "v1", text, "0" * 64, "md-1", "MIT", "declared")


def doc(*blocks, heading="# H"):
    return heading + "\n\n" + "\n\n".join(b.strip() for b in blocks) + "\n"


class Counting(FakeEmbedder):
    def __init__(self):
        super().__init__()
        self.texts = []

    def embed(self, texts, *, kind):
        self.texts += list(texts)
        return super().embed(texts, kind=kind)


def chunk(text, percentile=90, embedder=None, config=CFG, chunker=None):
    chunker = chunker or SemanticChunker(embedder or FakeEmbedder(), percentile=percentile)
    return chunker(src(text), config)


def test_cuts_where_the_topic_changes():
    text = doc(A * 3, A * 3, A * 3, B * 3, B * 3, B * 3)
    r = chunk(text)
    assert len(r.passages) == 2
    assert "quartz" not in r.passages[0].text and "apple" not in r.passages[1].text


def test_no_cut_when_every_block_is_alike():
    assert len(chunk(doc(A * 3, A * 3, A * 3, A * 3)).passages) == 1


def test_headings_always_cut_even_for_identical_text():
    text = f"# One\n\n{A * 3}\n\n# Two\n\n{A * 3}\n"
    r = chunk(text)
    assert len(r.passages) == 2 and [p.headings for p in r.passages] == [("One",), ("Two",)]


def test_distances_are_taken_inside_sections_only():
    # Section 1 has one real shift (A to B); section 2 is C, C. A cross-section distance would raise the threshold.
    text = f"# One\n\n{A * 3}\n\n{A * 3}\n\n{B * 3}\n\n# Two\n\n{A * 3}\n\n{A * 3}\n"
    texts = [p.text for p in chunk(text).passages]
    assert len(texts) == 3 and "apple" in texts[0] and "quartz" in texts[1] and "quartz" not in texts[2]


def test_a_code_block_is_never_split_and_is_embedded_whole():
    code = "```py\nx = 1\n\ny = 2\n```"
    counting = Counting()
    r = chunk(doc(A * 3, code, A * 3), embedder=counting)
    assert code in counting.texts
    assert any(code in p.text for p in r.passages)


def test_the_size_limits_still_apply():
    text = doc(*[A * 4] * 20)
    r = chunk(text, config=ChunkConfig(max_chars=400, hard_limit=1000))
    assert len(r.passages) > 1 and all(len(p.text) <= 400 for p in r.passages)


def test_lower_percentile_cuts_more():
    text = doc(A * 3, B * 3, C * 3, A * 3, B * 3, C * 3, A * 3, A * 3, B * 3)
    assert len(chunk(text, percentile=50).passages) >= len(chunk(text, percentile=90).passages)
    assert len(chunk(text, percentile=10).passages) > len(chunk(text, percentile=95).passages)


@pytest.mark.parametrize("bad", [0, 100, -1, 101, float("nan")])
def test_percentile_must_be_strictly_between_0_and_100(bad):
    with pytest.raises(ValueError, match="percentile"):
        SemanticChunker(FakeEmbedder(), percentile=bad)


def test_version_names_percentile_and_steering_model():
    r = chunk(doc(A * 3, B * 3))
    v = r.passages[0].chunker_version
    assert v.startswith("semantic-1/1500/6000/p90/fake/")
    other = SemanticChunker(FakeEmbedder(dim=16), percentile=90)(src(doc(A * 3, B * 3)), CFG).passages[0].chunker_version
    assert other != v and other.startswith("semantic-1/1500/6000/p90/fake/")
    assert chunk(doc(A * 3, B * 3), percentile=80).passages[0].chunker_version != v


def test_repeated_blocks_are_embedded_once():
    counting = Counting()
    chunker = SemanticChunker(counting, percentile=90)
    chunker(src(doc(A * 3, B * 3, C * 3), "a.md"), CFG)
    first = len(counting.texts)
    chunker(src(doc(A * 3, B * 3, C * 3), "b.md"), CFG)
    assert len(counting.texts) == first


def test_result_passes_the_checks_and_is_repeatable():
    source = src(doc(A * 3, A * 3, B * 3, B * 3, C * 3))
    chunker = SemanticChunker(FakeEmbedder(), percentile=90)
    r1, r2 = chunker(source, CFG), SemanticChunker(FakeEmbedder(), percentile=90)(source, CFG)
    assert check_chunking(source, r1.passages, CFG) == []
    assert [p.passage_id for p in r1.passages] == [p.passage_id for p in r2.passages]


def test_a_single_block_source_is_one_passage_without_error():
    assert len(chunk(doc(A * 3)).passages) == 1


def test_heading_only_text_gives_nothing():
    assert chunk("# A\n\n## B\n").passages == []


def test_the_rules_chunker_is_unchanged_by_the_new_hook():
    source = src(doc(A * 3, B * 3, C * 3))
    assert [p.text for p in chunk_source(source, CFG).passages] == [source.text.strip()]


def test_blocks_that_are_only_headings_do_not_take_part_in_distances():
    text = f"# H\n\n{A * 3}\n\n## Sub\n\n{A * 3}\n"
    counting = Counting()
    chunk(text, embedder=counting)
    assert all(t.strip() != "## Sub" for t in counting.texts)


def test_embedder_vectors_must_be_unit_length_floats():
    class Bad(FakeEmbedder):
        def embed(self, texts, *, kind):
            return np.ones((len(texts), self.dim), dtype=np.float32) * 5
    with pytest.raises(ValueError, match="unit length"):
        SemanticChunker(Bad())(src(doc(A * 3, B * 3, C * 3)), CFG)
