"""Spec 13 section 4.2 and ADR 0007: LlamaIndex splitter used for boundaries only. Needs `uv sync --group variants`;
skipped when LlamaIndex is not installed. Written before the code."""
import pytest

pytest.importorskip("llama_index.core")

from ragcore.chunking import ChunkConfig, check_chunking  # noqa: E402
from ragcore.contracts import Source  # noqa: E402
from ragcore.llamaindex_chunker import LlamaIndexChunker  # noqa: E402

CFG = ChunkConfig(max_chars=1500, hard_limit=6000)
SENT = "The retriever returns the passages that are closest to the question. "


def src(text, name="d.md"):
    return Source(f"s/{name}", "v1", text, "0" * 64, "md-1", "MIT", "declared")


def squash(passages):
    return "".join("".join(p.text.split()) for p in passages)


def test_each_markdown_section_starts_a_passage_and_slices_are_exact():
    text = f"# One\n\n{SENT * 3}\n\n## Two\n\n{SENT * 3}\n\n## Three\n\n{SENT * 3}\n"
    source = src(text)
    r = LlamaIndexChunker(chunk_tokens=375)(source, CFG)
    assert [p.headings for p in r.passages] == [("One",), ("One", "Two"), ("One", "Three")]
    assert all(source.text[p.start:p.end] == p.text for p in r.passages)
    assert check_chunking(source, r.passages, CFG, structure=False) == []


def test_a_long_section_is_split_near_the_token_size():
    source = src("# One\n\n" + SENT * 120)
    r = LlamaIndexChunker(chunk_tokens=60)(source, CFG)
    assert len(r.passages) >= 4
    assert check_chunking(source, r.passages, CFG, structure=False) == []
    assert squash(r.passages) == "".join(source.text.split())


def test_no_text_is_lost_or_repeated():
    source = src(f"# A\n\n{SENT * 40}\n\n```py\nx = 1\n\ny = 2\n```\n\n## B\n\n{SENT * 40}\n")
    r = LlamaIndexChunker(chunk_tokens=50)(source, CFG)
    assert squash(r.passages) == "".join(source.text.split())


def test_version_names_the_size_and_limits():
    r = LlamaIndexChunker(chunk_tokens=60)(src("# A\n\ntext here\n"), CFG)
    assert r.passages[0].chunker_version == "llamaindex-1/60/1500/6000"
    assert LlamaIndexChunker(chunk_tokens=375).version(CFG) == "llamaindex-1/375/1500/6000"


@pytest.mark.parametrize("bad", [0, -1])
def test_chunk_tokens_must_be_positive(bad):
    with pytest.raises(ValueError, match="chunk_tokens"):
        LlamaIndexChunker(chunk_tokens=bad)


def test_hard_limit_is_enforced_when_a_node_is_too_long():
    config = ChunkConfig(max_chars=200, hard_limit=300)
    source = src("# A\n\n" + "word " * 400)
    r = LlamaIndexChunker(chunk_tokens=5000)(source, config)
    assert len(r.passages) > 1 and all(len(p.text) <= 300 for p in r.passages)
    assert check_chunking(source, r.passages, config, structure=False) == []


def test_heading_only_text_gives_nothing():
    assert LlamaIndexChunker()(src("# A\n\n## B\n"), CFG).passages == []


def test_repeatable():
    source = src(f"# A\n\n{SENT * 50}\n\n## B\n\n{SENT * 50}\n")
    a, b = LlamaIndexChunker(chunk_tokens=80)(source, CFG), LlamaIndexChunker(chunk_tokens=80)(source, CFG)
    assert [p.passage_id for p in a.passages] == [p.passage_id for p in b.passages]


class Rewriting(LlamaIndexChunker):
    """A splitter whose node text no longer matches the source, as if LlamaIndex had normalised it."""
    def _node_texts(self, text):
        return [t if i % 2 == 0 else t + " (rewritten)" for i, t in enumerate(super()._node_texts(text))]


def test_nodes_that_cannot_be_found_are_counted_and_no_text_is_lost():
    source = src(f"# A\n\n{SENT * 60}\n")
    chunker = Rewriting(chunk_tokens=40, max_unmapped_share=1.0)
    r = chunker(source, CFG)
    assert chunker.unmapped > 0 and chunker.nodes > chunker.unmapped
    assert squash(r.passages) == "".join(source.text.split())
    assert check_chunking(source, r.passages, CFG, structure=False) == []


def test_too_many_unmapped_nodes_stops_the_run():
    chunker = Rewriting(chunk_tokens=40)
    chunker(src(f"# A\n\n{SENT * 60}\n"), CFG)
    with pytest.raises(RuntimeError, match="1%"):
        chunker.verify()


def test_few_unmapped_nodes_pass_verify():
    chunker = LlamaIndexChunker(chunk_tokens=40)
    chunker(src(f"# A\n\n{SENT * 60}\n"), CFG)
    assert chunker.unmapped == 0 and chunker.unmapped_share == 0.0
    chunker.verify()


def test_a_clear_message_when_llamaindex_is_missing(monkeypatch):
    import builtins
    real = builtins.__import__

    def fake(name, *a, **k):
        if name.startswith("llama_index"):
            raise ImportError("no llama")
        return real(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", fake)
    with pytest.raises(RuntimeError, match="variants"):
        LlamaIndexChunker()


class FirstNodeRewritten(LlamaIndexChunker):
    def _node_texts(self, text):
        texts = super()._node_texts(text)
        return ["not in the source at all"] + texts[1:]


def test_text_before_the_first_found_node_is_not_lost():
    distinct = " ".join(f"Sentence number {i} explains one retrieval idea." for i in range(60))  # No repeats to find early.
    source = src(f"{distinct}\n\n## B\n\n{SENT * 30}\n")  # Content first, so node 0 is real text.
    chunker = FirstNodeRewritten(chunk_tokens=50, max_unmapped_share=1.0)
    r = chunker(source, CFG)
    assert chunker.unmapped == 1 and squash(r.passages) == "".join(source.text.split())


def test_the_last_character_is_kept_when_the_text_has_no_final_newline():
    source = src("# A\n\nfirst paragraph here\n\nlast word")
    r = LlamaIndexChunker()(source, CFG)
    assert r.passages[-1].text.endswith("last word")


class BlankNodes(LlamaIndexChunker):
    def _node_texts(self, text):
        return ["", "   ", *super()._node_texts(text), "\n"]


def test_blank_nodes_are_ignored_and_not_counted():
    source = src(f"# A\n\n{SENT * 3}\n")
    plain, blanky = LlamaIndexChunker(), BlankNodes()
    plain(source, CFG), blanky(source, CFG)
    assert blanky.nodes == plain.nodes and blanky.unmapped == 0
