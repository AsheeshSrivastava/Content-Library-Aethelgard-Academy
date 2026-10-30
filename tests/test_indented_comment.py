"""An indented line that looks like a heading (a code comment inside an admonition) must never be dropped when it
ends up alone in a passage. Found on vllm-docs/custom_logitsprocs.md by the semantic chunker (spec 13 section 9i)."""
from ragcore.chunking import (ChunkConfig, _blocks, _finish, _scan, chunk_fixed, fixed_spans, heading_pack_spans)
from ragcore.contracts import Source
from ragcore.semantic import SemanticChunker
from ragcore.embed.base import FakeEmbedder

TEXT = ("# Doc\n\nIntro paragraph here.\n\n"
        "??? code \"Example\"\n\n    ``` python\n    # Import it\n    from a import b\n\n    # ...or...\n\n"
        "    # Define it locally\n    from c import d\n    ```\n\nAfter paragraph.\n")
CONFIG = ChunkConfig(max_chars=400, hard_limit=2000)


def source(text=TEXT):
    return Source("s1/a.md", "v1", text, "0" * 64, "md-1", "MIT", "declared")


def every_block_start(text):
    lines = _scan(text)
    return frozenset(s for s, _ in _blocks(lines, 0, len(lines)))


def covered(text, spans):
    """Every non-blank character of the text lies in some span."""
    inside = set()
    for s in spans:
        inside.update(range(s.start, s.end))
    return [i for i, ch in enumerate(text) if not ch.isspace() and i not in inside]


def test_indented_comment_alone_in_a_passage_is_kept():
    spans, _ = heading_pack_spans(TEXT, CONFIG, every_block_start(TEXT))
    assert any("# ...or..." in TEXT[s.start:s.end] for s in spans)
    lost = [i for i in covered(TEXT, spans) if TEXT[i] != "#" or True]
    # Only the top heading "# Doc" may be absent: it is a heading of the section, not text to search.
    assert all(i < len("# Doc") for i in lost)


def test_the_chunking_gate_accepts_the_result():
    spans, dropped = heading_pack_spans(TEXT, CONFIG, every_block_start(TEXT))
    _finish(source(), CONFIG, spans, dropped, "t-1", True)  # Raises ChunkingError when text is lost.


def test_real_heading_alone_is_still_dropped():
    text = "# A\n\n## B\n\nBody text here.\n"
    spans, dropped = heading_pack_spans(text, CONFIG, every_block_start(text))
    assert dropped >= 1 and all("Body" in text[s.start:s.end] for s in spans)


def test_fixed_windows_keep_an_indented_comment_too():
    config = ChunkConfig(max_chars=24, hard_limit=2000)
    text = "word " * 4 + "\n\n    # ...or...\n\nlast words here\n"
    spans, _ = fixed_spans(text, config)
    assert any("# ...or..." in text[s.start:s.end] for s in spans)
    chunk_fixed(source(text), config)  # The gate must accept it.


class EveryBlockApart(FakeEmbedder):
    """Pairs of neighbouring blocks all look different, so the semantic chunker cuts at the biggest gaps."""
    id = "apart"

    def embed(self, texts, *, kind):
        import numpy as np
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            out[i, hash(t) % self.dim] = 1.0
        return out


def test_semantic_chunker_survives_the_file_that_broke_it():
    result = SemanticChunker(EveryBlockApart(), percentile=1)(source(), CONFIG)
    assert any("# ...or..." in p.text for p in result.passages)
