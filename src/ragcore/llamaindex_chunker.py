"""LlamaIndex splitter as a chunker variant (ADR 0007, spec 13 section 4.2). Needs `uv sync --group variants`.

LlamaIndex is used for boundaries only: `MarkdownNodeParser` then `SentenceSplitter` (no overlap) decide where
nodes start; each passage is the exact slice of our text from one node start to the next, so the contract
(exact slices, no overlap, no lost text) holds. A node whose text cannot be found in the source is counted, and its
boundary is left out; over the allowed share the run stops."""
from __future__ import annotations

import bisect

from ragcore.chunking import (ChunkConfig, ChunkResult, _blocks, _finish, _hard_cut, _only_headings, _scan, _sections,
                              _trim)
from ragcore.contracts import Source, Span

DEFAULT_CHUNK_TOKENS = 375
MAX_UNMAPPED_SHARE = 0.01


def llamaindex_version(config: ChunkConfig, chunk_tokens: int = DEFAULT_CHUNK_TOKENS) -> str:
    """The chunker version without importing LlamaIndex, so labels and run ids work without the `variants` group."""
    return f"llamaindex-1/{chunk_tokens}/{config.max_chars}/{config.hard_limit}"


class LlamaIndexChunker:
    def __init__(self, chunk_tokens: int = DEFAULT_CHUNK_TOKENS, max_unmapped_share: float = MAX_UNMAPPED_SHARE) -> None:
        if not isinstance(chunk_tokens, int) or chunk_tokens < 1:
            raise ValueError(f"chunk_tokens must be a whole number of at least 1, got {chunk_tokens!r}")
        try:
            from llama_index.core import Document
            from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
            from llama_index.core.schema import MetadataMode
        except ImportError as exc:
            raise RuntimeError("LlamaIndex is not installed: run `uv sync --group variants`") from exc
        self._document, self._mode = Document, MetadataMode.NONE
        self._markdown = MarkdownNodeParser()
        self._sentences = SentenceSplitter(chunk_size=chunk_tokens, chunk_overlap=0, include_metadata=False,
                                           include_prev_next_rel=False)
        self.chunk_tokens, self.max_unmapped_share = chunk_tokens, max_unmapped_share
        self.nodes = self.unmapped = 0

    def version(self, config: ChunkConfig) -> str:
        return llamaindex_version(config, self.chunk_tokens)

    @property
    def unmapped_share(self) -> float:
        return self.unmapped / self.nodes if self.nodes else 0.0

    def verify(self) -> None:
        if self.unmapped_share > self.max_unmapped_share:
            raise RuntimeError(f"{self.unmapped} of {self.nodes} LlamaIndex nodes ({self.unmapped_share:.1%}) could not be "
                               f"found in the source text; the limit is {self.max_unmapped_share:.0%}. "
                               "Check the LlamaIndex version before trusting this variant.")

    def _node_texts(self, text: str) -> list[str]:
        nodes = self._markdown.get_nodes_from_documents([self._document(text=text)])
        nodes = self._sentences.get_nodes_from_documents(nodes)
        return [n.get_content(metadata_mode=self._mode) for n in nodes]

    def starts(self, text: str) -> list[int]:
        """Where each node begins in `text`, found by searching forward from the end of the previous node."""
        found, cursor = [], 0
        for node_text in self._node_texts(text):
            node_text = node_text.strip()
            if not node_text:
                continue
            self.nodes += 1
            pos = text.find(node_text, cursor)
            if pos < 0:
                self.unmapped += 1
                continue
            found.append(pos)
            cursor = pos + len(node_text)
        return found

    def spans(self, text: str, config: ChunkConfig) -> tuple[list[Span], int]:
        bounds = sorted({0, *self.starts(text)})
        ends = bounds[1:] + [len(text)]
        lines = _scan(text)
        sections = _sections(lines)
        section_starts = [lines[first].start for _, first, _ in sections]
        block_ends = [b[1] for _, first, end in sections for b in _blocks(lines, first, end)]
        line_ends = [ln.body_end for ln in lines if not ln.blank]
        spans, dropped = [], 0
        for b_start, b_end in zip(bounds, ends):
            for s, e in _hard_cut(text, b_start, b_end, config.hard_limit, block_ends, line_ends):
                headings_only = _only_headings(text[s:e])
                s, e = _trim(text, s, e)
                if headings_only:
                    dropped += 1
                    continue
                k = bisect.bisect_right(section_starts, s) - 1
                spans.append(Span(s, e, sections[k][0] if k >= 0 else ()))
        return spans, dropped

    def __call__(self, source: Source, config: ChunkConfig | None = None) -> ChunkResult:
        config = config or ChunkConfig()
        spans, dropped = self.spans(source.text, config)
        return _finish(source, config, spans, dropped, self.version(config), False)
