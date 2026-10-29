"""Semantic chunker (spec 13 section 4.2): cut where neighbouring blocks stop being alike, as judged by one embedder.

Blocks are the paragraphs and code blocks of each section. Distances are taken inside a section, never across a heading
(a heading always starts a new passage). A cut is made where the distance is above the chosen percentile of that
source's distances; the size limits of spec 09 then apply as usual."""
from __future__ import annotations

import hashlib

import numpy as np

from ragcore.chunking import (ChunkConfig, ChunkResult, _blocks, _finish, _only_headings, _scan, _sections,
                              heading_pack_spans)
from ragcore.contracts import Source

DEFAULT_PERCENTILE = 90


class SemanticChunker:
    """`SemanticChunker(embedder, percentile=90)(source, config)` returns a ChunkResult. Block vectors are kept by block
    text for the life of the object, so a block seen twice is embedded once."""

    def __init__(self, embedder, percentile: float = DEFAULT_PERCENTILE) -> None:
        if not (isinstance(percentile, (int, float)) and 0 < percentile < 100):
            raise ValueError(f"percentile must be above 0 and below 100, got {percentile!r}")
        self.embedder, self.percentile = embedder, percentile
        self._cache: dict[str, np.ndarray] = {}

    def version(self, config: ChunkConfig) -> str:
        fp = hashlib.sha256(self.embedder.fingerprint.encode("utf-8")).hexdigest()[:8]
        return f"semantic-1/{config.max_chars}/{config.hard_limit}/p{self.percentile:g}/{self.embedder.id}/{fp}"

    def _vectors(self, texts: list[str]) -> list[np.ndarray]:
        missing = sorted({t for t in texts if t not in self._cache})
        if missing:
            vectors = np.asarray(self.embedder.embed(missing, kind="passage"), dtype=np.float32)
            if vectors.shape[0] != len(missing) or not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-3):
                raise ValueError("the steering embedder must return one unit length vector per text")
            self._cache.update(zip(missing, vectors))
        return [self._cache[t] for t in texts]

    def breaks(self, text: str) -> frozenset[int]:
        """Block start offsets where the topic changes enough to start a new passage."""
        lines = _scan(text)
        runs: list[list[tuple[int, int]]] = []  # Blocks with something to say, per section.
        for _, first, end in _sections(lines):
            run = [(s, e) for s, e in _blocks(lines, first, end) if not _only_headings(text[s:e])]
            if run:
                runs.append(run)
        flat = [b for run in runs for b in run]
        vectors = dict(zip(flat, self._vectors([text[s:e] for s, e in flat])))
        pairs = [(a, b) for run in runs for a, b in zip(run, run[1:])]
        if not pairs:
            return frozenset()
        distances = [1.0 - float(vectors[a] @ vectors[b]) for a, b in pairs]
        threshold = float(np.percentile(distances, self.percentile))
        return frozenset(b[0] for (_, b), d in zip(pairs, distances) if d > threshold)

    def __call__(self, source: Source, config: ChunkConfig | None = None) -> ChunkResult:
        config = config or ChunkConfig()
        spans, dropped = heading_pack_spans(source.text, config, self.breaks(source.text))
        return _finish(source, config, spans, dropped, self.version(config), True)
