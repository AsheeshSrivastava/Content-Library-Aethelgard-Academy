"""The retriever adapter (spec 17 section 4, ADR 0011): ask a question, get citable passages from the index file.

The ranking is the stage 3 `HR` pipeline itself (`retrieve.make_ranker`), so what was measured is what runs: vector top
200 and keyword top 200, rank fusion, the reranker on the first 30. The index is opened read-only and loaded into
memory once; open a new Retriever to see the result of a later sync."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ragcore.index_store import IndexSetup, IndexStore, IndexStoreError, open_index
from ragcore.retrieve import N_RAW, RERANK_DEPTH, make_ranker
from ragcore.runs import Run


@dataclass(frozen=True)
class Result:
    """One passage with what a caller needs to cite it. `score` is the reranker's: compare only within one search."""
    passage_id: str
    source_id: str
    url: str
    licence: str
    heading_path: tuple[str, ...]
    text: str
    start: int
    end: int
    rank: int
    score: float


class Retriever:
    def __init__(self, store: IndexStore, embedder, reranker, *, n_raw: int = N_RAW, depth: int = RERANK_DEPTH) -> None:
        if not store.read_only:
            raise IndexStoreError("the retriever needs an index opened read-only")
        self.store = store
        self._passages, matrix = store.load_all()
        self._sources = store.sources()
        self._rank = None
        if self._passages:
            rows = [{"passage_id": p.passage_id, "source_id": p.source_id, "start": p.start, "end": p.end,
                     "headings": list(p.heading_path), "parent": None} for p in self._passages]
            run = Run("index", {}, rows, matrix)
            self._rank = make_ranker("HR", run, [p.text for p in self._passages], embedder, reranker, None, n_raw, depth)
        self._open = True

    @classmethod
    def open(cls, path: Path | str, setup: IndexSetup, embedder, reranker, *, n_raw: int = N_RAW,
             depth: int = RERANK_DEPTH) -> Retriever:
        """Open the index file read-only (never creating it) and load it. A different setup is refused."""
        store = open_index(path, setup, read_only=True)
        try:
            return cls(store, embedder, reranker, n_raw=n_raw, depth=depth)
        except BaseException:
            store.close()
            raise

    def __enter__(self) -> Retriever:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._open = False
        self.store.close()

    def search(self, query: str, k: int = 5, max_per_file: int | None = None) -> list[Result]:
        """The `k` best passages, best first (ranks from 1). With `max_per_file`, no source appears more often than
        that. Only the passages the reranker looked at can be returned, so a limit can leave fewer than `k`."""
        if not self._open:
            raise IndexStoreError("the retriever is closed")
        if k < 1:
            raise ValueError(f"k must be at least 1, got {k}")
        if max_per_file is not None and max_per_file < 1:
            raise ValueError(f"max_per_file must be at least 1, got {max_per_file}")
        if not query.strip() or self._rank is None:
            return []
        results: list[Result] = []
        used: dict[str, int] = {}
        for position, score in self._rank(query):
            passage = self._passages[position]
            if max_per_file is not None and used.get(passage.source_id, 0) >= max_per_file:
                continue
            used[passage.source_id] = used.get(passage.source_id, 0) + 1
            source = self._sources[passage.source_id]
            results.append(Result(passage.passage_id, passage.source_id, source.url, source.licence,
                                  passage.heading_path, passage.text, passage.start, passage.end,
                                  len(results) + 1, score))
            if len(results) == k:
                break
        return results
