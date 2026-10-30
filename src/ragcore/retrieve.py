"""Retriever over a run (spec 13 section 4.5): exact cosine search, parent mapping, one hit per source file."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from ragcore.evalset import Hit, Retriever
from ragcore.fusion import rrf
from ragcore.lexical import Bm25Index
from ragcore.rerank import rerank_scored
from ragcore.runs import Run

N_RAW = 200


def search_passages(run: Run, query_vector: np.ndarray, n_raw: int = N_RAW) -> list[tuple[dict, float]]:
    """Best `n_raw` passages by cosine (vectors are unit length), best first. Children of one parent count once,
    with the best child's score; the child's row stands for the group."""
    if n_raw < 1:
        raise ValueError("n_raw must be at least 1")
    query = np.asarray(query_vector, dtype=np.float32)
    if query.ndim != 1 or query.shape[0] != run.vectors.shape[1]:
        raise ValueError(f"query width {query.shape} does not match the run's {run.vectors.shape[1]}")
    if not len(run.rows):
        return []
    scores = run.vectors @ query
    top = np.lexsort((np.arange(len(scores)), -scores))[:n_raw]  # best first; ties by row position, so repeatable
    seen, hits = set(), []
    for i in top:
        row = run.rows[int(i)]
        key = row["parent"] or row["passage_id"]
        if key in seen:
            continue
        seen.add(key)
        hits.append((row, float(scores[i])))
    return hits


def make_retriever(run: Run, embedder, n_raw: int = N_RAW) -> Retriever:
    def retrieve(query: str, n: int) -> list[Hit]:
        vector = embedder.embed([query], kind="query")[0]
        files, seen = [], set()
        for row, score in search_passages(run, vector, n_raw):
            if row["source_id"] not in seen:
                seen.add(row["source_id"])
                files.append(Hit(row["source_id"], score))
        return files[:n]
    return retrieve


# --- stage 3 pipelines (spec 15 section 3.4) ---------------------------------------------------

PIPELINES = ("V", "L", "H", "VR", "HR")
RERANK_DEPTH = 30


def keyword_index(run: Run, texts: list[str]) -> Bm25Index:
    """BM25 over each passage's heading path and text (spec 15 section 3.1)."""
    _check_texts(run, texts)
    return Bm25Index([" ".join(row["headings"]) + "\n" + text for row, text in zip(run.rows, texts)])


def _check_texts(run: Run, texts: list[str]) -> None:
    if len(texts) != len(run.rows):
        raise ValueError(f"texts has {len(texts)} entries but the run has {len(run.rows)} passages")


def _vector_positions(run: Run, query_vector: np.ndarray, n_raw: int) -> list[tuple[int, float]]:
    scores = run.vectors @ np.asarray(query_vector, dtype=np.float32)
    top = np.lexsort((np.arange(len(scores)), -scores))[:n_raw]
    return [(int(i), float(scores[i])) for i in top]


def _files(run: Run, ranked: list[tuple[int, float]], n: int) -> list[Hit]:
    """One hit per source file, in rank order, scored by its best passage."""
    seen, hits = set(), []
    for position, score in ranked:
        source_id = run.rows[position]["source_id"]
        if source_id not in seen:
            seen.add(source_id)
            hits.append(Hit(source_id, score))
    return hits[:n]


def _ranker(variant: str, run: Run, texts: list[str], embedder, reranker, index: Bm25Index | None, n_raw: int,
            depth: int) -> Callable[[str], list[tuple[int, float]]]:
    """query -> passages of the run (row position, score), best first, for one of the five variants."""
    if variant not in PIPELINES:
        raise ValueError(f"unknown variant {variant!r}; choose from {', '.join(PIPELINES)}")
    _check_texts(run, texts)
    if any(row["parent"] for row in run.rows):
        raise ValueError("these pipelines work on runs without parent passages; this run has parents")
    if depth < 1:
        raise ValueError(f"depth must be at least 1, got {depth}")
    if variant != "L" and embedder is None:
        raise ValueError(f"variant {variant} needs an embedder")
    if variant in ("VR", "HR") and reranker is None:
        raise ValueError(f"variant {variant} needs a reranker")
    if variant not in ("V", "VR") and index is None:
        index = keyword_index(run, texts)

    def ranked(query: str) -> list[tuple[int, float]]:
        found: list[tuple[int, float]] = []
        if variant != "L":
            found = _vector_positions(run, embedder.embed([query], kind="query")[0], n_raw)
        if variant in ("L", "H", "HR"):
            keywords = index.search(query, n_raw)
            found = keywords if variant == "L" else rrf([[p for p, _ in found], [p for p, _ in keywords]])
        if variant in ("VR", "HR"):
            first = [p for p, _ in found[:depth]]
            found = [(first[i], score) for i, score in rerank_scored(reranker, query, [texts[p] for p in first])]
        return found
    return ranked


def make_pipeline(variant: str, run: Run, texts: list[str], *, embedder=None, reranker=None,
                  index: Bm25Index | None = None, n_raw: int = N_RAW, depth: int = RERANK_DEPTH) -> Retriever:
    """V vectors only, L keywords only, H both fused, VR / HR the first `depth` passages of V / H reordered by the
    reranker. `texts` is the text of each passage of the run, in row order. Answers are files (spec 15 section 3.4)."""
    ranker = _ranker(variant, run, texts, embedder, reranker, index, n_raw, depth)
    if variant == "V":
        return make_retriever(run, embedder, n_raw)
    return lambda query, n: _files(run, ranker(query), n)


@dataclass(frozen=True)
class PassageHit:
    source_id: str
    start: int
    end: int
    score: float


def make_passage_pipeline(variant: str, run: Run, texts: list[str], *, embedder=None, reranker=None,
                          index: Bm25Index | None = None, n_raw: int = N_RAW, depth: int = RERANK_DEPTH
                          ) -> Callable[[str, int], list[PassageHit]]:
    """The same five variants, answering with passages (no collapsing to files) for the stage 4 questions."""
    ranker = _ranker(variant, run, texts, embedder, reranker, index, n_raw, depth)

    def retrieve(query: str, n: int) -> list[PassageHit]:
        return [PassageHit(run.rows[p]["source_id"], run.rows[p]["start"], run.rows[p]["end"], score)
                for p, score in ranker(query)[:n]]
    return retrieve