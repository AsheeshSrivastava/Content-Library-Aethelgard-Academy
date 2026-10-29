"""Retriever over a run (spec 13 section 4.5): exact cosine search, parent mapping, one hit per source file."""
from __future__ import annotations

import numpy as np

from ragcore.evalset import Hit, Retriever
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
