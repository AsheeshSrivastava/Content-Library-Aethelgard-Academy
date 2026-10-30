"""Reciprocal rank fusion (spec 15 section 3.2): merge ranked lists without needing comparable scores."""
from __future__ import annotations

K = 60


def rrf(lists: list[list[int]], k: int = K, n: int | None = None) -> list[tuple[int, float]]:
    """Rows from all lists, best first, each with its fused score: the sum of 1 / (k + rank) over the lists it is in
    (ranks from 1; a row repeated inside one list counts once, at its first rank). Equal scores (the same ranks in the
    same lists) go to the lower row position, so the order is repeatable."""
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    if n is not None and n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    score: dict[int, float] = {}
    for ranked in lists:
        seen: set[int] = set()
        for rank, row in enumerate(ranked, start=1):
            if row in seen:
                continue
            seen.add(row)
            score[row] = score.get(row, 0.0) + 1.0 / (k + rank)
    fused = sorted(score, key=lambda row: (-score[row], row))
    return [(row, score[row]) for row in fused[:n]]
