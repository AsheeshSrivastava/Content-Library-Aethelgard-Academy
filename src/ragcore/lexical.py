"""Keyword index (spec 15 section 3.1, ADR 0009): tokens that keep code names intact, and BM25 over them."""
from __future__ import annotations

import math
import re

import numpy as np

K1 = 1.2
B = 0.75
_WORD = re.compile(r"[A-Za-z0-9_.]+")
_PIECE = re.compile(r"[A-Za-z0-9]+")
_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+")


def tokenize(text: str) -> list[str]:
    """Lower-case words and numbers. A name with `_`, `.` or a camelCase change is kept whole and also gives its parts:
    `load_snapshot` -> load_snapshot, load, snapshot. No stemming, no stop words."""
    tokens: list[str] = []
    for match in _WORD.finditer(text):
        word = match.group().strip(".")
        if not word:
            continue
        parts = [p.lower() for piece in _PIECE.findall(word) for p in _CAMEL.findall(piece)]
        if not parts:
            continue
        whole = word.lower()
        tokens += parts if parts == [whole] else [whole, *parts]
    return tokens


class Bm25Index:
    """BM25 over documents given as text. `search` returns (document position, score) best first; ties by position."""

    def __init__(self, documents: list[str], k1: float = K1, b: float = B) -> None:
        if k1 < 0:
            raise ValueError(f"k1 must be 0 or more, got {k1}")
        if not 0 <= b <= 1:
            raise ValueError(f"b must be from 0 to 1, got {b}")
        self.k1, self.b, self.size = k1, b, len(documents)
        postings: dict[str, tuple[list[int], list[int]]] = {}
        lengths = np.zeros(len(documents), dtype=np.float64)
        for position, document in enumerate(documents):
            counts: dict[str, int] = {}
            for token in tokenize(document):
                counts[token] = counts.get(token, 0) + 1
            lengths[position] = sum(counts.values())
            for token, tf in counts.items():
                rows, freqs = postings.setdefault(token, ([], []))
                rows.append(position)
                freqs.append(tf)
        self._postings = {t: (np.array(r, dtype=np.int64), np.array(f, dtype=np.float64)) for t, (r, f) in postings.items()}
        average = float(lengths.mean()) if len(lengths) else 0.0
        self._norm = 1.0 - b + b * (lengths / average) if average > 0 else np.ones(len(documents))

    def search(self, query: str, n: int = 200) -> list[tuple[int, float]]:
        if n < 1:
            raise ValueError(f"n must be at least 1, got {n}")
        scores = np.zeros(self.size, dtype=np.float64)
        for term in dict.fromkeys(tokenize(query)):  # Each query term counts once.
            posting = self._postings.get(term)
            if posting is None:
                continue
            rows, tf = posting
            idf = math.log(1.0 + (self.size - len(rows) + 0.5) / (len(rows) + 0.5))
            scores[rows] += idf * tf * (self.k1 + 1.0) / (tf + self.k1 * self._norm[rows])
        order = np.lexsort((np.arange(self.size), -scores))[:n]
        return [(int(i), float(scores[i])) for i in order if scores[i] > 0]
