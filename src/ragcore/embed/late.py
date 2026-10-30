"""Late chunking (spec 13 section 4.3, unit 13h): the whole document goes through the model once, and each passage's
vector is the mean of the token vectors that lie inside it. Long documents go through in overlapping windows."""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ragcore.embed.base import normalise

DEFAULT_OVERLAP = 512
MARGIN = 16  # Tokens kept free below the model limit: a window is tokenised again on its own and adds special tokens.


@dataclass
class LateResult:
    vectors: np.ndarray  # (n passages, dim) float32; rows of fallback passages are zero.
    fallback_long: list[int] = field(default_factory=list)  # Longer than one window.
    fallback_empty: list[int] = field(default_factory=list)  # No whole token inside the span.
    windows: int = 0  # Forward passes made.
    window_tokens: int = 0  # Window size used.


def plan_windows(n_tokens: int, window: int, overlap: int) -> list[tuple[int, int]]:
    """Token ranges [start, end) of the windows: stride window - overlap, the last one ends exactly at the end."""
    if window <= 0 or overlap < 0 or overlap >= window:
        raise ValueError("the window must be above 0 and the overlap from 0 up to less than the window")
    if n_tokens <= 0:
        return []
    if n_tokens <= window:
        return [(0, n_tokens)]
    stride, start, out = window - overlap, 0, []
    while True:
        if start + window >= n_tokens:
            out.append((n_tokens - window, n_tokens))
            return out
        out.append((start, start + window))
        start += stride


def _distance(window: tuple[int, int], first: int, last: int) -> int:
    """Tokens of context on the thinner side of the passage (tokens first..last) inside this window.

    A window at the edge of the document is not treated specially: the neighbouring window is always thinner on that
    side, so the choice is the same."""
    return min(first - window[0], window[1] - last)


def late_vectors(embedder, text: str, spans: list[tuple[int, int]], *, window: int | None = None,
                 overlap: int = DEFAULT_OVERLAP) -> LateResult:
    for start, end in spans:
        if not (0 <= start < end <= len(text)):
            raise ValueError(f"span {start}..{end} is empty or outside the text (length {len(text)})")
    limit = embedder.max_tokens
    size = min(window or limit, limit) - MARGIN
    if size <= 0:
        raise ValueError(f"window of {min(window or limit, limit)} tokens leaves nothing after the safety margin of {MARGIN}")
    token_spans = embedder.token_spans(text)
    windows = plan_windows(len(token_spans), size, overlap)
    starts, ends = [s for s, _ in token_spans], [e for _, e in token_spans]
    result = LateResult(np.zeros((len(spans), embedder.dim), dtype=np.float32), window_tokens=size)
    chosen: dict[int, list[int]] = {}
    for i, (start, end) in enumerate(spans):
        first, last = bisect_left(starts, start), bisect_right(ends, end)
        if last <= first:
            result.fallback_empty.append(i)
        else:
            fitting = [w for w in range(len(windows)) if windows[w][0] <= first and last <= windows[w][1]]
            if not fitting:  # Longer than a window, or cut by every window boundary.
                result.fallback_long.append(i)
            else:
                chosen.setdefault(max(fitting, key=lambda w: (_distance(windows[w], first, last), -w)), []).append(i)
    for w, members in sorted(chosen.items()):
        whole = len(windows) == 1
        cut_start = 0 if whole else token_spans[windows[w][0]][0]
        cut_end = len(text) if whole else token_spans[windows[w][1] - 1][1]
        states = embedder.token_states(text[cut_start:cut_end])
        result.windows += 1
        offsets = np.asarray(states.offsets, dtype=np.int64).reshape(-1, 2) + cut_start
        for i in members:
            inside = (offsets[:, 0] >= spans[i][0]) & (offsets[:, 1] <= spans[i][1])
            if not inside.any():
                result.fallback_empty.append(i)
                continue
            result.vectors[i] = states.vectors[inside].mean(axis=0)
    result.vectors = normalise(result.vectors)
    return result


def embed_passages_late(embedder, sources, passages, *, window: int | None = None, overlap: int = DEFAULT_OVERLAP,
                        batch: int = 256, progress: Callable[[int, int], None] | None = None):
    """Vectors for all passages in input order, plus counts for the run manifest. Passages that cannot be pooled
    late (longer than a window, or without a whole token) are embedded the normal way and counted."""
    by_source = {s.source_id: s for s in sources}
    groups: dict[str, list[int]] = {}
    for i, p in enumerate(passages):
        if p.source_id not in by_source:
            raise ValueError(f"passage {p.passage_id} belongs to source {p.source_id!r}, which was not given")
        groups.setdefault(p.source_id, []).append(i)
    total = len(passages)
    out = np.zeros((total, embedder.dim), dtype=np.float32)
    stats = {"late_windows": 0, "late_fallback_too_long": 0, "late_fallback_no_tokens": 0,
             "late_window_tokens": 0, "late_overlap": overlap}
    redo: list[int] = []
    done = 0
    for source_id, members in groups.items():
        result = late_vectors(embedder, by_source[source_id].text,
                              [(passages[i].start, passages[i].end) for i in members], window=window, overlap=overlap)
        for row, i in enumerate(members):
            out[i] = result.vectors[row]
        redo += [members[row] for row in result.fallback_long + result.fallback_empty]
        stats["late_windows"] += result.windows
        stats["late_fallback_too_long"] += len(result.fallback_long)
        stats["late_fallback_no_tokens"] += len(result.fallback_empty)
        stats["late_window_tokens"] = result.window_tokens
        done += len(members) - len(result.fallback_long) - len(result.fallback_empty)
        if progress:
            progress(done, total)
    for pos in range(0, len(redo), batch):
        part = redo[pos:pos + batch]
        out[part] = embedder.embed([passages[i].text for i in part], kind="passage")
        done += len(part)
        if progress:
            progress(done, total)
    if progress and done != total:
        progress(total, total)
    return out, stats
