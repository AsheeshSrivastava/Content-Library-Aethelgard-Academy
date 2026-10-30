"""Random quote places for the passage-level questions (spec 16 section 3).

Places are drawn at random from the source text itself, never from any chunker's passages, so no chunker is favoured.
A fixed seed makes the draw repeatable, and each source has its own random stream, so changing one source does not move
the picks of another."""
from __future__ import annotations

import bisect
import math
import random
import re
from dataclasses import dataclass

QUOTE_MIN, QUOTE_MAX = 40, 400
TRIES_PER_PICK = 40
MIN_ALNUM_SHARE = 0.5
_FENCE = re.compile(r"^\s*(```|~~~)")
_SENTENCE_END = re.compile(r"[.!?](?=\s)")
_LICENCE = ("licensed under", "permission is hereby granted", "spdx-license-identifier", "all rights reserved")
_BARE_NUMBER = re.compile(r"\b\d{2,}\b|\b\d+\.\d+\b")
_NUMBER = re.compile(r"\b\d+(\.\d+)?\s?(ms|s|sec|seconds|minutes|mb|gb|kb|%|bytes|characters|times|tokens)\b", re.I)


@dataclass(frozen=True)
class Candidate:
    source_id: str
    start: int
    end: int
    quote: str
    hint: str  # code, number or text: a guide for choosing a mix, not a label


def allocate(sizes: dict[str, int], total: int, cap: int) -> dict[str, int]:
    """Split `total` picks over sources in proportion to the square root of their size, none above `cap`.
    Gives fewer than `total` when every source is at the cap."""
    if cap < 1:
        raise ValueError(f"cap must be at least 1, got {cap}")
    if total < 0:
        raise ValueError(f"total must be 0 or more, got {total}")
    weights = {s: math.sqrt(max(0, n)) for s, n in sizes.items()}
    out: dict[str, int] = {}
    open_ = dict(weights)
    left = total
    while open_:
        scale = left / sum(open_.values()) if sum(open_.values()) else 0.0
        over = [s for s, w in open_.items() if w * scale > cap]
        if not over:
            break
        for s in over:
            out[s] = cap
            left -= cap
            del open_[s]
    if open_:
        scale = left / sum(open_.values()) if sum(open_.values()) else 0.0
        shares = {s: w * scale for s, w in open_.items()}
        for s, share in shares.items():
            out[s] = int(share)
        spare = left - sum(int(v) for v in shares.values())
        for s in sorted(shares, key=lambda s: (-(shares[s] - int(shares[s])), s))[:max(0, spare)]:
            out[s] += 1
    return {s: out.get(s, 0) for s in sorted(sizes)}


def blocks(text: str) -> list[tuple[int, int]]:
    """Spans of the blank-line-separated blocks of a text. A code fence is one block, blank lines inside it included."""
    spans, start, fence, pos = [], None, False, 0
    for line in text.splitlines(keepends=True):
        body = line.rstrip("\r\n")
        if _FENCE.match(body):
            fence = not fence
            if start is None:
                start = pos
            end = pos + len(body)
        elif body.strip() or fence:
            if start is None:
                start = pos
            end = pos + len(body)
        elif start is not None:
            spans.append((start, end))
            start = None
        pos += len(line)
    if start is not None:
        spans.append((start, end))
    return spans


def is_usable(block: str) -> bool:
    text = block.strip()
    if len(text) < QUOTE_MIN:
        return False
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if all(ln.startswith("#") for ln in lines):
        return False
    if sum("](" in ln and ln.lstrip("-*+ ").startswith("[") for ln in lines) * 2 >= len(lines):
        return False
    lowered = text.lower()
    if any(mark in lowered for mark in _LICENCE):
        return False
    return sum(c.isalnum() for c in text) / len(text) >= MIN_ALNUM_SHARE


def shape_quote(block: str) -> str | None:
    """The block, or its first part cut at a sentence end (or a line end) so that it is at most 400 characters."""
    text = block.strip()
    if len(text) <= QUOTE_MAX:
        return text
    window = text[:QUOTE_MAX + 1]
    ends = [m.end() for m in _SENTENCE_END.finditer(window) if m.end() <= QUOTE_MAX]
    cut = ends[-1] if ends else window.rfind("\n")
    piece = text[:cut].rstrip() if cut > 0 else ""
    return piece if len(piece) >= QUOTE_MIN else None


def hint(quote: str) -> str:
    if "```" in quote or quote.count("`") >= 2:
        return "code"
    if _NUMBER.search(quote) or _BARE_NUMBER.search(quote) or ("default" in quote.lower() and re.search(r"\d", quote)):
        return "number"
    return "text"


def group_of(source_id: str) -> str:
    """The register source a file belongs to: the first part of its id (`python-peps/pep-0008.rst` -> `python-peps`)."""
    return source_id.split("/", 1)[0]


def pick_candidates(sources: dict[str, str], total: int, seed: int, cap: int) -> list[Candidate]:
    """About `total` quote places drawn at random, spread over the register sources by the square root of their size
    (none above `cap`) and, inside a source, over its files by size. Fewer come back when a source runs out of usable
    unique text."""
    groups: dict[str, list[str]] = {}
    for source_id in sorted(sources):
        groups.setdefault(group_of(source_id), []).append(source_id)
    quotas = allocate({g: sum(len(sources[s]) for s in ids) for g, ids in groups.items()}, total, cap)
    out: list[Candidate] = []
    for group in groups:  # in file-id order, which is also register-source order
        ids = groups[group]
        spans = {s: blocks(sources[s]) for s in ids}
        starts = {s: [a for a, _ in spans[s]] for s in ids}
        first, length = [], 0
        for s in ids:
            first.append(length)
            length += len(sources[s])
        want, rng = quotas[group], random.Random(f"{seed}:{group}")
        taken: set[tuple[str, int]] = set()
        found = 0
        for _ in range(want * TRIES_PER_PICK):
            if found >= want or not length:
                break
            place = rng.randrange(length)
            source_id = ids[bisect.bisect_right(first, place) - 1]
            text = sources[source_id]
            k = bisect.bisect_right(starts[source_id], place - first[ids.index(source_id)]) - 1
            if k < 0 or (source_id, k) in taken:
                continue
            start, end = spans[source_id][k]
            if not is_usable(text[start:end]):
                continue
            quote = shape_quote(text[start:end])
            if quote is None or text.count(quote) != 1:
                continue
            taken.add((source_id, k))
            found += 1
            out.append(Candidate(source_id, start, start + len(quote), quote, hint(quote)))
    return out