"""Heading-aware chunking and the exact-slice check (spec 09).

A chunker only decides spans; `contracts.make_passages` turns them into passages that are exact slices of the
source text. `check_chunking` is written against Source and Passage only, so every chunker is held to it."""
from __future__ import annotations

import bisect
import re
from dataclasses import dataclass
from typing import Callable

from ragcore.contracts import Passage, Source, Span, content_sha256, make_passage_id, make_passages, verify_passage

HEADING = re.compile(r"^[ ]{0,3}(#{1,6})[ \t]+(.+?)(?:[ \t]+#+)?[ \t]*$")
FENCE = re.compile(r"^[ ]{0,3}(`{3,}|~{3,})(.*)$")
HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)

Chunker = Callable[[Source], list[Span]]


class ChunkingError(ValueError):
    """A chunker produced passages that fail the checks (spec 09)."""


@dataclass(frozen=True)
class ChunkConfig:
    max_chars: int = 1500  # Provisional: tune with the question test, not by guessing.
    hard_limit: int = 6000  # No passage may be longer; the embedding model refuses text over about 8000 tokens.

    def __post_init__(self) -> None:
        if not 0 < self.max_chars <= self.hard_limit:
            raise ValueError(f"need 0 < max_chars <= hard_limit, got {self.max_chars} and {self.hard_limit}")

    @property
    def version(self) -> str:
        return f"heading-pack-1/{self.max_chars}/{self.hard_limit}"


@dataclass
class ChunkResult:
    passages: list[Passage]
    dropped_empty: int = 0


@dataclass(frozen=True)
class _Line:
    start: int
    body_end: int  # End without the newline.
    end: int  # End with the newline.
    fenced: bool
    heading: tuple[int, str] | None

    @property
    def blank(self) -> bool:
        return not self.fenced and self.body_end == self.start


def _scan(text: str) -> list[_Line]:
    lines: list[_Line] = []
    fence = ""
    offset = 0
    for raw in text.splitlines(keepends=True):
        body = raw.rstrip("\n")
        m = FENCE.match(body)
        fenced = False
        if fence:
            fenced = True
            if m and m.group(1)[0] == fence[0] and len(m.group(1)) >= len(fence) and not m.group(2).strip():
                fence = ""
        elif m and not (m.group(1)[0] == "`" and "`" in m.group(2)):
            fence, fenced = m.group(1), True
        heading = None
        if not fenced and (h := HEADING.match(body)):
            heading = (len(h.group(1)), h.group(2).strip())
        blank = not fenced and not body.strip()
        lines.append(_Line(offset, offset + (0 if blank else len(body)), offset + len(raw), fenced, heading))
        offset += len(raw)
    return lines


def _only_headings(text: str) -> bool:
    """Nothing to search for or cite: blank, or only headings and HTML comments."""
    rest = HTML_COMMENT.sub("", text)
    return all(HEADING.match(line) for line in rest.splitlines() if line.strip())


def _sections(lines: list[_Line]) -> list[tuple[tuple[str, ...], int, int]]:
    """(heading path, first line, end line) per section; a skipped heading level contributes nothing."""
    starts, path = [(0, ())], []
    for i, line in enumerate(lines):
        if line.heading:
            level, title = line.heading
            path = path[:level - 1] + [""] * (level - 1 - len(path[:level - 1])) + [title]
            starts.append((i, tuple(p for p in path if p)))
    sections = []
    for k, (first, headings) in enumerate(starts):
        end = starts[k + 1][0] if k + 1 < len(starts) else len(lines)
        if end > first:
            sections.append((headings, first, end))
    return sections


def _blocks(lines: list[_Line], first: int, end: int) -> list[tuple[int, int]]:
    """Paragraph blocks (start, end offsets) of a section: runs of lines split at blank lines outside fences."""
    blocks, begin, last, in_fence = [], None, None, False
    for line in lines[first:end]:
        if line.blank:
            if begin is not None:
                blocks.append((begin, last))
                begin = None
        else:
            if begin is not None and line.fenced != in_fence:  # A fence is its own block even without blank lines.
                blocks.append((begin, last))
                begin = None
            if begin is None:
                begin = line.start
                in_fence = line.fenced
            last = line.body_end
    if begin is not None:
        blocks.append((begin, last))
    return blocks


def _cut_point(text: str, pos: int, window_end: int, block_ends: list[int], line_ends: list[int]) -> int:
    def usable(c: int) -> bool:
        return c > pos and not _only_headings(text[pos:c])

    for candidates in (block_ends, line_ends):
        fits = [c for c in candidates if pos < c <= window_end and usable(c)]
        if fits:
            return max(fits)
    for c in range(window_end, pos + (window_end - pos) // 2, -1):
        if text[c].isspace() and usable(c):
            return c
    return window_end


def _hard_cut(text: str, start: int, end: int, limit: int, block_ends: list[int], line_ends: list[int]) -> list[tuple[int, int]]:
    parts, pos = [], start
    while end - pos > limit:
        cut = _cut_point(text, pos, pos + limit, block_ends, line_ends)
        parts.append((pos, cut))
        pos = cut
        while pos < end and text[pos].isspace():
            pos += 1
    parts.append((pos, end))
    return parts


def heading_pack_spans(text: str, config: ChunkConfig) -> tuple[list[Span], int]:
    lines = _scan(text)
    spans, dropped = [], 0
    for headings, first, end in _sections(lines):
        blocks = _blocks(lines, first, end)
        pieces, begin, last, sticky = [], None, None, False
        for b_start, b_end in blocks:
            if begin is not None and b_end - begin > config.max_chars and not sticky:
                pieces.append((begin, last))
                begin = None
            if begin is None:
                begin = b_start
            last = b_end
            sticky = all(ln.heading for ln in lines[first:end] if ln.start >= b_start and ln.body_end <= b_end and not ln.blank)
        if begin is not None:
            pieces.append((begin, last))
        block_ends = [b[1] for b in blocks]
        line_ends = [ln.body_end for ln in lines[first:end] if not ln.blank]
        for p_start, p_end in pieces:
            for s, e in _hard_cut(text, p_start, p_end, config.hard_limit, block_ends, line_ends):
                s, e = _trim(text, s, e)
                if _only_headings(text[s:e]):
                    dropped += 1
                    continue
                spans.append(Span(s, e, headings))
    return spans, dropped


def _finish(source: Source, config: ChunkConfig, spans: list[Span], dropped: int, version: str,
            structure: bool) -> ChunkResult:
    passages = make_passages(source, spans, chunker_version=version)
    problems = check_chunking(source, passages, config, structure=structure)
    if problems:
        raise ChunkingError(f"{source.source_id}: " + "; ".join(problems[:5]))
    return ChunkResult(passages, dropped)


def chunk_source(source: Source, config: ChunkConfig | None = None, chunker: Chunker | None = None, *,
                 version: str | None = None, structure: bool = True) -> ChunkResult:
    """Chunk with `heading-pack`, or with `chunker` (spans only). `structure=False` is for chunkers that do not
    follow headings: the checks that are about headings are skipped, all others stay."""
    config = config or ChunkConfig()
    if chunker is None:
        spans, dropped = heading_pack_spans(source.text, config)
    else:
        spans, dropped = chunker(source), 0
    return _finish(source, config, spans, dropped, version or config.version, structure)


def _trim(text: str, s: int, e: int) -> tuple[int, int]:
    piece = text[s:e]
    return s + len(piece) - len(piece.lstrip()), e - (len(piece) - len(piece.rstrip()))


def fixed_spans(text: str, config: ChunkConfig) -> tuple[list[Span], int]:
    """Windows of at most `max_chars`, cut where the spec 09 hard-cut rules prefer. Headings play no part in the cut;
    the heading path is the one in force where a passage starts."""
    lines = _scan(text)
    sections = _sections(lines)
    starts = [lines[first].start for _, first, _ in sections]
    blocks = _blocks(lines, 0, len(lines))
    block_ends = [b[1] for b in blocks]
    line_ends = [ln.body_end for ln in lines if not ln.blank]
    spans, dropped = [], 0
    for s, e in _hard_cut(text, 0, len(text), config.max_chars, block_ends, line_ends):
        s, e = _trim(text, s, e)
        if _only_headings(text[s:e]):
            dropped += 1
            continue
        k = bisect.bisect_right(starts, s) - 1
        spans.append(Span(s, e, sections[k][0] if k >= 0 else ()))
    return spans, dropped


def merge_small_spans(text: str, config: ChunkConfig, min_chars: int) -> tuple[list[Span], int]:
    """`heading-pack` passages, then a passage under `min_chars` takes its right neighbour in the same top-level
    section while the join stays within `max_chars`."""
    spans, dropped = heading_pack_spans(text, config)
    merged: list[Span] = []
    for span in spans:
        cur = merged[-1] if merged else None
        if (cur is not None and cur.end - cur.start < min_chars and cur.headings[:1] == span.headings[:1]
                and span.end - cur.start <= config.max_chars):
            merged[-1] = Span(cur.start, span.end, cur.headings)
        else:
            merged.append(span)
    return merged, dropped


def chunk_fixed(source: Source, config: ChunkConfig | None = None) -> ChunkResult:
    config = config or ChunkConfig()
    spans, dropped = fixed_spans(source.text, config)
    return _finish(source, config, spans, dropped, f"fixed-1/{config.max_chars}/{config.hard_limit}", False)


def chunk_merge_small(source: Source, config: ChunkConfig | None = None, min_chars: int = 300) -> ChunkResult:
    if min_chars <= 0:
        raise ValueError(f"min_chars must be positive, got {min_chars}")
    config = config or ChunkConfig()
    spans, dropped = merge_small_spans(source.text, config, min_chars)
    version = f"merge-small-1/{config.max_chars}/{config.hard_limit}/{min_chars}"
    return _finish(source, config, spans, dropped, version, False)


def check_chunking(source: Source, passages: list[Passage], config: ChunkConfig, *,
                   structure: bool = True) -> list[str]:
    """The problems found in a chunking of `source`, empty when it is sound (spec 09)."""
    problems: list[str] = []
    lines = _scan(source.text)
    previous: Passage | None = None
    seen: set[str] = set()
    gaps: list[tuple[int, int]] = []
    cursor = 0
    for p in passages:
        label = p.passage_id
        if not verify_passage(source, p):
            problems.append(f"{label}: is not an exact slice of this source version")
            continue
        if p.passage_id in seen:
            problems.append(f"{label}: duplicate passage id")
        seen.add(p.passage_id)
        if (p.passage_id != make_passage_id(p.source_id, p.version, p.start, p.end, content_sha256(p.text), p.chunker_version)
                or p.content_id != content_sha256(p.text)):
            problems.append(f"{label}: ids do not match the passage")
        if len(p.text) > config.hard_limit:
            problems.append(f"{label}: over the hard limit ({len(p.text)} > {config.hard_limit})")
        if structure:
            for ln in lines:
                if ln.heading and p.start < ln.start < p.end:
                    problems.append(f"{label}: a heading line starts inside the passage")
                    break
        if previous is not None and p.start < previous.start:
            problems.append(f"{label}: passages are out of order")
        elif previous is not None and p.start < previous.end:
            problems.append(f"{label}: overlaps the passage before it")
        else:
            gaps.append((cursor, p.start))
        cursor = max(cursor, p.end)
        previous = p
    gaps.append((cursor, len(source.text)))
    for a, b in gaps:
        if b > a and not _only_headings(source.text[a:b]):
            problems.append(f"text lost between {a} and {b}: {source.text[a:b].strip()[:40]!r}")
    return problems
