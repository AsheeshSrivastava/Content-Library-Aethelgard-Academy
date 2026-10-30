"""Scoring passage-level questions (spec 16 section 5): a passage holds a quote when it covers all of it."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ragcore.compare import QResult
from ragcore.passage_questions import PassageQuestion, check_questions
from ragcore.retrieve import PassageHit
from ragcore.runs import Run

DEPTH = 10  # Passages looked at per question: the first, the first 3, and the first 10 (recall).


@dataclass(frozen=True)
class QuoteSpan:
    source_id: str
    start: int
    end: int


def quote_span(item: PassageQuestion, sources: dict[str, str]) -> QuoteSpan:
    """Where a question's quote sits in its source; refuses what the question checks refuse."""
    problems = check_questions([item], sources)
    if problems:
        raise ValueError(problems[0])
    source_id = item.source if item.source in sources else next(s for s in sources if item.source in s)
    start = sources[source_id].index(item.quote)
    return QuoteSpan(source_id, start, start + len(item.quote))


def holds(hit: PassageHit, span: QuoteSpan) -> bool:
    return hit.source_id == span.source_id and hit.start <= span.start and span.end <= hit.end


def score_passage_questions(questions: list[PassageQuestion], spans: list[QuoteSpan],
                            retriever: Callable[[str, int], list[PassageHit]], *, k: int = 3) -> list[QResult]:
    if len(questions) != len(spans):
        raise ValueError("one place per question is needed")
    out = []
    for item, span in zip(questions, spans):
        right = [holds(h, span) for h in retriever(item.q, DEPTH)]
        rank = right.index(True) + 1 if any(right) else None
        out.append(QResult(item.q, bool(right and right[0]), any(right[:k]), 1 / rank if rank else 0.0, rank is not None))
    return out


def cut_flags(run: Run, spans: list[QuoteSpan]) -> list[bool]:
    """True for each quote that no passage of the run holds whole: the chunking cut it."""
    by_source: dict[str, list[tuple[int, int]]] = {}
    for row in run.rows:
        by_source.setdefault(row["source_id"], []).append((row["start"], row["end"]))
    return [not any(a <= s.start and s.end <= b for a, b in by_source.get(s.source_id, [])) for s in spans]
