"""Paired comparison of runs and the pre-registered decision rule (spec 13 sections 4.6 and 5)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ragcore.evalset import Question, QuestionSet, Retriever, SCORE_DEPTH

SECONDARY_SETS = ("rag-questions", "rag-questions.yaml")
FIRST_GAIN, TOP_GAIN, MAX_DROP = 5, 3, 2


@dataclass(frozen=True)
class QResult:
    q: str
    first: bool
    top_k: bool
    rr: float  # 1 / rank of the first right file within `depth`, else 0
    recall10: bool


def score_questions(questions: Iterable[Question], retriever: Retriever, *, k: int = 3,
                    depth: int = SCORE_DEPTH) -> list[QResult]:
    results = []
    for item in questions:
        hits = retriever(item.q, max(k, depth))
        right = [any(piece in h.source_id for piece in item.expect) for h in hits]
        rank = right.index(True) + 1 if any(right) else None
        within = rank is not None and rank <= depth
        results.append(QResult(item.q, bool(right and right[0]), any(right[:k]), 1 / rank if within else 0.0, within))
    return results


def tally(results: list[QResult]) -> dict:
    n = len(results)
    return {"total": n, "first": sum(r.first for r in results), "top_k": sum(r.top_k for r in results),
            "recall10": sum(r.recall10 for r in results), "mrr": sum(r.rr for r in results) / n if n else 0.0}


def _same(a: list[QResult], b: list[QResult]) -> None:
    if [r.q for r in a] != [r.q for r in b]:
        raise ValueError("paired comparison needs the same questions in the same order")


def paired(a: list[QResult], b: list[QResult]) -> dict:
    _same(a, b)
    out = {}
    for field in ("first", "top_k"):
        only_a = sum(x.__dict__[field] and not y.__dict__[field] for x, y in zip(a, b))
        only_b = sum(y.__dict__[field] and not x.__dict__[field] for x, y in zip(a, b))
        out[f"only_a_{field}"], out[f"only_b_{field}"], out[f"net_{field}"] = only_a, only_b, only_b - only_a
    return out


def verdict(candidate: list[QResult], baseline: list[QResult]) -> str:
    """beats / close / worse, by the rule fixed before any run (spec 13 section 5)."""
    _same(candidate, baseline)
    d_first = sum(r.first for r in candidate) - sum(r.first for r in baseline)
    d_top = sum(r.top_k for r in candidate) - sum(r.top_k for r in baseline)
    if (d_first >= FIRST_GAIN and d_top >= -MAX_DROP) or (d_top >= TOP_GAIN and d_first >= -MAX_DROP):
        return "beats"
    return "close" if d_first >= -MAX_DROP and d_top >= -MAX_DROP else "worse"


def split_sets(sets: Iterable[QuestionSet]) -> tuple[list[QuestionSet], list[QuestionSet]]:
    """Primary = every file except rag-questions; secondary = rag-questions alone."""
    sets = list(sets)
    return ([s for s in sets if s.name not in SECONDARY_SETS], [s for s in sets if s.name in SECONDARY_SETS])


def markdown_table(runs: dict[str, list[QResult]], *, baseline: str) -> str:
    if baseline not in runs:
        raise ValueError(f"baseline {baseline!r} is not one of the runs")
    lines = ["| Run | Questions | Right first | Top 3 | MRR | Recall@10 | Net first | Net top 3 | Verdict |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name, results in runs.items():
        t = tally(results)
        if name == baseline:
            net_first = net_top = "-"
            call = "baseline"
        else:
            p = paired(runs[baseline], results)
            net_first, net_top, call = f"{p['net_first']:+d}", f"{p['net_top_k']:+d}", verdict(results, runs[baseline])
        lines.append(f"| {name} | {t['total']} | {t['first']} | {t['top_k']} | {t['mrr']:.3f} | {t['recall10']} | "
                     f"{net_first} | {net_top} | {call} |")
    return "\n".join(lines)
