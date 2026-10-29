"""The question test (spec 12): load question files, check them against the library, score any retriever.

A retriever is a function, so chunkers, embedders and rerankers are all measured with the same ruler."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Callable, Iterable

import yaml

SCORE_DEPTH = 10  # How far down the results to look for the best right result.


@dataclass(frozen=True)
class Question:
    q: str
    expect: tuple[str, ...]


@dataclass(frozen=True)
class QuestionSet:
    name: str
    questions: tuple[Question, ...]
    outside: tuple[str, ...] = ()


@dataclass(frozen=True)
class Hit:
    source_id: str
    score: float


Retriever = Callable[[str, int], list[Hit]]


@dataclass(frozen=True)
class Problem:
    kind: str
    set: str
    message: str


def _text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def load_question_file(path: Path) -> QuestionSet:
    def bad(why: str) -> ValueError:
        return ValueError(f"{path.name}: {why}")

    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise bad(f"not valid YAML ({exc})") from None
    raw = data.get("questions") if isinstance(data, dict) else None
    if not isinstance(raw, list):
        raise bad("needs a list called questions.")
    questions = []
    for item in raw:
        if not (isinstance(item, dict) and _text(item.get("q")) and isinstance(item.get("expect"), list)
                and item["expect"] and all(_text(piece) for piece in item["expect"])):
            raise bad("each question needs a q and a non-empty expect list of non-blank strings.")
        questions.append(Question(item["q"], tuple(item["expect"])))
    outside = data.get("outside", [])
    if not isinstance(outside, list) or not all(_text(q) for q in outside):
        raise bad("outside must be a list of non-blank questions.")
    return QuestionSet(path.stem, tuple(questions), tuple(outside))


def load_question_dir(folder: Path) -> list[QuestionSet]:
    paths = sorted(Path(folder).glob("*.yaml"))
    if not paths:
        raise ValueError(f"{folder}: no question files (*.yaml).")
    return [load_question_file(p) for p in paths]


def check_questions(sets: Iterable[QuestionSet], source_ids: Iterable[str], broad_share: float = 0.25) -> list[Problem]:
    """Problems that would make the numbers meaningless: pieces that match nothing or nearly everything, repeats."""
    ids = list(source_ids)
    problems: list[Problem] = []
    seen: dict[str, str] = {}
    answerable: set[str] = set()
    sets = list(sets)
    for s in sets:
        for question in s.questions:
            if question.q in seen:
                problems.append(Problem("duplicate", s.name, f"{question.q!r} is also in {seen[question.q]}"))
            seen.setdefault(question.q, s.name)
            answerable.add(question.q)
            matched = 0
            for piece in question.expect:
                count = sum(piece in sid for sid in ids)
                if count == 0:
                    problems.append(Problem("no-match", s.name, f"{piece!r} (question {question.q!r}) matches no source"))
                else:
                    matched += 1
                    if count > broad_share * len(ids):
                        problems.append(Problem("broad", s.name, f"{piece!r} (question {question.q!r}) matches "
                                                f"{count} of {len(ids)} sources"))
            if not matched:
                problems.append(Problem("unreachable", s.name, f"{question.q!r} cannot be answered by any source"))
    for s in sets:
        for q in s.outside:
            if q in answerable:
                problems.append(Problem("outside-is-answerable", s.name, f"{q!r} is also an answerable question"))
    return problems


@dataclass
class Report:
    k: int
    total: int = 0
    first: int = 0
    top_k: int = 0
    misses: list = field(default_factory=list)  # (question, source ids found in the first k)
    right_scores: list = field(default_factory=list)  # best score of a right result, per question that has one
    right_questions: list = field(default_factory=list)
    outside_scores: list = field(default_factory=list)  # best score for each question the library cannot answer
    outside_questions: list = field(default_factory=list)

    @property
    def lowest_right(self):
        return min(self.right_scores) if self.right_scores else None

    @property
    def highest_outside(self):
        return max(self.outside_scores) if self.outside_scores else None

    @property
    def suggested_threshold(self):
        """Midway between the weakest right score and the strongest outside score, or None if they overlap."""
        if self.lowest_right is None or self.highest_outside is None or self.highest_outside >= self.lowest_right:
            return None
        return (self.lowest_right + self.highest_outside) / 2

    def merged(self, other: "Report") -> "Report":
        return Report(self.k, self.total + other.total, self.first + other.first, self.top_k + other.top_k,
                      self.misses + other.misses, self.right_scores + other.right_scores,
                      self.right_questions + other.right_questions, self.outside_scores + other.outside_scores,
                      self.outside_questions + other.outside_questions)

    def summary(self) -> str:
        def rate(n):
            return f"{n} of {self.total} ({round(100 * n / self.total) if self.total else 0}%)"
        lines = [f"Right lesson first for {rate(self.first)}; in the top {self.k} for {rate(self.top_k)}."]
        lines += [f"  missed: {q} -> {', '.join(found) or 'nothing'}" for q, found in self.misses]
        if self.outside_scores and self.right_scores:
            weakest = self.right_questions[self.right_scores.index(self.lowest_right)]
            strongest = self.outside_questions[self.outside_scores.index(self.highest_outside)]
            lines.append(f"Scores: weakest right match {self.lowest_right:.3f} ({weakest}); "
                         f"strongest match for a question outside the library {self.highest_outside:.3f} ({strongest}).")
            for label, scores in (("right matches", self.right_scores), ("outside matches", self.outside_scores)):
                lines.append(f"  {label}: lowest {min(scores):.3f}, median {median(scores):.3f}, highest {max(scores):.3f}")
            if self.suggested_threshold is None:
                lines.append("Scores overlap: no clean threshold. Outside questions can score as high as weak right ones.")
            else:
                lines.append(f"suggested threshold: {self.suggested_threshold:.3f}")
        return "\n".join(lines)


def evaluate(questions: Iterable[Question], outside: Iterable[str], retriever: Retriever, *, k: int = 3,
             depth: int = SCORE_DEPTH) -> Report:
    questions = list(questions)
    report = Report(k, total=len(questions))
    for item in questions:
        hits = retriever(item.q, max(k, depth))
        right = [any(piece in h.source_id for piece in item.expect) for h in hits]
        report.first += bool(right and right[0])
        if any(right[:k]):
            report.top_k += 1
        else:
            report.misses.append((item.q, [h.source_id for h in hits[:k]]))
        scored = [h.score for h, ok in zip(hits[:depth], right) if ok]
        if scored:
            report.right_scores.append(max(scored))
            report.right_questions.append(item.q)
    for question in outside:
        hits = retriever(question, 1)
        report.outside_scores.append(hits[0].score if hits else 0.0)
        report.outside_questions.append(question)
    return report


def evaluate_sets(sets: Iterable[QuestionSet], retriever: Retriever, *, k: int = 3,
                  depth: int = SCORE_DEPTH) -> dict[str, Report]:
    """One report per question file, and `all` for the lot (each question is asked once)."""
    reports = {s.name: evaluate(s.questions, s.outside, retriever, k=k, depth=depth) for s in sets}
    total = Report(k)
    for report in reports.values():
        total = total.merged(report)
    return {**reports, "all": total}
