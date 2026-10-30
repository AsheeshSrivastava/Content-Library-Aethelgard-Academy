"""Passage-level questions (spec 16): loading, the per-question checks and the mix rule of the whole set."""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import yaml

from ragcore.quote_picker import group_of

KINDS = ("fact", "how-to", "code", "number", "context")
QUOTE_MIN, QUOTE_MAX = 40, 400
COPY_RUN = 4  # A question may not repeat this many words in a row from its quote.
MIN_STEM = 6  # Shorter file names ("api") are too common in questions to check.
MIN_SHARE = {"code": 0.25, "number": 0.10, "context": 0.15}
MAX_KIND_SHARE, MAX_PER_SOURCE, MIN_SOURCES, FULL_SET = 0.40, 12, 15, 100
_KEYS = {"q", "source", "quote", "kind"}
_WORD = re.compile(r"\w+")


@dataclass(frozen=True)
class PassageQuestion:
    q: str
    source: str
    quote: str
    kind: str


def _text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _read(path: Path) -> list[PassageQuestion]:
    def bad(why: str) -> ValueError:
        return ValueError(f"{path.name}: {why}")

    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise bad(f"not valid YAML ({exc})") from None
    items = data.get("questions") if isinstance(data, dict) else None
    if not isinstance(items, list) or not items:
        raise bad("needs a non-empty list called questions")
    out = []
    for n, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            raise bad(f"question {n} must be a mapping")
        for key in item:
            if key not in _KEYS:
                raise bad(f"question {n}: unknown key {key!r}")
        if not (_text(item.get("q")) and _text(item.get("source"))):
            raise bad(f"question {n} needs q and source as text")
        if not _text(item.get("quote")):
            raise bad(f"question {n} needs a quote as text")
        if item.get("kind") not in KINDS:
            raise bad(f"question {n}: kind must be one of {', '.join(KINDS)}")
        out.append(PassageQuestion(item["q"].strip(), item["source"].strip(), item["quote"], item["kind"]))
    return out


def load_passage_questions(path: Path) -> list[PassageQuestion]:
    """One file, or every .yaml file of a folder in name order."""
    path = Path(path)
    if path.is_dir():
        files = sorted(path.glob("*.yaml"))
        if not files:
            raise ValueError(f"{path}: no question files (*.yaml)")
        return [q for f in files for q in _read(f)]
    return _read(path)


def _words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def _runs(text: str) -> set[tuple[str, ...]]:
    words = _words(text)
    return {tuple(words[i:i + COPY_RUN]) for i in range(len(words) - COPY_RUN + 1)}


def _resolve(source: str, sources: dict[str, str]) -> list[str]:
    if source in sources:
        return [source]
    return [s for s in sources if source in s]


def _file(item: PassageQuestion, sources: dict[str, str] | None) -> str:
    found = _resolve(item.source, sources) if sources else []
    return found[0] if len(found) == 1 else item.source


def check_questions(questions: list[PassageQuestion], sources: dict[str, str]) -> list[str]:
    """Problems found in each question against the loaded source texts (source id -> text); empty means all fine."""
    problems: list[str] = []
    for item in questions:
        def add(why: str) -> None:
            problems.append(f"'{item.q}': {why}")

        found = _resolve(item.source, sources)
        if len(found) != 1:
            add(f"no source matches {item.source!r}" if not found else f"{len(found)} sources match {item.source!r}")
        elif not QUOTE_MIN <= len(item.quote) <= QUOTE_MAX:
            add(f"quote is {len(item.quote)} characters; it must be {QUOTE_MIN} to {QUOTE_MAX}")
        else:
            times = sources[found[0]].count(item.quote)
            if times != 1:
                add("quote not found in the source" if not times else f"quote is in the source {times} times, not once")
            stem = Path(found[0]).stem.lower()
            if len(stem) >= MIN_STEM and stem in item.q.lower():
                add(f"names the file ({stem})")
        if _runs(item.q) & _runs(item.quote):
            add(f"repeats {COPY_RUN} words in a row from the quote")
    for label, key in (("same question", lambda x: " ".join(_words(x.q))), ("same quote", lambda x: x.quote)):
        for value, count in Counter(key(x) for x in questions).items():
            if count > 1:
                problems.append(f"{label} listed {count} times: {value[:60]!r}")
    return problems


def check_mix(questions: list[PassageQuestion], sources: dict[str, str] | None = None) -> list[str]:
    """The spec 16 section 2 mix rule for the whole set. With `sources`, a question counts for the register source of the
    file it names; without, for the first part of its `source` text."""
    n = len(questions)
    if n < FULL_SET:
        return [f"{n} questions; the set should have about 150"]
    problems = []
    kinds = Counter(x.kind for x in questions)
    for kind, share in MIN_SHARE.items():
        if kinds[kind] / n < share:
            problems.append(f"{kind} is {kinds[kind]} of {n}; it needs at least {share:.0%}")
    for kind, count in kinds.items():
        if count / n > MAX_KIND_SHARE:
            problems.append(f"{kind} is {count} of {n}, above {MAX_KIND_SHARE:.0%}")
    per_source = Counter(group_of(_file(x, sources)) for x in questions)
    for source, count in per_source.items():
        if count > MAX_PER_SOURCE:
            problems.append(f"{source} has {count} questions, above {MAX_PER_SOURCE}")
    if len(per_source) < MIN_SOURCES:
        problems.append(f"{len(per_source)} sources are covered; at least {MIN_SOURCES} are needed")
    return problems
