"""Snapshot loader: a pinned snapshot folder plus its register record become contracts.Source objects (spec 06).

Only gated files load. Every file ends in exactly one bucket and nothing is dropped without a named reason."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from ragcore.contracts import Source, normalize_text
from ragcore.errors import LoadError, LoadRefused, SkipShareExceeded
from ragcore.files import MAX_FILE_BYTES, classify_file, raw_problem
from ragcore.gate import LicenceRules, check_path, licence_for_path
from ragcore.loaders.plain import PLAIN_READERS, PLAIN_VERSION, Reader  # noqa: F401
from ragcore.loaders.registry import DEFAULT_READERS
from ragcore.register import SourceRecord, to_contract_fields

LOADABLE_STATUSES = ("approved", "limited")
BUCKETS = ("read", "licence", "ignored", "code-config", "skipped", "withheld")


@dataclass
class LoadResult:
    sources: list[Source] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)
    withheld: dict[str, str] = field(default_factory=dict)
    code_config: list[str] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=lambda: dict.fromkeys(BUCKETS, 0))

    @property
    def skip_share(self) -> float:
        candidates = self.counts["read"] + self.counts["skipped"]
        return self.counts["skipped"] / candidates if candidates else 1.0


def load_snapshot(folder: Path, record: SourceRecord, rules: LicenceRules, *,
                  readers: dict[str, tuple[str, Reader]] | None = None, max_skip_share: float = 0.25,
                  max_file_bytes: int = MAX_FILE_BYTES) -> LoadResult:
    if record.status not in LOADABLE_STATUSES:
        raise LoadRefused(f"{record.id}: status is {record.status}; only {' or '.join(LOADABLE_STATUSES)} sources load.")
    version = (record.fetch.commit if record.fetch and record.fetch.commit else "") or record.version_ref
    if not version:
        raise LoadRefused(f"{record.id}: no pinned commit or version, so the text cannot be tied to a version.")
    folder = Path(folder)
    readers = {ext.lower(): pair for ext, pair in (DEFAULT_READERS if readers is None else readers).items()}
    result = LoadResult()

    for path in sorted(folder.rglob("*")):
        parts = path.relative_to(folder).parts
        if ".git" in parts or not (path.is_file() or path.is_symlink()):
            continue
        rel = path.relative_to(folder).as_posix()
        bucket, reason = classify_file(rel, symlink=path.is_symlink(), loaders=readers)
        if bucket == "code-config":
            result.code_config.append(rel)
        if bucket != "read":
            result.counts[bucket] += 1
            if bucket == "skipped":
                result.skipped[rel] = reason
            continue

        verdict = check_path(record, rules, rel)
        if verdict.status == "reference-only":
            result.counts["withheld"] += 1
            result.withheld[rel] = " ".join(verdict.reasons)
            continue

        def skip(why: str) -> None:
            result.counts["skipped"] += 1
            result.skipped[rel] = why

        reader = readers.get(path.suffix.lower())
        if reader is None:
            skip("no-reader")
            continue
        try:
            raw = path.read_bytes()
        except OSError:
            skip("unreadable")
            continue
        problem = raw_problem(raw, max_file_bytes)
        if problem:
            skip(problem)
            continue
        loader_version, read = reader
        try:
            text = normalize_text(read(raw))
        except UnicodeDecodeError:
            skip("not-utf8")
            continue
        except LoadError as exc:
            skip(str(exc))
            continue
        except Exception:
            skip("unreadable")
            continue
        if not text.strip():
            skip("empty")
            continue
        result.counts["read"] += 1
        result.sources.append(Source(
            source_id=f"{record.id}/{rel}", version=version, text=text, raw_sha256=hashlib.sha256(raw).hexdigest(),
            loader_version=loader_version, **{**to_contract_fields(record),
                                              "licence": licence_for_path(record, rules, rel)}))

    candidates = result.counts["read"] + result.counts["skipped"]
    if candidates == 0 or result.skip_share > max_skip_share:  # A folder with nothing to read cannot pass.
        raise SkipShareExceeded(
            f"{record.id}: {result.counts['skipped']} of {result.counts['read'] + result.counts['skipped']} "
            f"files were skipped ({result.skip_share:.0%}, limit {max_skip_share:.0%}).", result)
    return result
