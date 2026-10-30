"""Corpus exclusions (spec 14, ADR 0008): reviewed noise left out of the searchable corpus, per source, by path glob.

The register, the snapshots and the licence gate are not touched; this only filters sources that are already loaded."""
from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

import yaml

ACTIONS = ("exclude", "include")
_KEYS = {"exclude", "include", "reason"}


@dataclass(frozen=True)
class Rule:
    action: str  # "exclude" or "include"
    glob: str
    reason: str = ""


@dataclass
class Applied:
    kept: list
    excluded: list = field(default_factory=list)  # (source_id, reason)
    unused: list = field(default_factory=list)  # (source id, glob) of rules that matched no file


def _rule(source_id: str, position: int, raw: object) -> Rule:
    def bad(why: str) -> ValueError:
        return ValueError(f"{source_id}: rule {position}: {why}")

    if not isinstance(raw, dict):
        raise bad("must be a mapping with exclude or include, and a reason")
    for key in raw:
        if key not in _KEYS:
            raise bad(f"unknown key {key!r}")
    actions = [a for a in ACTIONS if a in raw]
    if len(actions) != 1:
        raise bad("needs exactly one of exclude or include")
    action = actions[0]
    glob = raw[action]
    if not isinstance(glob, str) or not glob.strip():
        raise bad("the glob must be non-empty text")
    path = PurePosixPath(glob)
    if glob.startswith("/") or ".." in path.parts or "\\" in glob:
        raise bad("the glob must be a relative posix path without '..'")
    reason = raw.get("reason", "")
    if reason != "" and not isinstance(reason, str):
        raise bad("reason must be text")
    if action == "exclude" and not reason.strip():
        raise bad("an exclude needs a written reason")
    return Rule(action, glob, reason.strip())


def load_exclusions(path: Path) -> dict[str, list[Rule]]:
    path = Path(path)
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"{path.name}: not valid YAML ({exc})") from None
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"{path.name}: must be a mapping of source id to a list of rules")
    out: dict[str, list[Rule]] = {}
    for source_id, rules in data.items():
        if not isinstance(source_id, str) or not source_id.strip():
            raise ValueError(f"{path.name}: every source id must be text, got {source_id!r}")
        if not isinstance(rules, list):
            raise ValueError(f"{source_id}: must be a list of rules")
        if not rules:
            raise ValueError(f"{source_id}: needs at least one rule")
        out[source_id] = [_rule(source_id, i, raw) for i, raw in enumerate(rules, 1)]
    return out


def matching_rule(rules: list[Rule], rel_path: str) -> Rule | None:
    """The last rule whose glob matches the whole path; None when no rule matches (the file stays)."""
    found = None
    for rule in rules:
        if fnmatch.fnmatchcase(rel_path, rule.glob):
            found = rule
    return found


def apply_exclusions(sources: list, exclusions: dict[str, list[Rule]]) -> Applied:
    kept, excluded = [], []
    used: set[tuple[str, int]] = set()
    for source in sources:
        source_id, _, rel = source.source_id.partition("/")
        rules = exclusions.get(source_id)
        if not rules or not rel:
            kept.append(source)
            continue
        hits = [i for i, rule in enumerate(rules) if fnmatch.fnmatchcase(rel, rule.glob)]
        used.update((source_id, i) for i in hits)
        last = rules[hits[-1]] if hits else None
        if last is not None and last.action == "exclude":
            excluded.append((source.source_id, last.reason))
        else:
            kept.append(source)
    unused = [(sid, rule.glob) for sid, rules in exclusions.items() for i, rule in enumerate(rules) if (sid, i) not in used]
    return Applied(kept, excluded, unused)
