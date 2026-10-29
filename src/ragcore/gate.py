"""The licence gate: the most a source's licence lets the library take from it (spec 02)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from ragcore.errors import RegisterError

# Best first. A verdict never says "rejected": that is a person's decision.
RANK = {"reference-only": 0, "limited": 1, "approved": 2}
CLASSES = ("allowed", "share-alike", "non-commercial")


@dataclass(frozen=True)
class Verdict:
    status: str  # the most the gate allows: approved, limited or reference-only
    reasons: tuple[str, ...]
    share_alike: bool = False


@dataclass(frozen=True)
class LicenceRules:
    classes: dict[str, frozenset[str]]

    @classmethod
    def load(cls, path: Path) -> LicenceRules:
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            raise RegisterError(f"{path.name}: not valid YAML: {exc}") from exc
        if not isinstance(data, dict):
            raise RegisterError(f"{path.name}: licence rules must be a mapping of class to licence ids.")
        unknown = sorted(set(data) - set(CLASSES))
        if unknown:
            raise RegisterError(f"{path.name}: unknown licence class: {', '.join(unknown)}.")
        classes = {name: frozenset(map(str, data.get(name) or ())) for name in CLASSES}
        seen: dict[str, str] = {}
        for name, ids in classes.items():
            for licence_id in sorted(ids):
                if licence_id in seen:
                    raise RegisterError(f"{path.name}: {licence_id} is listed as {seen[licence_id]} and {name}.")
                seen[licence_id] = name
        return cls(classes)

    def classify(self, licence_id: str) -> str:
        """allowed, share-alike, non-commercial, or unknown (which is treated as reference-only)."""
        for name, ids in self.classes.items():
            if licence_id in ids:
                return name
        return "unknown"


def _verdict(licence_id: str, covers: str, rules: LicenceRules) -> Verdict:
    kind = rules.classify(licence_id)
    if kind == "unknown":
        return Verdict("reference-only", (f"The licence '{licence_id}' is not on the allowed list, so we cite this source and copy none of it.",))
    if kind == "non-commercial":
        return Verdict("reference-only", (f"The licence '{licence_id}' is non-commercial, which blocks use if the academy becomes a product.",))
    if covers == "unknown":
        return Verdict("reference-only", ("What the licence covers is not known, so nothing is clearly allowed.",))
    reasons = []
    status = "approved"
    if covers != "both":
        status = "limited"
        reasons.append(f"The licence clearly covers only the {covers}.")
    if kind == "share-alike":
        status = "limited"
        reasons.append("Share-alike: anything we write from it carries the same licence.")
    return Verdict(status, tuple(reasons) or ("Open licence covering code and text.",), share_alike=kind == "share-alike")


def check(record, rules: LicenceRules) -> Verdict:
    """Say the most the library may take from a source, with reasons in plain words."""
    return _verdict(record.licence.id, record.licence.covers, rules)


def _matches(prefix: str, path: str) -> bool:
    prefix = prefix.rstrip("/")
    return path == prefix or path.startswith(prefix + "/")


def check_path(record, rules: LicenceRules, path: str) -> Verdict:
    """The verdict for one file. The longest matching `licence_overrides` prefix names its licence;
    an override can lower the record's own verdict but never raise it."""
    own = check(record, rules)
    matching = [p for p in record.licence_overrides if _matches(p, path)]
    if not matching:
        return own
    best = max(matching, key=lambda p: len(p.rstrip("/")))
    override = _verdict(record.licence_overrides[best], record.licence.covers, rules)
    return override if RANK[override.status] < RANK[own.status] else own


def licence_for_path(record, rules: LicenceRules, path: str) -> str:
    """The licence id that governs one file: the override that `check_path` applies, else the record's own."""
    own = check(record, rules)
    matching = [p for p in record.licence_overrides if _matches(p, path)]
    if not matching:
        return record.licence.id
    best = max(matching, key=lambda p: len(p.rstrip("/")))
    override = _verdict(record.licence_overrides[best], record.licence.covers, rules)
    return record.licence_overrides[best] if RANK[override.status] < RANK[own.status] else record.licence.id
