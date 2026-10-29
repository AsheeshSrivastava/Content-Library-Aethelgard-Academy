"""The source register: one YAML record per source, with an enforced status path (spec 02)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from ragcore.errors import RegisterError, TransitionError
from ragcore.gate import RANK, LicenceRules, check

__all__ = [
    "RegisterError", "TransitionError", "Licence", "FetchSpec", "SourceRecord",
    "load_record", "load_register", "save_record", "set_status", "to_contract_fields",
]

KINDS = ("docs", "repo", "paper", "book", "video", "guide", "course")
AREAS = ("python", "ai-engineering", "backend")
COVERS = ("code", "text", "both", "unknown")
BASES = ("detected", "declared", "unknown")
STATUSES = ("quarantine", "approved", "limited", "reference-only", "rejected")
SAFE_FILE = re.compile(r"[A-Za-z0-9._][A-Za-z0-9._/-]*")  # No wildcards, so git treats a name as exactly that file.
FULL_SHA = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True)
class Licence:
    id: str
    file_url: str
    covers: str
    note: str = ""
    file_hash: str = ""
    basis: str = "declared"


@dataclass(frozen=True)
class FetchSpec:
    """Where a pinned copy comes from: a git repository (optionally pinned by commit) or one web page."""
    git: str = ""
    ref: str = ""
    commit: str = ""  # Full 40-hex SHA; when set the fetcher fetches exactly that commit (spec 03).
    paths: tuple[str, ...] = ()
    files: tuple[str, ...] = ()
    web: str = ""
    licence_url: str = ""  # With `web`: the licence text kept beside the page.


@dataclass(frozen=True)
class SourceRecord:
    id: str
    title: str
    owner: str
    url: str
    kind: str
    area: str
    licence: Licence
    version_ref: str
    fetched: date | None
    content_hash: str
    status: str
    reviewer: str = ""
    reviewed_on: date | None = None
    override_reason: str = ""
    notes: str = ""
    fetch: FetchSpec | None = None
    max_mb: int = 200
    licence_overrides: dict[str, str] = field(default_factory=dict)  # path prefix -> licence id


def _fail(path: Path, message: str) -> RegisterError:
    return RegisterError(f"{path.name}: {message}")


def _text(path: Path, data: dict[str, Any], key: str, *, required: bool = True, where: str = "") -> str:
    value = data.get(key)
    if value is None or (isinstance(value, str) and not value.strip()):
        if required:
            raise _fail(path, f"missing field '{where}{key}'.")
        return ""
    return str(value).strip()


def _choice(path: Path, data: dict[str, Any], key: str, options: tuple[str, ...], *, where: str = "") -> str:
    value = _text(path, data, key, where=where)
    if value not in options:
        raise _fail(path, f"field '{where}{key}' must be one of {', '.join(options)}; found '{value}'.")
    return value


def _date(path: Path, data: dict[str, Any], key: str) -> date | None:
    value = data.get(key)
    if value in (None, ""):
        return None
    if isinstance(value, date):
        return value
    raise _fail(path, f"field '{key}' must be a date like 2026-10-22.")


def _plain_path(value: str) -> bool:
    return bool(SAFE_FILE.fullmatch(value)) and ".." not in value.split("/") and "//" not in value


def load_record(path: Path) -> SourceRecord:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise _fail(path, f"not valid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise _fail(path, "a source record must be a mapping.")
    lic = data.get("licence")
    if not isinstance(lic, dict):
        raise _fail(path, "missing field 'licence'.")
    version = data.get("version") or {}
    licence = Licence(
        id=_text(path, lic, "id", where="licence."),
        file_url=_text(path, lic, "file_url", required=False),
        covers=_choice(path, lic, "covers", COVERS, where="licence."),
        note=_text(path, lic, "note", required=False),
        file_hash=_text(path, lic, "file_hash", required=False),
        basis=str(lic.get("basis") or "declared").strip(),
    )
    if licence.basis not in BASES:
        raise _fail(path, f"field 'licence.basis' must be one of {', '.join(BASES)}; found '{licence.basis}'.")
    if licence.basis == "unknown" and licence.id != "unknown":
        raise _fail(path, "licence.basis 'unknown' is allowed only with licence.id 'unknown'.")
    return SourceRecord(
        id=_text(path, data, "id"),
        title=_text(path, data, "title"),
        owner=_text(path, data, "owner"),
        url=_text(path, data, "url"),
        kind=_choice(path, data, "kind", KINDS),
        area=_choice(path, data, "area", AREAS),
        licence=licence,
        version_ref=_text(path, version, "ref", required=False),
        fetched=_date(path, version, "fetched"),
        content_hash=_text(path, data, "content_hash", required=False),
        status=_choice(path, data, "status", STATUSES),
        reviewer=_text(path, data, "reviewer", required=False),
        reviewed_on=_date(path, data, "reviewed_on"),
        override_reason=_text(path, data, "override_reason", required=False),
        notes=_text(path, data, "notes", required=False),
        fetch=_fetch_spec(path, data.get("fetch")),
        max_mb=_max_mb(path, data.get("max_mb")),
        licence_overrides=_overrides(path, data.get("licence_overrides")),
    )


def _fetch_spec(path: Path, value: Any) -> FetchSpec | None:
    if value is None:
        return None
    git = str(value.get("git") or "").strip() if isinstance(value, dict) else ""
    web = str(value.get("web") or "").strip() if isinstance(value, dict) else ""
    if bool(git) == bool(web):
        raise _fail(path, "field 'fetch' needs either a 'git' address or a 'web' address, not both.")
    if web and not str(value.get("licence_url") or "").strip():
        raise _fail(path, "field 'fetch.web' needs a 'licence_url'.")
    commit = str(value.get("commit") or "").strip()
    if commit and not FULL_SHA.fullmatch(commit):
        raise _fail(path, "field 'fetch.commit' must be a full 40-character lowercase hex commit SHA.")
    paths = value.get("paths") or []
    if not isinstance(paths, list) or not all(isinstance(p, str) and p.strip() for p in paths):
        raise _fail(path, "field 'fetch.paths' must be a list of folder names.")
    files = value.get("files") or []
    if not isinstance(files, list) or not all(isinstance(f, str) and _plain_path(f) and not f.endswith("/") for f in files):
        raise _fail(path, "field 'fetch.files' must be a list of plain file paths inside the repository "
                          "(letters, digits, dot, dash, underscore and slash; no '..').")
    return FetchSpec(git, str(value.get("ref") or "").strip(), commit, tuple(paths), tuple(files),
                     web, str(value.get("licence_url") or "").strip())


def _max_mb(path: Path, value: Any) -> int:
    if value is None:
        return 200
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise _fail(path, "field 'max_mb' must be a whole number of megabytes, 1 or more.")
    return value


def _overrides(path: Path, value: Any) -> dict[str, str]:
    if value is None:
        return {}
    ok = isinstance(value, dict) and all(
        isinstance(k, str) and isinstance(v, str) and v.strip() and _plain_path(k.rstrip("/") or "x")
        for k, v in value.items())
    if not ok:
        raise _fail(path, "field 'licence_overrides' must map plain relative path prefixes to licence ids.")
    return {k: v.strip() for k, v in value.items()}


def load_register(folder: Path) -> list[SourceRecord]:
    """Every record in a folder, sorted by id; a duplicate id stops the load."""
    found: dict[str, tuple[SourceRecord, Path]] = {}
    for path in sorted(folder.glob("*.yaml")):
        record = load_record(path)
        if record.id in found:
            raise _fail(path, f"duplicate source id '{record.id}' (also in {found[record.id][1].name}).")
        found[record.id] = (record, path)
    return [found[key][0] for key in sorted(found)]


def save_record(record: SourceRecord, path: Path) -> None:
    lic = record.licence
    data: dict[str, Any] = {
        "id": record.id, "title": record.title, "owner": record.owner, "url": record.url,
        "kind": record.kind, "area": record.area,
        "licence": {"id": lic.id, "file_url": lic.file_url, "covers": lic.covers, "note": lic.note,
                    "file_hash": lic.file_hash, "basis": lic.basis},
        "version": {"ref": record.version_ref, "fetched": record.fetched},
        "content_hash": record.content_hash, "status": record.status,
        "reviewer": record.reviewer, "reviewed_on": record.reviewed_on,
        "override_reason": record.override_reason, "notes": record.notes, "max_mb": record.max_mb,
    }
    if record.licence_overrides:
        data["licence_overrides"] = dict(record.licence_overrides)
    fetch = record.fetch
    if fetch is not None:
        out: dict[str, Any] = {}
        if fetch.git:
            out["git"] = fetch.git
        for name in ("ref", "commit"):
            if getattr(fetch, name):
                out[name] = getattr(fetch, name)
        out["paths"], out["files"] = list(fetch.paths), list(fetch.files)
        if fetch.web:
            out["web"], out["licence_url"] = fetch.web, fetch.licence_url
        data["fetch"] = out
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _missing_evidence(record: SourceRecord) -> list[str]:
    missing = []
    if not record.licence.file_url:
        missing.append("the licence file URL")
    if not record.version_ref:
        missing.append("a pinned version or commit")
    if not record.licence.file_hash:
        missing.append("the licence file hash")
    if not record.content_hash:
        missing.append("a content hash")
    return missing


def set_status(record: SourceRecord, new: str, rules: LicenceRules, *, reviewer: str, on: date,
               override_reason: str = "") -> SourceRecord:
    """Move a source to a new status, or refuse with the reason."""
    if new not in STATUSES:
        raise TransitionError(f"Unknown status '{new}'.")
    if record.status == "rejected":
        raise TransitionError("A rejected source is final; add a new record instead.")
    if not reviewer.strip():
        raise TransitionError("A status change needs a reviewer name.")
    changes: dict[str, Any] = {"status": new, "reviewer": reviewer.strip(), "reviewed_on": on}
    if new in ("approved", "limited"):
        missing = _missing_evidence(record)
        if missing:
            raise TransitionError(f"Cannot move to {new} without {', '.join(missing)}.")
        verdict = check(record, rules)
        if RANK[new] > RANK[verdict.status]:
            if not override_reason.strip():
                raise TransitionError(
                    f"The licence gate allows at most {verdict.status} ({' '.join(verdict.reasons)}); "
                    "an override needs a written reason.")
            changes["override_reason"] = override_reason.strip()
    return replace(record, **changes)


def to_contract_fields(record: SourceRecord) -> dict[str, str]:
    """The licence id and basis a loaded contracts.Source carries."""
    return {"licence": record.licence.id, "licence_basis": record.licence.basis}
