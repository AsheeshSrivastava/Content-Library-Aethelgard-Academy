"""Inspection of a pinned snapshot: what is in it, what loads, what is dirty (spec 05).

Nothing here changes a snapshot or indexes anything. Until loaders exist, files are read in a plain "look" mode
(UTF-8, normalised with contracts.normalize_text). Loaders register themselves in READERS and are then checked."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ragcore.contracts import NORMALIZER_VERSION, normalize_text
from ragcore.fetch import snapshot_hash
from ragcore.files import (CODE_CONFIG_EXTS, CODE_CONFIG_NAMES, IGNORED_EXTS, MAX_FILE_BYTES, NOTEBOOK_EXT, TEXT_EXTS,  # noqa: F401
                           classify_file, is_licence, raw_problem)
from ragcore.register import RegisterError, load_record

Reader = Callable[[bytes], str]

MIN_CHARS = 100
EXAMPLES = 5
EXAMPLE_CHARS = 120
RAW_KINDS = ("bom", "crlf", "not_nfc")
MARKUP_KINDS = ("html_tags", "jsx_tags", "import_export", "rst_directives", "rst_roles", "md_tables")

READERS: dict[str, Reader] = {}  # Loaders register here by extension (".md" -> function).


def register_reader(extension: str, reader: Reader) -> None:
    READERS[extension.lower()] = reader


class LoadError(Exception):
    """A reader that refuses a file; the message is the reason code shown in the report."""


# --- look readers ------------------------------------------------------------------------

def _decode(raw: bytes) -> str:
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise LoadError("not-utf8") from exc


def look_text(raw: bytes) -> str:
    return _decode(raw)


def _notebook(raw: bytes) -> dict:
    try:
        data = json.loads(_decode(raw))
    except (ValueError, LoadError) as exc:
        raise LoadError("unreadable") from exc
    if not isinstance(data, dict) or not isinstance(data.get("cells"), list):
        raise LoadError("unreadable")
    return data


def _source(cell: dict) -> str:
    src = cell.get("source", "")
    return "".join(src) if isinstance(src, list) else str(src)


def look_notebook(raw: bytes) -> str:
    """Markdown cells as they are, code cells fenced, outputs left out."""
    parts = []
    for cell in _notebook(raw)["cells"]:
        if not isinstance(cell, dict):
            continue
        if cell.get("cell_type") == "markdown":
            parts.append(_source(cell))
        elif cell.get("cell_type") == "code":
            parts.append("```python\n" + _source(cell) + "\n```")
    return "\n\n".join(parts)


def _notebook_stats(raw: bytes) -> dict:
    cells = [c for c in _notebook(raw)["cells"] if isinstance(c, dict)]
    out = sum(len(json.dumps(c.get("outputs", []))) for c in cells if c.get("cell_type") == "code")
    return {"markdown": sum(c.get("cell_type") == "markdown" for c in cells),
            "code": sum(c.get("cell_type") == "code" for c in cells), "output_chars": out, "bytes": len(raw)}


# --- text analysis -----------------------------------------------------------------------------

_HTML = re.compile(r"</?[a-z][a-z0-9-]*(?:\s[^<>]*)?/?>")
_JSX = re.compile(r"</?[A-Z][A-Za-z0-9.]*(?:\s[^<>]*)?/?>")
_IMPORT = re.compile(r"^(?:import|export)\s+\S", re.MULTILINE)
_DIRECTIVE = re.compile(r"^\s*\.\.\s+[\w:-]+::", re.MULTILINE)
_ROLE = re.compile(r":[\w:-]+:`[^`\n]+`")
_TABLE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$", re.MULTILINE)
_MD_HEADING = re.compile(r"^#{1,6}\s+\S")
_UNDERLINE = re.compile(r"^([=\-~^\"#*+`:.'_])\1{2,}\s*$")
_RST_DIRECTIVE_LINE = re.compile(r"^\s*\.\.\s+[\w:-]+::")
_PATTERNS = {"html_tags": _HTML, "jsx_tags": _JSX, "import_export": _IMPORT, "rst_directives": _DIRECTIVE,
             "rst_roles": _ROLE, "md_tables": _TABLE}


def _outside_fences(text: str) -> str:
    kept, inside = [], False
    for line in text.split("\n"):
        if line.strip().startswith(("```", "~~~")):
            inside = not inside
            continue
        if not inside:
            kept.append(line)
    return "\n".join(kept)


def markup_counts(text: str) -> dict[str, int]:
    body = _outside_fences(text)
    return {kind: len(pattern.findall(body)) for kind, pattern in _PATTERNS.items()}


def _first_markup_line(text: str, kind: str) -> str:
    match = _PATTERNS[kind].search(_outside_fences(text))
    if not match:
        return ""
    body = _outside_fences(text)
    start = body.rfind("\n", 0, match.start()) + 1
    end = body.find("\n", match.end())
    return body[start: end if end >= 0 else len(body)].strip()[:EXAMPLE_CHARS]


def _is_underline(line: str, title: str) -> bool:
    return bool(_UNDERLINE.match(line)) and len(line.strip()) >= len(title.strip()) > 0 and not _UNDERLINE.match(title)


def sections(text: str) -> list[tuple[str, str]]:
    """Split at Markdown and rst headings (never inside code fences): [(heading, text)]."""
    lines = text.split("\n")
    starts: list[int] = []
    inside = False
    for i, line in enumerate(lines):
        if line.strip().startswith(("```", "~~~")):
            inside = not inside
            continue
        if inside:
            continue
        if _MD_HEADING.match(line) or (line.strip() and i + 1 < len(lines) and _is_underline(lines[i + 1], line)):
            starts.append(i)
    bounds = ([0] if not starts or starts[0] > 0 else []) + starts + [len(lines)]
    found = []
    for a, b in zip(bounds, bounds[1:]):
        body = "\n".join(lines[a:b]).strip()
        if body:
            found.append((lines[a].strip()[:EXAMPLE_CHARS], body))
    return found


def _prose_chars(text: str) -> int:
    lines, prose, in_directive, skip_next = text.split("\n"), 0, False, False
    for i, line in enumerate(lines):
        if skip_next:
            skip_next = False
            continue
        if _RST_DIRECTIVE_LINE.match(line):
            in_directive = True
            continue
        if in_directive:
            if not line.strip() or line[:1].isspace():
                continue
            in_directive = False
        if _MD_HEADING.match(line) or _UNDERLINE.match(line):
            continue
        if line.strip() and i + 1 < len(lines) and _is_underline(lines[i + 1], line):
            skip_next = True
            continue
        prose += len(line.strip())
    return prose


def _has_front_matter(text: str) -> bool:
    return text.startswith("---\n") and re.search(r"\n---[ \t]*\n", text[4:2000]) is not None


def _near_key(text: str) -> str:
    return re.sub(r"[\W_]+", "", text.lower())


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --- report -------------------------------------------------------------------------------------

@dataclass
class Report:
    source_id: str
    snapshot_hash: str
    section_cap: int
    max_skip_share: float
    files_seen: int = 0
    buckets: dict = field(default_factory=lambda: {"read": 0, "licence": 0, "ignored": 0, "code-config": 0, "skipped": 0})
    skipped: dict = field(default_factory=dict)
    code_config: list = field(default_factory=list)
    extensions: dict = field(default_factory=dict)
    raw: dict = field(default_factory=lambda: {k: 0 for k in RAW_KINDS})
    raw_examples: dict = field(default_factory=lambda: {k: [] for k in RAW_KINDS})
    markup: dict = field(default_factory=lambda: {k: 0 for k in MARKUP_KINDS})
    markup_examples: dict = field(default_factory=lambda: {k: [] for k in MARKUP_KINDS})
    front_matter: int = 0
    tiny: int = 0
    tiny_examples: list = field(default_factory=list)
    stubs: int = 0
    stub_examples: list = field(default_factory=list)
    sections_total: int = 0
    sections_over: int = 0
    sections_over_examples: list = field(default_factory=list)
    sections_under: int = 0
    notebooks: dict = field(default_factory=lambda: {"files": 0, "markdown_cells": 0, "code_cells": 0, "output_share": 0.0})
    idempotence_failures: list = field(default_factory=list)
    exact_duplicates: list = field(default_factory=list)
    near_duplicates: list = field(default_factory=list)
    loader: dict = field(default_factory=dict)
    compare: dict = field(default_factory=dict)
    show_compare: bool = False

    @property
    def skip_share(self) -> float:
        candidates = self.buckets["read"] + self.buckets["skipped"]
        return self.buckets["skipped"] / candidates if candidates else 1.0

    @property
    def candidates(self) -> int:
        return self.buckets["read"] + self.buckets["skipped"]

    @property
    def loader_problems(self) -> int:
        return len(self.loader.get("failures", [])) + len(self.loader.get("regressions", []))

    @property
    def gate_passed(self) -> bool:
        return self.candidates > 0 and self.skip_share <= self.max_skip_share and self.loader_problems == 0

    def gate_line(self) -> str:
        if self.candidates == 0:
            return "GATE FAILED: no candidate files"
        parts = []
        if self.skip_share > self.max_skip_share:
            parts.append(f"{self.skip_share:.4g} of candidate files skipped (limit {self.max_skip_share:.4g})")
        if self.loader_problems:
            parts.append(f"{self.loader_problems} loader problem(s)")
        return "GATE FAILED: " + "; ".join(parts) if parts else "GATE PASSED"

    def to_json(self) -> dict:
        data = {k: v for k, v in self.__dict__.items() if k != "show_compare"}
        data.update(skip_share=self.skip_share, gate_passed=self.gate_passed, normalizer_version=NORMALIZER_VERSION)
        return json.loads(json.dumps(data, sort_keys=True))

    def render(self) -> str:
        out = [f"Inspection report: {self.source_id}", f"Snapshot hash: {self.snapshot_hash}",
               f"Normaliser version: {NORMALIZER_VERSION}",
               f"Thresholds: section cap {self.section_cap}, tiny under {MIN_CHARS}, max skip share {self.max_skip_share:.4g}",
               "", f"Files seen: {self.files_seen}"]
        out += [f"  {name}: {count}" for name, count in self.buckets.items()]
        out += ["", "By extension (files, bytes):"]
        out += [f"  {ext or '(none)'}: {n}, {b}" for ext, (n, b) in sorted(self.extensions.items())]
        by_reason: dict[str, list[str]] = {}
        for path, reason in sorted(self.skipped.items()):
            by_reason.setdefault(reason, []).append(path)
        out += ["", "Skipped files:"] if by_reason else ["", "Skipped files: none"]
        for reason, paths in sorted(by_reason.items()):
            out.append(f"  {reason} ({len(paths)})")
            out += [f"    {p}" for p in paths[:50]]
            if len(paths) > 50:
                out.append(f"    ... and {len(paths) - 50} more")
        out += ["", f"Code and config files (outside the gate): {len(self.code_config)}"]
        out += [f"    {p}" for p in sorted(self.code_config)[:50]]
        if len(self.code_config) > 50:
            out.append(f"    ... and {len(self.code_config) - 50} more (all named in the JSON report)")
        out += ["", "Text problems in the input (files): " + ", ".join(f"{k} {v}" for k, v in self.raw.items())]
        out += [f"  e.g. {k}: {', '.join(v)}" for k, v in self.raw_examples.items() if v]
        out += ["Leftover markup (matches): " + ", ".join(f"{k} {v}" for k, v in self.markup.items())]
        out += [f"  e.g. {k}: {e}" for k, ex in self.markup_examples.items() for e in ex[:2]]
        out += [f"Front matter present: {self.front_matter} files", f"Tiny files (under {MIN_CHARS} characters): {self.tiny}"]
        out += [f"  e.g. {p}" for p in self.tiny_examples]
        out += [f"Directive-only stubs: {self.stubs}"] + [f"  e.g. {p}" for p in self.stub_examples]
        out += [f"Sections: {self.sections_total}, over the cap {self.sections_over}, under {MIN_CHARS} characters {self.sections_under}"]
        out += [f"  e.g. over: {p}" for p in self.sections_over_examples]
        nb = self.notebooks
        out += [f"Notebooks: {nb['files']} files, {nb['markdown_cells']} markdown cells, {nb['code_cells']} code cells, "
                f"outputs are {nb['output_share']:.1%} of the bytes"]
        out += [f"Idempotence failures: {len(self.idempotence_failures)}"] + [f"  {p}" for p in self.idempotence_failures]
        out += [f"Exact duplicate groups: {len(self.exact_duplicates)}"] + [f"  {', '.join(g)}" for g in self.exact_duplicates]
        out += [f"Near duplicate groups: {len(self.near_duplicates)}"] + [f"  {', '.join(g)}" for g in self.near_duplicates]
        if self.loader:
            used = self.loader["used"]
            out += ["", "Loaders: " + (", ".join(f"{e} {u}" for e, u in sorted(used.items())) or "none used")]
            fallback = sorted(e for e, u in used.items() if u == "look")
            if fallback:
                out.append(f"  read by the look reader (no loader yet): {', '.join(fallback)}")
            out += [f"  failure: {p}: {c}" for p, c in self.loader["failures"]]
            out += [f"  regression: {p}: {w}" for p, w in self.loader["regressions"]]
            if self.show_compare:
                out.append("Loader comparison per extension (before -> after):")
                for ext, c in sorted(self.compare.items()):
                    out.append(f"  {ext}: characters {c['chars_before']} -> {c['chars_after']}, "
                               f"markup {c['markup_before']} -> {c['markup_after']}")
        out += ["", self.gate_line(), ""]
        return "\n".join(out)


# --- the inspection -----------------------------------------------------------------------------

def _add_example(bucket: list, text: str) -> None:
    if len(bucket) < EXAMPLES:
        bucket.append(text)


def inspect_folder(folder: Path, *, source_id: str = "", loaders: dict[str, Reader] | None = None,
                   section_cap: int = 6000, max_skip_share: float = 0.25, max_file_bytes: int = MAX_FILE_BYTES,
                   compare: bool = False) -> Report:
    folder = Path(folder)
    report = Report(source_id or folder.name, snapshot_hash(folder), section_cap, max_skip_share, show_compare=compare)
    loaders = {k.lower(): v for k, v in (loaders or {}).items()}
    used: dict[str, str] = {}
    failures: list[tuple[str, str]] = []
    regressions: list[tuple[str, str]] = []
    cmp: dict[str, dict] = {}
    exact: dict[str, list[str]] = {}
    near: dict[str, list[str]] = {}
    exact_of: dict[str, str] = {}
    nb_bytes = nb_out = 0

    for path in sorted(folder.rglob("*")):
        rel = path.relative_to(folder).as_posix()
        if ".git" in path.relative_to(folder).parts or not (path.is_file() or path.is_symlink()):
            continue
        report.files_seen += 1
        ext = path.suffix.lower()
        size = path.stat().st_size if path.exists() else 0
        count, total = report.extensions.get(ext, (0, 0))
        report.extensions[ext] = (count + 1, total + size)

        def skip(reason: str) -> None:
            report.buckets["skipped"] += 1
            report.skipped[rel] = reason

        bucket, reason = classify_file(rel, symlink=path.is_symlink(), loaders=loaders)
        if bucket == "skipped":
            skip(reason)
            continue
        if bucket != "read":
            report.buckets[bucket] += 1
            if bucket == "code-config":
                report.code_config.append(rel)
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
        look_reader = look_notebook if ext == NOTEBOOK_EXT else look_text
        try:
            decoded = look_reader(raw)
        except LoadError as exc:
            skip(str(exc))
            continue
        except Exception:
            skip("unreadable")
            continue
        look = normalize_text(decoded)
        text = look
        loader = loaders.get(ext)
        if loader is not None:
            used[ext] = "loader"
            try:
                loaded = loader(raw)
            except LoadError as exc:
                skip(str(exc))
                if not str(exc).startswith("stub"):
                    regressions.append((rel, f"skipped-by-loader: {exc}"))
                continue
            except Exception:
                skip("unreadable")
                regressions.append((rel, "skipped-by-loader: unreadable"))
                continue
            text = loaded
            _check_loader(rel, raw, loader, look, loaded, failures)
            c = cmp.setdefault(ext, {"chars_before": 0, "chars_after": 0, "markup_before": 0, "markup_after": 0})
            c["chars_before"] += len(look)
            c["chars_after"] += len(loaded)
            c["markup_before"] += sum(markup_counts(look).values())
            c["markup_after"] += sum(markup_counts(loaded).values())
        elif ext in TEXT_EXTS or ext == NOTEBOOK_EXT:
            used[ext] = "look"
        report.buckets["read"] += 1

        # Problems in the input itself.
        if normalize_text(look) != look:
            report.idempotence_failures.append(rel)
        for kind, hit in (("bom", decoded.startswith(chr(0xFEFF))), ("crlf", "\r" in decoded),
                          ("not_nfc", unicodedata.normalize("NFC", decoded) != decoded)):
            if hit:
                report.raw[kind] += 1
                _add_example(report.raw_examples[kind], rel)
        if ext == NOTEBOOK_EXT:
            stats = _notebook_stats(raw)
            report.notebooks["files"] += 1
            report.notebooks["markdown_cells"] += stats["markdown"]
            report.notebooks["code_cells"] += stats["code"]
            nb_bytes += stats["bytes"]
            nb_out += stats["output_chars"]

        # Problems in the text the report is about.
        for kind, n in markup_counts(text).items():
            if n:
                report.markup[kind] += n
                _add_example(report.markup_examples[kind], f"{rel}: {_first_markup_line(text, kind)}")
        if _has_front_matter(text):
            report.front_matter += 1
        if len(text.strip()) < MIN_CHARS:
            report.tiny += 1
            _add_example(report.tiny_examples, rel)
        elif _prose_chars(text) < MIN_CHARS:
            report.stubs += 1
            _add_example(report.stub_examples, rel)
        for heading, body in sections(text):
            report.sections_total += 1
            if len(body) > section_cap:
                report.sections_over += 1
                _add_example(report.sections_over_examples, f"{rel}#{heading}")
            if len(body) < MIN_CHARS:
                report.sections_under += 1
        digest = _digest(text)
        exact_of[rel] = digest
        exact.setdefault(digest, []).append(rel)
        near.setdefault(_digest(_near_key(text)), []).append(rel)

    if nb_bytes:
        report.notebooks["output_share"] = round(nb_out / nb_bytes, 4)
    report.exact_duplicates = sorted(sorted(g) for g in exact.values() if len(g) > 1)
    report.near_duplicates = sorted(sorted(g) for g in near.values()
                                    if len(g) > 1 and len({exact_of[p] for p in g}) > 1)
    if loaders:
        report.loader = {"used": used, "failures": sorted(failures), "regressions": sorted(regressions)}
        report.compare = cmp
    return report


def _check_loader(rel: str, raw: bytes, loader: Reader, look: str, loaded: str, failures: list) -> None:
    look_len, out_len = len(look.strip()), len(loaded.strip())
    if look_len >= MIN_CHARS and out_len == 0:
        failures.append((rel, "empty-output"))
    elif look_len >= MIN_CHARS and out_len < 0.2 * look_len:
        failures.append((rel, "too-short"))
    if normalize_text(loaded) != loaded:
        failures.append((rel, "not-normalised"))
    if sum(markup_counts(loaded).values()) > sum(markup_counts(look).values()):
        failures.append((rel, "more-markup"))
    try:
        again = loader(raw)
    except Exception:
        again = None
    if again != loaded:
        failures.append((rel, "non-deterministic"))


def write_report(report: Report, out_dir: Path) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = report.snapshot_hash.split(":")[-1][:12]
    txt, js = out_dir / f"{report.source_id}-{prefix}.txt", out_dir / f"{report.source_id}-{prefix}.json"
    txt.write_text(report.render(), encoding="utf-8", newline="\n")
    js.write_text(json.dumps(report.to_json(), sort_keys=True, indent=2), encoding="utf-8", newline="\n")
    return txt, js


# --- command line -----------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect a pinned snapshot folder (spec 05).")
    parser.add_argument("folder", nargs="?", help="snapshot folder")
    parser.add_argument("--id", default="", help="source id used in the report name")
    parser.add_argument("--source", help="source id; with --register and --snapshots finds the folder")
    parser.add_argument("--register", help="folder of register records (yaml)")
    parser.add_argument("--snapshots", help="folder that holds <id>/<version>/ snapshots")
    parser.add_argument("--out", default="reports", help="where to write the report")
    parser.add_argument("--max-skip-share", type=float, default=0.25)
    parser.add_argument("--section-cap", type=int, default=6000)
    parser.add_argument("--loaders", choices=("look", "real"), default="look")
    parser.add_argument("--compare", action="store_true", help="show the loader before/after table")
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code or 0) if exc.code != 0 else 0

    source_id = args.id
    if args.source:
        if not (args.register and args.snapshots):
            print("--source needs --register and --snapshots", file=sys.stderr)
            return 2
        try:
            record = load_record(Path(args.register) / f"{args.source}.yaml")
        except (RegisterError, OSError) as exc:
            print(f"Cannot read the record for {args.source}: {exc}", file=sys.stderr)
            return 2
        folder = Path(args.snapshots) / record.id / record.version_ref
        source_id = source_id or record.id
    elif args.folder:
        folder = Path(args.folder)
    else:
        print("Give a snapshot folder, or --source with --register and --snapshots.", file=sys.stderr)
        return 2
    if not folder.is_dir():
        print(f"Not a folder: {folder}", file=sys.stderr)
        return 2

    report = inspect_folder(folder, source_id=source_id, loaders=dict(READERS) if args.loaders == "real" else None,
                            section_cap=args.section_cap, max_skip_share=args.max_skip_share, compare=args.compare)
    txt, js = write_report(report, Path(args.out))
    print(report.render())
    print(f"Saved: {txt}\nSaved: {js}")
    return 0 if report.gate_passed else 1
