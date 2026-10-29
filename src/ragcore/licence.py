"""Licence detection from a licence file, and the cross-check against the declared licence (spec 04).

A person declares and approves; this module only warns. Text that does not clearly match one of the licences
below gives no match, never a guess. Phrase list adapted from production-rag `licenses.py`."""
from __future__ import annotations

import html
import re
from dataclasses import dataclass, replace
from pathlib import Path

MAX_LICENCE_FILE_BYTES = 200 * 1024
_WINDOW = 4000        # Only the start of a file says what the file is.
_TITLE_WINDOW = 300

_COPYRIGHT_LINE = re.compile(r"^[ \t]*(copyright|\(c\))\b.*$", re.IGNORECASE | re.MULTILINE)
_SPACES = re.compile(r"\s+")
_TAGS = re.compile(r"<(script|style)\b.*?</\1\s*>|<[^>]+>", re.IGNORECASE | re.DOTALL)
_INLINE_TAGS = re.compile(r"</?(b|i|u|em|strong|span|a|code|small|sup|sub)\b[^>]*>", re.IGNORECASE)  # no word break
_ISC = re.compile(r"permission to use, copy, modify,? (and/or|and) distribute this software "
                  r"for any purpose with or without fee is hereby granted")
_BSD_OPENING = ("redistribution and use in source and binary forms, with or without "
                "modification, are permitted provided that the following conditions are met")
_CC = re.compile(r"creative commons (attribution|attribution-sharealike) 4\.0 international public license")
_CC_IDS = {"attribution": "CC-BY-4.0", "attribution-sharealike": "CC-BY-SA-4.0"}


def _prepare(text: str) -> str:
    if re.search(r"<(html|!doctype|body)\b", text[:2000], re.IGNORECASE):
        text = html.unescape(_TAGS.sub(" ", _INLINE_TAGS.sub("", text)))
    without_copyright = _COPYRIGHT_LINE.sub(" ", text[: _WINDOW * 3])
    return _SPACES.sub(" ", without_copyright.lower()).strip()[:_WINDOW]


def _version(title: str, name: str) -> str | None:
    match = re.search(rf"{name}\W+version (\d+(?:\.\d+)?)", title)
    return match.group(1) if match else None


def detect_licence(text: str) -> str | None:
    """An SPDX id for a licence text, or None if unsure. Never raises."""
    try:
        body = _prepare(text)
        return _detect(body) if body else None
    except Exception:  # odd input must never stop a fetch
        return None


def _detect(body: str) -> str | None:
    title = body[:_TITLE_WINDOW]
    # GNU family is judged by the first licence name in the title: the GPL text itself mentions the others later.
    gnu = {name: title.find(name) for name in ("gnu affero general public license",
                                                "gnu lesser general public license", "gnu general public license")}
    gnu = {name: at for name, at in gnu.items() if at >= 0}
    if gnu:
        name = min(gnu, key=gnu.get)
        version = _version(title, name)
        if name.startswith("gnu affero"):
            return "AGPL-3.0" if version == "3" else None
        if name.startswith("gnu lesser"):
            return "LGPL-3.0" if version == "3" else None
        return {"3": "GPL-3.0", "2": "GPL-2.0"}.get(version or "")
    if "mozilla public license" in title:
        return "MPL-2.0" if _version(title, "mozilla public license") == "2.0" else None
    if "apache license" in title and _version(title, "apache license") == "2.0":
        return "Apache-2.0"
    if "cc0 1.0 universal" in title:
        return "CC0-1.0"
    if "this is free and unencumbered software released into the public domain" in body:
        return "Unlicense"
    cc = _CC.search(body)
    if cc:
        return _CC_IDS[cc.group(1)]
    if _BSD_OPENING in body:
        if "redistributions of source code must retain the above copyright notice" not in body:
            return None
        if "redistributions in binary form must reproduce the above copyright notice" not in body:
            return None
        if "advertising materials" in body:
            return None
        if "neither the name of" in body and "endorse or promote products" in body:
            return "BSD-3-Clause"
        return "BSD-2-Clause"
    if ("permission is hereby granted, free of charge, to any person obtaining a copy of this software and "
            "associated documentation files" in body and "the software is provided" in body):
        return "MIT"
    if _ISC.search(body) and "the software is provided" in body:
        return "ISC"
    return None


def read_licence_file(folder: Path, name: str) -> str | None:
    """The text of the licence file the fetch chose, or None (no name, per-document, missing, too large, not UTF-8)."""
    if not name or name != Path(name).name:
        return None
    path = Path(folder) / name
    try:
        if not path.is_file() or path.stat().st_size > MAX_LICENCE_FILE_BYTES:
            return None
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


@dataclass(frozen=True)
class Outcome:
    kind: str      # match / undetected / mismatch
    basis: str     # detected / declared
    warning: str = ""


def _norm(licence_id: str) -> str:
    value = licence_id.strip().lower()
    for suffix in ("-or-later", "-only"):  # the text cannot tell these apart
        if value.endswith(suffix):
            value = value[: -len(suffix)]
    return value


def cross_check(declared: str, detected: str | None) -> Outcome:
    """Compare the declared licence with what the file says. A mismatch is a warning, not an error."""
    if not detected:
        return Outcome("undetected", "declared")
    alternatives = {_norm(part) for part in re.split(r"\s+or\s+", declared.strip(), flags=re.IGNORECASE)}
    if _norm(detected) in alternatives:
        return Outcome("match", "detected")
    return Outcome("mismatch", "declared",
                   f"the licence file says {detected}, the record says {declared.strip()}")


def recheck_licence(record, detected: str | None):
    """Run the cross-check on a record again, for example after the declared id was corrected."""
    out = cross_check(record.licence.id, detected)
    return replace(record, licence=replace(record.licence, basis=out.basis), licence_warning=out.warning)
