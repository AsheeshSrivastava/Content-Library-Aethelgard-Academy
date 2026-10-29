"""Markdown and MDX reader, version md-1 (spec 07): the words a person would read, without markup syntax.

Fenced code blocks and inline code spans are never changed. Everything else is cleaned by fixed rules; tag syntax,
attributes and ESM lines may disappear, prose never changes or moves."""
from __future__ import annotations

import re

import yaml

from ragcore.contracts import normalize_text

MD_VERSION = "md-1"

_FENCE = re.compile(r"^[ \t]*(`{3,}|~{3,})(.*)$")
_INLINE_CODE = re.compile(r"(?<!`)(`+)(?!`)(.+?)(?<!`)\1(?!`)", re.S)
_FRONT_MATTER = re.compile(r"\A---[ \t]*\n(.*?)\n---[ \t]*(?:\n|\Z)", re.S)
_H1 = re.compile(r"#[ \t]")

_NESTED = r"\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}"
_ATTRS = rf"""(?:\s+[^\s<>/="']+(?:\s*=\s*(?:"[^"]*"|'[^']*'|{_NESTED}|[^\s<>"']+))?)*"""
_HTML_TAGS = ("div span details summary iframe img br hr p center picture source video a abbr sup sub kbd table "
              "thead tbody tr td th ul ol li").split()

_COMMENT = re.compile(r"<!--.*?-->|\{/\*.*?\*/\}", re.S)
_IMPORT = re.compile(
    r"""^[ \t]*import\s+(?:(?:[\w$]+|\*\s+as\s+[\w$]+|\{[^}]*\})(?:\s*,\s*(?:\*\s+as\s+[\w$]+|\{[^}]*\}))?\s+from\s+)?"""
    r"""(["'])[^"'\n]+\1;?[ \t]*(?:\n|\Z)""", re.M)
_EXPORT = re.compile(r"^[ \t]*export\s+(?:const|let|var|function|default|\{)[^\n]*(?:\n|\Z)", re.M)
_IMG = re.compile(rf"<img\b{_ATTRS}\s*/?>", re.I)
_ALT = re.compile(r"""\balt\s*=\s*(?:"([^"]*)"|'([^']*)')""", re.I)
_EMBED = re.compile(rf"<(iframe|video)\b{_ATTRS}\s*/?>.*?</\1\s*>|<(?:iframe|video)\b{_ATTRS}\s*/>",
                    re.S | re.I)
_BR = re.compile(r"<br\s*/?>", re.I)
_JSX = re.compile(rf"(?<![\w`])<[A-Z][\w.:-]*{_ATTRS}\s*/?>|</[A-Z][\w.:-]*\s*>")  # Not "List<String>".
_HTML = re.compile(rf"</?(?:{'|'.join(_HTML_TAGS)})\b{_ATTRS}\s*/?>", re.I)


def _segments(text: str) -> list[tuple[bool, str]]:
    """Split into (is_code, text) pieces; a fence stays open to the end of the text if never closed."""
    out: list[tuple[bool, str]] = []
    prose: list[str] = []
    code: list[str] = []
    fence = ""
    for line in text.splitlines(keepends=True):
        m = _FENCE.match(line.rstrip("\n"))
        if not fence:
            if m and not (m.group(1)[0] == "`" and "`" in m.group(2)):
                if prose:
                    out.append((False, "".join(prose)))
                    prose = []
                fence, code = m.group(1), [line]
            else:
                prose.append(line)
        else:
            code.append(line)
            if m and m.group(1)[0] == fence[0] and len(m.group(1)) >= len(fence) and not m.group(2).strip():
                out.append((True, "".join(code)))
                fence, code = "", []
    if code:
        out.append((True, "".join(code)))
    if prose:
        out.append((False, "".join(prose)))
    return out


def _front_matter(text: str) -> str:
    m = _FRONT_MATTER.match(text)
    if not m:
        return text
    try:
        meta = yaml.safe_load(m.group(1))
    except yaml.YAMLError:
        return text
    if not isinstance(meta, dict):
        return text
    body = text[m.end():]
    title = str(meta.get("title") or "").strip()
    first = next((ln for ln in body.splitlines() if ln.strip()), "")
    if title and not _H1.match(first):
        body = f"# {title}\n\n{body}"
    return body


def _clean_prose(chunk: str, *, esm: bool) -> str:
    spans: list[str] = []

    def mask(m: re.Match) -> str:
        spans.append(m.group(0))
        return f"\x00{len(spans) - 1}\x00"

    chunk = _INLINE_CODE.sub(mask, chunk)
    chunk = _COMMENT.sub("", chunk)
    if esm:
        chunk = _EXPORT.sub("", _IMPORT.sub("", chunk))

    def alt(m: re.Match) -> str:
        found = _ALT.search(m.group(0))
        return (found.group(1) or found.group(2) or "").strip() if found else ""

    chunk = _IMG.sub(alt, chunk)
    chunk = _EMBED.sub("", chunk)
    chunk = _BR.sub("\n", chunk)
    chunk = _JSX.sub("", chunk)
    chunk = _HTML.sub("", chunk)
    chunk = re.sub(r"[ \t]+$", "", chunk, flags=re.M)
    chunk = re.sub(r"\n{3,}", "\n\n", chunk)
    return re.sub(r"\x00(\d+)\x00", lambda m: spans[int(m.group(1))], chunk)


def _read(raw: bytes, *, esm: bool) -> str:
    text = _front_matter(normalize_text(raw.decode("utf-8")))
    out = "".join(piece if is_code else _clean_prose(piece, esm=esm) for is_code, piece in _segments(text))
    out = out.lstrip("\n")
    return out.rstrip("\n") + "\n" if out.strip() else ""


def read_md(raw: bytes) -> str:
    return _read(raw, esm=False)


def read_mdx(raw: bytes) -> str:
    return _read(raw, esm=True)
