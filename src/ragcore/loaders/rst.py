"""reStructuredText reader, version rst-1 (spec 08): the words a person would read, in Markdown form.

Headings become `#` lines, literal blocks and code directives become fenced blocks, roles and directives lose their
syntax. Code is never cleaned; tag syntax, markers and option lines may disappear, prose never changes or moves."""
from __future__ import annotations

import re

from ragcore.contracts import normalize_text
from ragcore.loaders.markdown import _segments

RST_VERSION = "rst-1"

FENCE_DIRECTIVES = {"code-block", "sourcecode", "code", "doctest", "testcode", "testsetup", "testoutput",
                    "testcleanup", "productionlist", "grammar-snippet", "parsed-literal"}
LANGUAGE_DIRECTIVES = {"code-block", "sourcecode", "code"}
DROP_DIRECTIVES = {"toctree", "index", "include", "literalinclude", "raw", "contents", "only", "module",
                   "currentmodule", "sectionauthor", "moduleauthor", "highlight", "default-domain", "rst-class",
                   "tabularcolumns", "audit-event", "audit-event-table", "availability", "image"}
VERSION_DIRECTIVES = {"versionadded", "versionchanged", "deprecated", "deprecated-removed", "versionremoved"}
DROP_ARGUMENT = {"figure"}
CODE_ROLES = {"class", "meth", "func", "mod", "module", "exc", "attr", "const", "data", "keyword", "deco", "decorator",
              "envvar", "option", "program", "file", "samp", "kbd", "pdbcmd", "opcode", "fixture", "confval", "obj",
              "type", "macro", "member", "expr", "code", "command"}

_DIRECTIVE = re.compile(r"^(?P<indent>[ \t]*)\.\.[ \t]+(?P<name>[\w:+.-]+)::(?:[ \t]+(?P<arg>.*?))?[ \t]*$")
_COMMENT = re.compile(r"^[ \t]*\.\.(?:[ \t]|$)")
_OPTION = re.compile(r"^[ \t]+:[\w-]+:(?:[ \t].*)?$")
_ADORNMENT = re.compile(r"^([^\w\s])\1+$")
_FENCE = re.compile(r"^[ \t]*(`{3,}|~{3,})(.*)$")

_LITERAL = re.compile(r"(?<!`)``(?!`)(.+?)(?<!`)``(?!`)")
_LINK = re.compile(r"(?<![\w:`])`([^`<>]+?)(?:[ \t]*<[^`>]*>)?`__?(?!\w)")
_ROLE = re.compile(r"(?<![\w`]):(?P<role>[\w:+.-]+):`(?P<body>[^`]*)`")
_TITLED = re.compile(r"^(?P<title>.*?)\s*<(?P<target>[^<>]*)>$", re.S)
_FOOTNOTE_REF = re.compile(r"[ \t]?\[(?:#[\w.-]*|\d+|[\w.-]+)\]_(?!\w)")


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" \t"))


def _dedent(lines: list[str]) -> list[str]:
    widths = [_indent(ln) for ln in lines if ln.strip()]
    cut = min(widths) if widths else 0
    return [ln[cut:] if ln.strip() else "" for ln in lines]


def _trim(lines: list[str]) -> list[str]:
    start, end = 0, len(lines)
    while start < end and not lines[start].strip():
        start += 1
    while end > start and not lines[end - 1].strip():
        end -= 1
    return lines[start:end]


def _collect(lines: list[str], start: int, base: int) -> tuple[list[str], int]:
    """The block after `start`: blank lines and lines indented deeper than `base`; trailing blanks are given back."""
    end = start
    while end < len(lines) and (not lines[end].strip() or _indent(lines[end]) > base):
        end += 1
    while end > start and not lines[end - 1].strip():
        end -= 1
    return lines[start:end], end


def _heading(lines: list[str], i: int, styles: dict, out: list[str]) -> int:
    """If a heading starts at line i, emit it and return the number of lines used, else 0."""
    line = lines[i]
    if _indent(line) or not line.strip():
        return 0
    stripped = line.strip()
    previous_blank = not out or not out[-1].strip()  # Judged on what is kept, so a second read agrees.
    if _ADORNMENT.match(stripped) and i + 2 < len(lines) and previous_blank:
        title, under = lines[i + 1], lines[i + 2]
        if (title.strip() and not _indent(title) and not _ADORNMENT.match(title.strip()) and under.strip() == stripped
                and len(stripped) >= len(title.strip())):
            level = styles.setdefault((stripped[0], True), len(styles) + 1)
            out.append("#" * level + " " + title.strip())
            return 3
    if i + 1 < len(lines) and previous_blank and not _ADORNMENT.match(stripped):
        under = lines[i + 1].strip()
        if (_ADORNMENT.match(under) and not _indent(lines[i + 1]) and len(under) >= len(stripped)
                and (len(under) >= 3 or len(under) >= len(stripped))):
            level = styles.setdefault((under[0], False), len(styles) + 1)
            out.append("#" * level + " " + stripped)
            return 2
    return 0


def _fence(out: list[str], body: list[str], language: str = "") -> None:
    out.append("```" + language)
    out.extend(_trim(_dedent(body)))
    out.append("```")


def _directive(lines: list[str], i: int, m: re.Match, styles: dict, out: list[str]) -> int:
    indent, name, arg = len(m["indent"]), m["name"].lower(), (m["arg"] or "").strip()
    base = name.split(":")[-1]
    j = i + 1
    while j < len(lines) and _OPTION.match(lines[j]) and _indent(lines[j]) > indent:
        j += 1
    body, end = _collect(lines, j, indent)
    if base in FENCE_DIRECTIVES:
        _fence(out, body, arg if base in LANGUAGE_DIRECTIVES else "")
    elif base in DROP_DIRECTIVES:
        pass
    elif base in VERSION_DIRECTIVES:
        cleaned = _trim(_process(_dedent(body), styles))
        if cleaned:
            cleaned[0] = f"{arg}: {cleaned[0]}" if arg else cleaned[0]
            out.extend(cleaned)
    else:
        if arg and base not in DROP_ARGUMENT:
            out.append(arg)
        out.extend(_process(_dedent(body), styles))
    return end


def _process(lines: list[str], styles: dict) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        fence = _FENCE.match(line)
        if fence and not (fence.group(1)[0] == "`" and "`" in fence.group(2)):
            out.append(line)
            i += 1
            while i < len(lines):
                out.append(lines[i])
                closing = _FENCE.match(lines[i])
                i += 1
                if (closing and closing.group(1)[0] == fence.group(1)[0] and len(closing.group(1)) >= len(fence.group(1))
                        and not closing.group(2).strip()):
                    break
            continue
        m = _DIRECTIVE.match(line)
        if m:
            i = _directive(lines, i, m, styles, out)
            continue
        if _COMMENT.match(line):
            _, i = _collect(lines, i + 1, _indent(line))
            continue
        used = _heading(lines, i, styles, out)
        if used:
            i += used
            continue
        stripped = line.strip()
        if (_ADORNMENT.match(stripped) and len(stripped) >= 4 and not _indent(line)
                and (not out or not out[-1].strip())):
            i += 1  # A transition.
            continue
        text = line.rstrip()
        if text.endswith("::"):
            prefix = text[:-2]
            if prefix.strip():
                out.append(prefix.rstrip() if prefix.endswith(" ") else prefix + ":")
            peek = i + 1
            while peek < len(lines) and not lines[peek].strip():
                peek += 1
            if peek < len(lines) and _indent(lines[peek]) > _indent(line):
                block, end = _collect(lines, peek, _indent(line))
                out.append("")
                _fence(out, block)
                i = end
                continue
            i += 1
            continue
        out.append(line)
        i += 1
    return out


def _role(m: re.Match) -> str:
    role, body = m["role"].lower(), m["body"]
    base = role.split(":")[-1]
    titled = _TITLED.match(body)
    text = (titled["title"] if titled and titled["title"].strip() else titled["target"] if titled else body).strip()
    if base in ("pep", "rfc"):
        return f"{base.upper()} {text}"
    code = base in CODE_ROLES
    if code and not (titled and titled["title"].strip()):
        if text.startswith("!"):
            text = text[1:]
        elif text.startswith("~"):
            text = re.split(r"\.|::", text[1:])[-1]
        elif text.startswith("."):
            text = text[1:]
    return f"`{text}`" if code else text


def _inline(chunk: str) -> str:
    spans: list[str] = []

    def mask(m: re.Match) -> str:
        spans.append(f"`{m.group(1)}`")
        return f"\x00{len(spans) - 1}\x00"

    chunk = _LITERAL.sub(mask, chunk)
    chunk = _LINK.sub(lambda m: m.group(1).strip(), chunk)
    chunk = _ROLE.sub(_role, chunk)
    chunk = _FOOTNOTE_REF.sub("", chunk)
    chunk = re.sub(r"[ \t]+$", "", chunk, flags=re.M)
    chunk = re.sub(r"\n{3,}", "\n\n", chunk)
    return re.sub(r"\x00(\d+)\x00", lambda m: spans[int(m.group(1))], chunk)


def _clean(text: str) -> str:
    text = "\n".join(_process(text.split("\n"), {}))
    pieces, after_code = [], False
    for is_code, piece in _segments(text):
        pieces.append(piece if is_code else re.sub(r"\A\n{2,}", "\n", _inline(piece)) if after_code else _inline(piece))
        after_code = is_code
    out = "".join(pieces).lstrip("\n")
    return out.rstrip("\n") + "\n" if out.strip() else ""


def read_rst(raw: bytes) -> str:
    text = _clean(normalize_text(raw.decode("utf-8")))
    for _ in range(3):  # Removing a construct can bring two lines together that now form a heading: read until stable.
        again = _clean(text)
        if again == text:
            break
        text = again
    return text