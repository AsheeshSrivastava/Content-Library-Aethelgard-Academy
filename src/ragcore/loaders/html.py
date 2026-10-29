"""HTML reader, version html-1 (spec 10): the words a person would read, as Markdown-style text.

Uses only the standard library parser. Scripts, styles, navigation and template syntax disappear; headings, lists,
tables, code and links keep their words."""
from __future__ import annotations

import re
from html.parser import HTMLParser

from ragcore.contracts import normalize_text

HTML_VERSION = "html-1"

_DROP = {"script", "style", "head", "nav", "footer", "template", "noscript", "svg", "iframe"}
_BLOCK = set("p div section article main header ul ol table tr dl dt dd blockquote figure details summary form".split())
_HEADING = re.compile(r"h([1-6])")
_TEMPLATE = re.compile(r"\{%.*?%\}|\{#.*?#\}|\{\{.*?\}\}", re.S)


class _Reader(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.out: list[str] = []
        self.pre: list[str] | None = None
        self.codes: list[str] = []
        self.drop = 0  # Depth inside a dropped element.
        self.title = ""
        self.in_title = False
        self.has_h1 = False
        self.first_cell = True

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "title":
            self.in_title = True
            return
        if tag in _DROP:
            self.drop += 1
            return
        if self.drop:
            return
        if tag == "pre":
            self.pre = []
        elif tag == "code" and self.pre is None:
            self.out.append("`")
        elif tag == "br":
            self.out.append("\n")
        elif tag == "li":
            self.out.append("\n- ")
        elif tag in ("td", "th"):
            self.out.append("" if self.first_cell else " | ")
            self.first_cell = False
        elif tag == "img":
            self.out.append(dict(attrs).get("alt") or "")
        elif m := _HEADING.fullmatch(tag):
            self.has_h1 = self.has_h1 or m.group(1) == "1"
            self.out.append("\n\n" + "#" * int(m.group(1)) + " ")
        elif tag == "tr":
            self.first_cell = True
            if not (self.out and self.out[-1].endswith("\n")):
                self.out.append("\n")
        elif tag in _BLOCK:
            self.out.append("\n\n")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _DROP:
            return  # A self-closing script or svg has no content to hide.
        self.handle_starttag(tag, attrs)
        if tag not in ("br", "img"):
            self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self.in_title = False
        elif tag in _DROP:
            self.drop = max(0, self.drop - 1)
        elif self.drop:
            return
        elif tag == "pre" and self.pre is not None:
            code = "".join(self.pre).strip("\n")
            self.pre = None
            n = max([2] + [len(r) for r in re.findall(r"`+", code)]) + 1
            self.codes.append("`" * n + "\n" + code + "\n" + "`" * n)
            self.out.append(f"\n\n\x00{len(self.codes) - 1}\x00\n\n")
        elif tag == "code" and self.pre is None:
            self.out.append("`")
        elif _HEADING.fullmatch(tag) or tag in _BLOCK:
            self.out.append("\n\n" if tag != "tr" else "\n")

    def handle_data(self, data: str) -> None:
        if self.in_title:
            self.title += data
        elif self.drop:
            return
        elif self.pre is not None:
            self.pre.append(data)
        else:
            self.out.append(re.sub(r"\s+", " ", data))


def read_html(raw: bytes) -> str:
    text = normalize_text(raw.decode("utf-8"))
    masked: list[str] = []

    def keep_pre(m: re.Match) -> str:
        masked.append(m.group(0))
        return f"\x01{len(masked) - 1}\x01"

    # Template syntax outside <pre> is removed per text node; inside <pre> it must survive, and the parser
    # would otherwise see "{% ... <tag> ... %}" as markup. Remove it from the whole text except <pre> blocks.
    text = re.sub(r"<pre\b.*?</pre\s*>", keep_pre, text, flags=re.S | re.I)
    text = _TEMPLATE.sub("", text)
    text = re.sub(r"\x01(\d+)\x01", lambda m: masked[int(m.group(1))], text)
    reader = _Reader()
    reader.feed(text)
    reader.close()
    body = "".join(reader.out)
    body = re.sub(r"[ \t]*\n[ \t]*", "\n", body)
    body = re.sub(r"\n-(?=\n|$)", "", body)  # An empty list item.
    body = re.sub(r" {2,}", " ", body)
    body = re.sub(r"\n{3,}", "\n\n", body).strip()
    title = re.sub(r"\s+", " ", reader.title).strip()
    if title and not reader.has_h1:
        body = f"# {title}\n\n{body}" if body else f"# {title}"
    body = re.sub(r"\x00(\d+)\x00", lambda m: reader.codes[int(m.group(1))], body)
    return body + "\n" if body.strip() else ""
