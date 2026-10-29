"""Turn a web page of a paper (arXiv's HTML) into plain Markdown. Standard library only; page code is never run."""
from __future__ import annotations

import re
from html.parser import HTMLParser

SKIPPED = {"script", "style", "svg", "nav", "header", "footer", "button", "noscript", "annotation", "annotation-xml"}
SKIPPED_CLASSES = ("ltx_bibliography", "ltx_page_footer", "ltx_role_footnote_mark")
VOID = {"img", "br", "hr", "meta", "link", "input", "source", "col", "area", "base", "wbr"}
BLOCKS = {"p", "li", "figcaption", "tr", "blockquote", "h1", "h2", "h3", "h4", "h5", "h6", "caption"}
HEADINGS = {"h1": "# ", "h2": "## ", "h3": "### ", "h4": "#### ", "h5": "#### ", "h6": "#### "}


class _Page(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.lines: list[str] = []
        self._buffer: list[str] = []
        self._prefix = ""
        self._skip_tag = ""  # Tag name whose subtree is being skipped (or the maths element).
        self._skip_depth = 0
        self._cells = 0

    def _flush(self) -> None:
        text = re.sub(r"\s+", " ", "".join(self._buffer)).strip()
        self._buffer = []
        if text:
            self.lines.append(f"{self._prefix}{text}" if self._prefix else text)
        self._prefix = ""

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if self._skip_tag:
            if tag == self._skip_tag:
                self._skip_depth += 1
            return
        classes = attributes.get("class") or ""
        if tag == "math":  # Keep the author's own maths text, skip the rendered markup.
            self._buffer.append(attributes.get("alttext") or "")
            self._skip_tag, self._skip_depth = "math", 1
            return
        if tag in SKIPPED or any(name in classes for name in SKIPPED_CLASSES):
            if tag not in VOID:
                self._skip_tag, self._skip_depth = tag, 1
            return
        if tag in BLOCKS:
            self._flush()
            self._prefix = HEADINGS.get(tag, "- " if tag == "li" else "> " if tag == "blockquote" else "")
            self._cells = 0
        elif tag in ("td", "th"):
            if self._cells:
                self._buffer.append(" | ")
            self._cells += 1

    def handle_endtag(self, tag):
        if self._skip_tag:
            if tag == self._skip_tag:
                self._skip_depth -= 1
                if self._skip_depth == 0:
                    self._skip_tag = ""
            return
        if tag in BLOCKS:
            self._flush()

    def handle_data(self, data):
        if not self._skip_tag:
            self._buffer.append(data)


def html_to_markdown(html: str) -> str:
    """Headings, paragraphs, list items and table rows as Markdown lines; scripts, navigation and references dropped."""
    page = _Page()
    page.feed(html)
    page.close()
    page._flush()
    return "\n\n".join(page.lines) + "\n"
