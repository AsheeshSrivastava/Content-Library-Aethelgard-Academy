"""The readers in use: the best reader available for each extension. Each format spec adds its line here."""
from __future__ import annotations

from ragcore.errors import LoadError
from ragcore.files import MIN_CHARS
from ragcore.loaders.html import HTML_VERSION, read_html
from ragcore.loaders.markdown import MD_VERSION, read_md, read_mdx
from ragcore.loaders.notebook import NOTEBOOK_VERSION, read_ipynb
from ragcore.loaders.plain import PLAIN_READERS, Reader
from ragcore.loaders.rst import RST_VERSION, read_rst


def _no_stubs(read: Reader) -> Reader:
    """A page with under MIN_CHARS characters left after cleaning has nothing worth retrieving: name it a stub."""
    def checked(raw: bytes) -> str:
        text = read(raw)
        if not text.strip():
            raise LoadError("empty")
        if len(text.strip()) < MIN_CHARS:
            raise LoadError("stub")
        return text
    return checked


DEFAULT_READERS: dict[str, tuple[str, Reader]] = {
    **PLAIN_READERS,
    ".md": (MD_VERSION, _no_stubs(read_md)),
    ".mdx": (MD_VERSION, _no_stubs(read_mdx)),
    ".rst": (RST_VERSION, _no_stubs(read_rst)),
    ".html": (HTML_VERSION, _no_stubs(read_html)),
    ".htm": (HTML_VERSION, _no_stubs(read_html)),
    ".ipynb": (NOTEBOOK_VERSION, _no_stubs(read_ipynb)),
}
