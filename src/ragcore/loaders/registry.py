"""The readers in use: the best reader available for each extension. Each format spec adds its line here."""
from __future__ import annotations

from ragcore.errors import LoadError
from ragcore.files import MIN_CHARS
from ragcore.loaders.markdown import MD_VERSION, read_md, read_mdx
from ragcore.loaders.plain import PLAIN_READERS, Reader


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
}
