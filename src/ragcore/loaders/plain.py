"""The stand-in reader for text types that have no real reader yet: UTF-8, no markup removal (spec 06)."""
from __future__ import annotations

from typing import Callable

from ragcore.contracts import normalize_text
from ragcore.files import TEXT_EXTS

Reader = Callable[[bytes], str]
PLAIN_VERSION = "plain-1"
PLAIN_READERS: dict[str, tuple[str, Reader]] = {ext: (PLAIN_VERSION, lambda raw: normalize_text(raw.decode("utf-8"))) for ext in TEXT_EXTS}
