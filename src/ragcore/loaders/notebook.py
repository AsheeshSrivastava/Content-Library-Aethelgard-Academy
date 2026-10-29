"""Jupyter notebook reader, version ipynb-1 (spec 11): prose cells as prose, code cells as fenced code.

Outputs, raw cells, attachments and metadata are dropped; they change from run to run and are not the author's words."""
from __future__ import annotations

import json
import re

from ragcore.contracts import normalize_text
from ragcore.errors import LoadError
from ragcore.loaders.markdown import read_md

NOTEBOOK_VERSION = "ipynb-1"

_LANGUAGE = re.compile(r"[\w+#.-]+")


def _source(cell: dict) -> str:
    src = cell.get("source", "")
    return "".join(src) if isinstance(src, list) else str(src)


def _language(notebook: dict) -> str:
    meta = notebook.get("metadata")
    meta = meta if isinstance(meta, dict) else {}
    for key, field in (("language_info", "name"), ("kernelspec", "language")):
        value = (meta.get(key) or {}).get(field) if isinstance(meta.get(key), dict) else None
        if isinstance(value, str) and value:
            return value if _LANGUAGE.fullmatch(value) else ""
    return ""


def read_ipynb(raw: bytes) -> str:
    try:
        notebook = json.loads(raw.decode("utf-8-sig"))
    except json.JSONDecodeError:
        raise LoadError("bad-notebook") from None
    if not isinstance(notebook, dict) or not isinstance(notebook.get("cells"), list):
        raise LoadError("bad-notebook")
    if not isinstance(notebook.get("nbformat"), int) or notebook["nbformat"] < 4:
        raise LoadError("old-notebook")
    language = _language(notebook)
    parts: list[str] = []
    for cell in notebook["cells"]:
        if not isinstance(cell, dict):
            continue
        kind, text = cell.get("cell_type"), normalize_text(_source(cell))
        if kind == "markdown":
            text = read_md(text.encode("utf-8")).strip("\n")
        elif kind == "code" and text.strip():
            fence = "`" * max([2] + [len(run) for run in re.findall(r"`+", text)]) + "`"
            text = f"{fence}{language}\n{text.strip(chr(10))}\n{fence}"
        else:
            continue
        if text.strip():
            parts.append(text)
    return "\n\n".join(parts) + "\n" if parts else ""
