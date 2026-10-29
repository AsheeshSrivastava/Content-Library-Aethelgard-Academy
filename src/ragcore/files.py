"""Which bucket a snapshot file belongs to, shared by the inspection script and the snapshot loader (spec 06).

Every file under a snapshot folder (never `.git`) lands in exactly one of: read, licence, ignored, code-config,
skipped (with a reason code)."""
from __future__ import annotations

from pathlib import PurePosixPath
from typing import Collection

from ragcore.fetch import LICENCE_NAMES

TEXT_EXTS = (".md", ".mdx", ".rst", ".txt", ".html", ".htm")
NOTEBOOK_EXT = ".ipynb"
IGNORED_EXTS = frozenset({
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".bmp", ".tif", ".tiff", ".svg",
    ".woff", ".woff2", ".ttf", ".otf", ".eot", ".mp3", ".mp4", ".wav", ".webm", ".mov",
    ".zip", ".gz", ".tar", ".tgz", ".7z", ".css", ".js", ".map", ".lock",
})
CODE_CONFIG_EXTS = frozenset({
    ".py", ".pyi", ".ts", ".tsx", ".jsx", ".mjs", ".cjs", ".sh", ".bat", ".yml", ".yaml", ".json",
    ".jsonc", ".toml", ".cfg", ".ini", ".cff", ".xml", ".env", ".dot", ".inc",
})
CODE_CONFIG_NAMES = frozenset({
    "makefile", "dockerfile", ".gitkeep", ".gitignore", ".gitattributes", ".editorconfig", ".prettierignore",
    ".prettierrc", ".ignore", ".git-blame-ignore-revs", ".htaccess", ".gitmodules", ".npmrc", ".nvmrc",
    ".python-version", ".python-versions", ".textlintignore", ".dockerignore",
})
MAX_FILE_BYTES = 2 * 1024 * 1024
MIN_CHARS = 100  # Under this many characters a page or section is too small to stand alone.


def is_licence(rel: str) -> bool:
    """A licence file: only at the folder root. The same name deeper down (vendored code) is an ordinary file."""
    name = rel.upper()
    return "/" not in rel and (name in {n.upper() for n in LICENCE_NAMES} or name.startswith(("LICENSE", "LICENCE")))


def classify_file(rel: str, *, symlink: bool = False, loaders: Collection[str] = ()) -> tuple[str, str]:
    """(bucket, reason) for a file by its relative posix path. The reason is set only for `skipped`.
    `loaders` holds the lower-case extensions that have a registered reader; they win over code-config and no-reader."""
    if symlink:
        return "skipped", "symlink"
    if is_licence(rel):
        return "licence", ""
    name = PurePosixPath(rel).name
    ext = PurePosixPath(rel).suffix.lower()
    if ext in IGNORED_EXTS:
        return "ignored", ""
    if ext not in loaders and (ext in CODE_CONFIG_EXTS or name.lower() in CODE_CONFIG_NAMES):
        return "code-config", ""
    if ext not in TEXT_EXTS and ext != NOTEBOOK_EXT and ext not in loaders:
        return "skipped", "no-reader"
    return "read", ""


def raw_problem(raw: bytes, max_file_bytes: int = MAX_FILE_BYTES) -> str:
    """The skip reason for file bytes that cannot be read at all, or "" when they can."""
    if len(raw) > max_file_bytes:
        return "too-large"
    if not raw.strip():
        return "empty"
    if b"\x00" in raw:
        return "binary"
    return ""
