"""Pinned fetch: an exact, hashed copy of a source and of its licence file. Never runs fetched files (spec 03)."""
from __future__ import annotations

import hashlib
import os
import re
import shutil
import stat
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path
from urllib.parse import urlparse

from ragcore.register import FULL_SHA, SourceRecord
from ragcore.webtext import html_to_markdown

# Dual-licensed repositories (such as uv) ship one file per licence; the MIT option is hashed.
LICENCE_NAMES = ("LICENSE", "LICENSE.md", "LICENSE.txt", "COPYING", "COPYING.md", "COPYING.txt", "LICENSE-MIT",
                 "LICENSE-APACHE")
GIT_TIMEOUT_S = 900
PER_DOCUMENT = "per-document"  # Name shown when the licence is stated inside each named file, not in a licence file.
WEB_TIMEOUT_S = 60
MAX_PAGE_BYTES = 20 * 1024 * 1024


class FetchError(RuntimeError):
    """A fetch that was refused or failed; nothing is left behind."""


@dataclass(frozen=True)
class FetchResult:
    ref: str
    content_hash: str
    licence_hash: str
    licence_name: str
    dropped: tuple[str, ...] = ()


def check_host(url: str, hosts: frozenset[str]) -> None:
    parsed = urlparse(url)
    ok = parsed.scheme == "https" and (parsed.hostname or "") in hosts and "@" not in parsed.netloc
    if not ok:
        raise FetchError(f"{url} is not an allowed source: only https addresses on {', '.join(sorted(hosts))} may be fetched.")


def _files(folder: Path) -> list[Path]:
    found = []
    for path in sorted(folder.rglob("*")):
        if ".git" in path.relative_to(folder).parts or path.is_symlink() or not path.is_file():
            continue
        found.append(path)
    return found


def snapshot_hash(folder: Path) -> str:
    """sha256 over the sorted relative paths and file bytes; the .git folder is not part of a snapshot."""
    digest = hashlib.sha256()
    for path in _files(folder):
        digest.update(path.relative_to(folder).as_posix().encode("utf-8") + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return "sha256:" + digest.hexdigest()


def drop_unsafe(folder: Path) -> list[str]:
    """Remove symlinks, which could point out of the snapshot; return what was removed."""
    dropped = []
    for path in sorted(folder.rglob("*")):
        if path.is_symlink():
            dropped.append(path.relative_to(folder).as_posix())
            path.unlink()
    return dropped


def _git(*args: str, cwd: Path | None = None) -> str:
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    try:
        done = subprocess.run(["git", "-c", "core.autocrlf=false", *args], cwd=cwd, env=env, check=True,
                              capture_output=True, text=True, timeout=GIT_TIMEOUT_S)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
        detail = (getattr(exc, "stderr", "") or str(exc)).strip().splitlines()[-1:] or [""]
        raise FetchError(f"Could not fetch or read the repository: {detail[0]}") from exc
    return done.stdout.strip()


def _remove(folder: Path) -> None:
    def writable(function, path, _info):  # git object files are read-only on Windows
        os.chmod(path, stat.S_IWRITE)
        function(path)

    shutil.rmtree(folder, onerror=writable)


def _temp_folder(parent: Path) -> Path:
    work = parent / f".tmp-{os.getpid()}-{int(time.time() * 1000)}"
    work.mkdir(parents=True)
    return work


def fetch_git(record: SourceRecord, snapshots: Path, hosts: frozenset[str], *, allow_local: bool = False) -> FetchResult:
    """Fetch at a pinned commit into snapshots/<id>/<commit>/ and hash it.

    The host check is made here, not by the caller. `allow_local` is a test hook that lets a file:// address through."""
    spec = record.fetch
    if spec is None or not spec.git:
        raise FetchError(f"{record.id}: the record has no git address to fetch.")
    url = spec.git
    if not (allow_local and url.startswith("file:///")):
        check_host(url, hosts)
    if spec.commit and not FULL_SHA.fullmatch(spec.commit):
        raise FetchError(f"{record.id}: commit '{spec.commit}' must be a full 40-character lowercase hex SHA.")
    paths, files = spec.paths, spec.files
    partial = url.startswith("https://")
    parent = snapshots / record.id
    work = _temp_folder(parent)
    try:
        repo = work / "repo"
        if spec.commit:  # Fetch exactly this commit; no branch is cloned.
            repo.mkdir()
            _git("init", "--quiet", cwd=repo)
            _git("remote", "add", "origin", url, cwd=repo)
            fetch = ["fetch", "--quiet", "--depth", "1"]
            if partial:
                _git("config", "remote.origin.promisor", "true", cwd=repo)
                _git("config", "remote.origin.partialclonefilter", "blob:none", cwd=repo)
                fetch += ["--filter=blob:none"]
            _git(*fetch, "origin", spec.commit, cwd=repo)
        else:
            clone = ["clone", "--quiet", "--depth", "1", "--no-checkout"]
            if spec.ref:
                clone += ["--branch", spec.ref]
            if partial:
                clone += ["--filter=blob:none"]
            _git(*clone, url, str(repo))
        if files:  # Named files need the pattern mode; folders and the root licence files are listed as patterns too.
            patterns = [*(f"/{name}" for name in LICENCE_NAMES), *(f"/{p.strip('/')}/" for p in paths),
                        *(f"/{f}" for f in files)]
            _git("sparse-checkout", "set", "--no-cone", *patterns, cwd=repo)
        elif paths:
            _git("sparse-checkout", "set", "--cone", *paths, cwd=repo)
        _git("checkout", "--quiet", *(["FETCH_HEAD"] if spec.commit else []), cwd=repo)
        sha = _git("rev-parse", "HEAD", cwd=repo)
        if spec.commit and sha != spec.commit:
            raise FetchError(f"{record.id}: asked for commit {spec.commit} but the checkout is {sha}.")
        _remove(repo / ".git")
        dropped = drop_unsafe(repo)
        size = sum(path.stat().st_size for path in _files(repo))
        if size > record.max_mb * 1024 * 1024:
            raise FetchError(f"The copy is larger than {record.max_mb} MB; raise max_mb in the record or choose fewer paths.")
        licence = next((repo / name for name in LICENCE_NAMES if (repo / name).is_file()), None)
        licence_hash = "sha256:" + hashlib.sha256(licence.read_bytes()).hexdigest() if licence else ""
        licence_name = licence.name if licence else ""
        if not licence and files:  # Licence stated inside each named document (the PEPs): hash those documents.
            digest = hashlib.sha256()
            for name in sorted(files):
                digest.update(name.encode() + b"\n" + (repo / name).read_bytes() + b"\n")
            licence_hash, licence_name = "sha256:" + digest.hexdigest(), PER_DOCUMENT
        result = FetchResult(sha, snapshot_hash(repo), licence_hash, licence_name, tuple(dropped))
        destination = parent / sha
        if destination.exists():
            _remove(destination)
        os.replace(repo, destination)
        return result
    finally:
        _remove(work)


def _download(url: str, hosts: frozenset[str]) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "ragcore/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=WEB_TIMEOUT_S) as response:
            check_host(response.geturl(), hosts)  # A redirect must stay on an allowed host.
            data = response.read(MAX_PAGE_BYTES + 1)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise FetchError(f"Could not download {url}: {exc}") from exc
    if len(data) > MAX_PAGE_BYTES:
        raise FetchError(f"{url} is larger than {MAX_PAGE_BYTES // (1024 * 1024)} MB.")
    return data


def fetch_web(record: SourceRecord, snapshots: Path, hosts: frozenset[str], *, download=None) -> FetchResult:
    """Pin one arXiv paper: its HTML page as paper.md and the licence text its abstract page declares, as LICENSE.

    The version name is taken from the text itself, so a changed page is a new pin."""
    download = download or _download
    spec = record.fetch
    if spec is None or not spec.web:
        raise FetchError(f"{record.id}: the record has no web address to fetch.")
    page_url, licence_url = spec.web, spec.licence_url
    check_host(page_url, hosts)
    check_host(licence_url, hosts)
    abstract = download(page_url.replace("/html/", "/abs/"), hosts).decode("utf-8", "replace")
    declared = re.sub(r"^https?://", "", licence_url.split("/legalcode")[0]).rstrip("/")
    if declared not in abstract.replace("http://", "https://").replace("https://", ""):
        raise FetchError(f"The paper's page does not declare the licence at {declared}; check the record.")
    text = html_to_markdown(download(page_url, hosts).decode("utf-8", "replace"))
    if len(text.split()) < 500:
        raise FetchError("The page gave too little text to be a paper; nothing was kept.")
    parent = snapshots / record.id
    work = _temp_folder(parent)
    try:
        folder = work / "doc"
        folder.mkdir()
        (folder / "paper.md").write_text(text, encoding="utf-8", newline="\n")
        licence = download(licence_url, hosts)
        (folder / "LICENSE").write_bytes(licence)
        ref = "web-" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
        result = FetchResult(ref, snapshot_hash(folder), "sha256:" + hashlib.sha256(licence).hexdigest(), "LICENSE")
        destination = parent / ref
        if destination.exists():
            _remove(destination)
        os.replace(folder, destination)
        return result
    finally:
        _remove(work)


def apply_fetch(record: SourceRecord, result: FetchResult, *, on: date) -> SourceRecord:
    """Record the pin. If the licence text changed since it was read, an approved or limited source goes back to quarantine."""
    status = record.status
    if record.licence.file_hash != result.licence_hash and status in ("approved", "limited"):
        status = "quarantine"
    return replace(record, version_ref=result.ref, fetched=on, content_hash=result.content_hash, status=status,
                   licence=replace(record.licence, file_hash=result.licence_hash))


def verify(record: SourceRecord, snapshots: Path) -> list[str]:
    """Problems with the stored snapshot of a pinned source; empty means no drift."""
    folder = snapshots / record.id / record.version_ref
    if not record.version_ref or not folder.is_dir():
        return [f"{record.id}: no stored snapshot for the pinned version; fetch it again."]
    if snapshot_hash(folder) != record.content_hash:
        return [f"{record.id}: the stored copy has changed since it was pinned."]
    return []
