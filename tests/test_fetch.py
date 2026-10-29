"""Spec 03: pinned fetch. Offline only; the network test lives in test_fetch_network.py. Written before the code."""
import hashlib
import os
import subprocess
import urllib.error
from dataclasses import replace
from datetime import date

import pytest

from ragcore import fetch
from ragcore.fetch import FetchError, FetchResult, apply_fetch, check_host, fetch_git, fetch_web, snapshot_hash, verify
from ragcore.register import FetchSpec, Licence, SourceRecord

HOSTS = frozenset({"github.com", "arxiv.org", "creativecommons.org"})
ON = date(2026, 10, 29)


def git(cwd, *args):
    done = subprocess.run(
        ["git", "-c", "user.name=T", "-c", "user.email=t@example.org", "-c", "core.autocrlf=false", *args],
        cwd=cwd, check=True, capture_output=True, text=True)
    return done.stdout.strip()


def put(root, files):
    for name, data in files.items():
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data if isinstance(data, bytes) else data.encode())


@pytest.fixture
def repo(tmp_path):
    """A local repository with two commits. Returns (path, first_sha, second_sha)."""
    root = tmp_path / "origin"
    root.mkdir()
    git(root, "init", "-q", "-b", "main")
    git(root, "config", "uploadpack.allowAnySHA1InWant", "true")
    put(root, {"LICENSE": "MIT licence text\n", "docs/a.md": "# A\nfirst\n", "docs/b.md": "# B\n",
               "other/c.md": "# C\n", "top.md": "top\n"})
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "one")
    first = git(root, "rev-parse", "HEAD")
    git(root, "tag", "v1")
    put(root, {"docs/a.md": "# A\nsecond\n", "docs/new.md": "new\n"})
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "two")
    return root, first, git(root, "rev-parse", "HEAD")


def rec(url, **fetch_kw):
    return SourceRecord(
        id="demo", title="T", owner="O", url="https://example.org", kind="docs", area="python",
        licence=Licence(id="MIT", file_url="https://x/LICENSE", covers="both", file_hash="old"),
        version_ref="", fetched=None, content_hash="", status="approved",
        fetch=FetchSpec(git=url, **fetch_kw), max_mb=5)


def run(repo_path, snapshots, **kw):
    allow = kw.pop("allow_local", True)
    return fetch_git(rec(repo_path.as_uri(), **kw), snapshots, HOSTS, allow_local=allow)


# --- host check -------------------------------------------------------------

@pytest.mark.parametrize("url", [
    "http://github.com/a/b", "https://evil.org/a/b", "https://github.com.evil.org/a/b",
    "https://user:pw@github.com/a/b", "https://user@github.com/a/b", "ftp://github.com/a", "github.com/a/b", "",
])
def test_check_host_rejects(url):
    with pytest.raises(FetchError):
        check_host(url, HOSTS)


def test_check_host_accepts():
    check_host("https://github.com/a/b", HOSTS)


def test_fetch_git_checks_host_itself(tmp_path):
    for url in ("http://github.com/a/b", "https://evil.org/a/b"):
        with pytest.raises(FetchError, match="not an allowed source"):
            fetch_git(rec(url), tmp_path, HOSTS)
    assert not any(tmp_path.iterdir())


def test_local_urls_refused_without_the_test_hook(repo, tmp_path):
    with pytest.raises(FetchError, match="not an allowed source"):
        run(repo[0], tmp_path / "snap", allow_local=False)


# --- clone, hashing, layout --------------------------------------------------

def test_fetch_records_sha_and_layout(repo, tmp_path):
    path, first, second = repo
    snap = tmp_path / "snap"
    res = run(path, snap)
    assert isinstance(res, FetchResult)
    assert res.ref == second
    folder = snap / "demo" / second
    assert (folder / "docs" / "a.md").read_text() == "# A\nsecond\n"
    assert not (folder / ".git").exists()
    assert res.content_hash == snapshot_hash(folder)
    assert res.licence_name == "LICENSE"
    assert res.licence_hash == "sha256:" + hashlib.sha256(b"MIT licence text\n").hexdigest()
    assert [p.name for p in (snap / "demo").iterdir()] == [second]  # no temp folder left


def test_ref_tag_is_honoured(repo, tmp_path):
    path, first, _ = repo
    res = run(path, tmp_path / "snap", ref="v1")
    assert res.ref == first


def test_same_input_same_hash_and_one_byte_changes_it(repo, tmp_path):
    path, _, second = repo
    h1 = run(path, tmp_path / "s1").content_hash
    h2 = run(path, tmp_path / "s2").content_hash
    assert h1 == h2
    folder = tmp_path / "s2" / "demo" / second
    f = folder / "top.md"
    f.write_bytes(f.read_bytes() + b"x")
    assert snapshot_hash(folder) != h1


def test_hash_depends_on_path_names(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    put(a, {"x.md": "same"})
    put(b, {"y.md": "same"})
    assert snapshot_hash(a) != snapshot_hash(b)


def test_sparse_paths_and_files(repo, tmp_path):
    path, _, second = repo
    folder = tmp_path / "snap" / "demo" / second
    run(path, tmp_path / "snap", paths=("docs",))
    assert (folder / "docs" / "b.md").exists() and (folder / "LICENSE").exists()
    assert not (folder / "other").exists()
    assert (folder / "top.md").exists()  # git's cone mode always keeps files at the repository root
    run(path, tmp_path / "snap2", files=("other/c.md",))
    folder2 = tmp_path / "snap2" / "demo" / second
    assert (folder2 / "other" / "c.md").exists() and (folder2 / "LICENSE").exists()
    assert not (folder2 / "docs").exists()


def test_symlink_is_dropped_and_reported(repo, tmp_path):
    path, _, _ = repo
    try:
        os.symlink("top.md", tmp_path / "probe")
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not available here")
    os.symlink("top.md", path / "link.md")
    git(path, "add", "-A")
    git(path, "commit", "-q", "-m", "link")
    res = run(path, tmp_path / "snap")
    assert res.dropped == ("link.md",)
    assert not (tmp_path / "snap" / "demo" / res.ref / "link.md").exists()


def test_size_cap_refuses_and_leaves_nothing(repo, tmp_path):
    path, _, _ = repo
    put(path, {"big.bin": b"0" * (6 * 1024 * 1024)})
    git(path, "add", "-A")
    git(path, "commit", "-q", "-m", "big")
    snap = tmp_path / "snap"
    with pytest.raises(FetchError, match="larger than 5 MB"):
        run(path, snap)
    assert list((snap / "demo").iterdir()) == []


@pytest.mark.parametrize("name", ["LICENSE", "LICENSE.md", "LICENSE.txt", "COPYING", "COPYING.md", "COPYING.txt",
                                  "LICENSE-MIT", "LICENSE-APACHE"])
def test_each_licence_file_name_is_found(tmp_path, name):
    root = tmp_path / "o"
    root.mkdir()
    git(root, "init", "-q", "-b", "main")
    put(root, {name: "terms\n", "a.md": "a\n"})
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "x")
    res = run(root, tmp_path / "snap")
    assert res.licence_name == name


def test_dual_licence_uses_mit(tmp_path):
    root = tmp_path / "o"
    root.mkdir()
    git(root, "init", "-q", "-b", "main")
    put(root, {"LICENSE-APACHE": "apache\n", "LICENSE-MIT": "mit\n", "a.md": "a\n"})
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "x")
    res = run(root, tmp_path / "snap")
    assert res.licence_name == "LICENSE-MIT"
    assert res.licence_hash == "sha256:" + hashlib.sha256(b"mit\n").hexdigest()


def test_per_document_licence(tmp_path):
    root = tmp_path / "o"
    root.mkdir()
    git(root, "init", "-q", "-b", "main")
    put(root, {"peps/pep-1.rst": "one\n", "peps/pep-2.rst": "two\n"})
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "x")
    a = run(root, tmp_path / "s1", files=("peps/pep-1.rst", "peps/pep-2.rst"))
    assert a.licence_name == "per-document" and a.licence_hash.startswith("sha256:")
    b = run(root, tmp_path / "s2", files=("peps/pep-2.rst", "peps/pep-1.rst"))
    assert a.licence_hash == b.licence_hash  # order of the list does not matter
    c = run(root, tmp_path / "s3", files=("peps/pep-1.rst",))
    assert c.licence_hash != a.licence_hash


def test_refetch_same_sha_replaces_cleanly(repo, tmp_path):
    path, _, second = repo
    snap = tmp_path / "snap"
    run(path, snap)
    stale = snap / "demo" / second / "stale.txt"
    stale.write_text("old")
    res = run(path, snap)
    assert not stale.exists()
    assert [p.name for p in (snap / "demo").iterdir()] == [second]
    assert res.content_hash == snapshot_hash(snap / "demo" / second)


def test_interrupted_copy_leaves_no_folder(repo, tmp_path, monkeypatch):
    def boom(*_a, **_k):
        raise OSError("disk full")
    monkeypatch.setattr(fetch.os, "replace", boom)
    snap = tmp_path / "snap"
    with pytest.raises(OSError):
        run(repo[0], snap)
    assert list((snap / "demo").iterdir()) == []


def test_bad_clone_url_raises_fetch_error(tmp_path):
    with pytest.raises(FetchError):
        fetch_git(rec((tmp_path / "nothing").as_uri()), tmp_path / "snap", HOSTS, allow_local=True)
    assert list((tmp_path / "snap" / "demo").iterdir()) == []


# --- pin by exact commit SHA -------------------------------------------------

def test_pin_by_sha_gives_that_commit(repo, tmp_path):
    path, first, second = repo
    res = run(path, tmp_path / "snap", commit=first)
    assert res.ref == first
    folder = tmp_path / "snap" / "demo" / first
    assert (folder / "docs" / "a.md").read_text() == "# A\nfirst\n"
    assert not (folder / "docs" / "new.md").exists() and not (folder / ".git").exists()
    assert [p.name for p in (tmp_path / "snap" / "demo").iterdir()] == [first]


def test_pin_by_sha_same_hash_as_ref_fetch(repo, tmp_path):
    path, first, _ = repo
    by_ref = run(path, tmp_path / "s1", ref="v1")
    by_sha = run(path, tmp_path / "s2", commit=first)
    assert (by_sha.ref, by_sha.content_hash, by_sha.licence_hash) == (by_ref.ref, by_ref.content_hash, by_ref.licence_hash)


def test_pin_by_sha_with_sparse_paths(repo, tmp_path):
    path, first, _ = repo
    run(path, tmp_path / "snap", commit=first, paths=("docs",))
    folder = tmp_path / "snap" / "demo" / first
    assert (folder / "docs" / "b.md").exists() and not (folder / "other").exists()


@pytest.mark.parametrize("bad", ["abc1234", "main", "v1", "A" * 40, "a" * 39, "a" * 41])
def test_pin_refuses_non_full_sha(repo, tmp_path, bad):
    snap = tmp_path / "snap"
    with pytest.raises(FetchError, match="commit"):
        run(repo[0], snap, commit=bad)
    assert not snap.exists() or list(snap.rglob("*")) == [] or list((snap / "demo").iterdir()) == []


def test_pin_unknown_sha_fails_and_leaves_nothing(repo, tmp_path):
    snap = tmp_path / "snap"
    with pytest.raises(FetchError):
        run(repo[0], snap, commit="b" * 40)
    assert list((snap / "demo").iterdir()) == []


def test_pin_refuses_when_head_differs(repo, tmp_path, monkeypatch):
    path, first, second = repo
    real = fetch._git

    def lying(*args, cwd=None):
        out = real(*args, cwd=cwd)
        return second if args[:2] == ("rev-parse", "HEAD") else out
    monkeypatch.setattr(fetch, "_git", lying)
    snap = tmp_path / "snap"
    with pytest.raises(FetchError, match=first):
        run(path, snap, commit=first)
    assert list((snap / "demo").iterdir()) == []


# --- apply_fetch and verify ----------------------------------------------------

def test_apply_fetch_quarantines_on_changed_licence(repo, tmp_path):
    res = FetchResult("a" * 40, "sha256:c", "sha256:new", "LICENSE")
    r = rec("https://github.com/a/b")
    out = apply_fetch(r, res, on=ON)
    assert out.status == "quarantine" and out.licence.file_hash == "sha256:new"
    assert (out.version_ref, out.fetched, out.content_hash) == ("a" * 40, ON, "sha256:c")
    limited = apply_fetch(replace(r, status="limited"), res, on=ON)
    assert limited.status == "quarantine"


def test_apply_fetch_keeps_status_when_licence_unchanged():
    res = FetchResult("a" * 40, "sha256:c", "old", "LICENSE")
    for status in ("approved", "limited", "quarantine", "reference-only"):
        assert apply_fetch(replace(rec("https://github.com/a/b"), status=status), res, on=ON).status == status


def test_apply_fetch_does_not_touch_other_statuses_on_licence_change():
    res = FetchResult("a" * 40, "sha256:c", "sha256:new", "LICENSE")
    for status in ("reference-only", "rejected"):
        assert apply_fetch(replace(rec("https://github.com/a/b"), status=status), res, on=ON).status == status


def test_verify(repo, tmp_path):
    snap = tmp_path / "snap"
    res = run(repo[0], snap)
    r = apply_fetch(rec(repo[0].as_uri()), res, on=ON)
    assert verify(r, snap) == []
    assert "no stored snapshot" in verify(replace(r, version_ref="c" * 40), snap)[0]
    assert "no stored snapshot" in verify(replace(r, version_ref=""), snap)[0]
    f = snap / "demo" / res.ref / "top.md"
    f.write_bytes(f.read_bytes() + b"!")
    assert "changed" in verify(r, snap)[0]


# --- web fetch (stubbed downloader) -------------------------------------------

PAGE_URL = "https://arxiv.org/html/2610.00001"
LICENCE_URL = "https://creativecommons.org/licenses/by/4.0/legalcode"
BODY = "<html><body><h1>Title</h1>" + "".join(f"<p>{'word ' * 50}{i}</p>" for i in range(12)) + "</body></html>"
ABSTRACT = "Licence: https://creativecommons.org/licenses/by/4.0/"


def web_rec(page=PAGE_URL, licence=LICENCE_URL):
    return replace(rec(""), fetch=FetchSpec(web=page, licence_url=licence))


def stub(pages):
    def download(url, hosts):
        check_host(url, hosts)
        if url not in pages:
            raise FetchError(f"Could not download {url}")
        return pages[url]
    return download


def good_pages(**over):
    pages = {PAGE_URL.replace("/html/", "/abs/"): ABSTRACT.encode(), PAGE_URL: BODY.encode(), LICENCE_URL: b"CC BY 4.0 text"}
    pages.update(over)
    return pages


def test_web_fetch_ok(tmp_path):
    res = fetch_web(web_rec(), tmp_path, HOSTS, download=stub(good_pages()))
    assert res.ref.startswith("web-") and len(res.ref) == 16
    folder = tmp_path / "demo" / res.ref
    assert (folder / "paper.md").read_text(encoding="utf-8").startswith("# Title")
    assert (folder / "LICENSE").read_bytes() == b"CC BY 4.0 text"
    assert res.licence_name == "LICENSE" and res.content_hash == snapshot_hash(folder)


def test_web_changed_page_is_new_pin(tmp_path):
    a = fetch_web(web_rec(), tmp_path, HOSTS, download=stub(good_pages()))
    b = fetch_web(web_rec(), tmp_path, HOSTS, download=stub(good_pages(**{PAGE_URL: BODY.replace("Title", "Other").encode()})))
    assert a.ref != b.ref


def test_web_licence_not_declared(tmp_path):
    pages = good_pages(**{PAGE_URL.replace("/html/", "/abs/"): b"All rights reserved"})
    with pytest.raises(FetchError, match="does not declare"):
        fetch_web(web_rec(), tmp_path, HOSTS, download=stub(pages))
    assert not list(tmp_path.rglob("*.md"))


def test_web_too_little_text(tmp_path):
    with pytest.raises(FetchError, match="too little"):
        fetch_web(web_rec(), tmp_path, HOSTS, download=stub(good_pages(**{PAGE_URL: b"<p>short</p>"})))
    assert list((tmp_path / "demo").iterdir()) == [] if (tmp_path / "demo").exists() else True


def test_web_urls_checked_for_host(tmp_path):
    with pytest.raises(FetchError, match="not an allowed source"):
        fetch_web(web_rec(page="https://evil.org/html/1"), tmp_path, HOSTS, download=stub(good_pages()))
    with pytest.raises(FetchError, match="not an allowed source"):
        fetch_web(web_rec(licence="http://creativecommons.org/x"), tmp_path, HOSTS, download=stub(good_pages()))


class FakeResponse:
    def __init__(self, data, url):
        self._data, self._url = data, url

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def geturl(self):
        return self._url

    def read(self, n=-1):
        return self._data if n < 0 else self._data[:n]


def test_download_refuses_redirect_off_host(monkeypatch):
    monkeypatch.setattr(fetch.urllib.request, "urlopen", lambda *a, **k: FakeResponse(b"x", "https://evil.org/p"))
    with pytest.raises(FetchError, match="not an allowed source"):
        fetch._download("https://arxiv.org/abs/1", HOSTS)


def test_download_refuses_oversize(monkeypatch):
    monkeypatch.setattr(fetch, "MAX_PAGE_BYTES", 10)
    monkeypatch.setattr(fetch.urllib.request, "urlopen", lambda *a, **k: FakeResponse(b"x" * 50, "https://arxiv.org/p"))
    with pytest.raises(FetchError, match="larger than"):
        fetch._download("https://arxiv.org/abs/1", HOSTS)


def test_download_wraps_network_errors(monkeypatch):
    def fail(*a, **k):
        raise urllib.error.URLError("down")
    monkeypatch.setattr(fetch.urllib.request, "urlopen", fail)
    with pytest.raises(FetchError, match="Could not download"):
        fetch._download("https://arxiv.org/abs/1", HOSTS)
