"""Spec 06: the snapshot loader. Written before the code."""
import hashlib
import shutil
from dataclasses import replace

import pytest

from ragcore.contracts import Source, normalize_text
from ragcore.errors import LoadRefused, SkipShareExceeded
from ragcore.gate import LicenceRules, licence_for_path
from ragcore.loaders.snapshot import PLAIN_VERSION, LoadResult, load_snapshot
from ragcore.register import FetchSpec, Licence, SourceRecord

COMMIT = "c" * 40
LONG = "A sentence about retrieval. " * 10
DECOMPOSED_E = chr(0x65) + chr(0x301)


def rules(tmp_path):
    path = tmp_path / "licences.yaml"
    path.write_text("allowed: [MIT, Apache-2.0]\nshare-alike: [CC-BY-SA-4.0]\nnon-commercial: [CC-BY-NC-4.0]\n",
                    encoding="utf-8")
    return LicenceRules.load(path)


def record(status="approved", lic="MIT", covers="both", **kw):
    kw.setdefault("fetch", FetchSpec(git="https://example.org/r.git", commit=COMMIT))
    return SourceRecord(
        id="s1", title="T", owner="O", url="https://example.org", kind="docs", area="python",
        licence=Licence(id=lic, file_url="https://example.org/LICENSE", covers=covers, basis="detected"),
        version_ref="v1", fetched=None, content_hash="h", status=status, **kw)


def tree(root, files):
    for name, data in files.items():
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data if isinstance(data, bytes) else data.encode("utf-8"))
    return root


def load(tmp_path, files, rec=None, **kw):
    folder = tree(tmp_path / "snap", files)
    return load_snapshot(folder, rec or record(), rules(tmp_path), **kw)


# --- who may load ---------------------------------------------------------------------------

@pytest.mark.parametrize("status", ["approved", "limited"])
def test_approved_and_limited_load(tmp_path, status):
    r = load(tmp_path, {"a.md": LONG}, record(status))
    assert isinstance(r, LoadResult) and len(r.sources) == 1


@pytest.mark.parametrize("status", ["quarantine", "reference-only", "rejected"])
def test_other_statuses_are_refused(tmp_path, status):
    with pytest.raises(LoadRefused):
        load(tmp_path, {"a.md": LONG}, record(status))


def test_a_record_with_no_version_is_refused(tmp_path):
    with pytest.raises(LoadRefused):
        load(tmp_path, {"a.md": LONG}, replace(record(fetch=None), version_ref=""))


def test_reference_only_override_withholds_those_files_with_a_reason(tmp_path):
    rec = record(licence_overrides={"vendor": "CC-BY-NC-4.0"})
    r = load(tmp_path, {"a.md": LONG, "vendor/b.md": LONG, "vendor/deep/c.md": LONG}, rec)
    assert [s.source_id for s in r.sources] == ["s1/a.md"]
    assert sorted(r.withheld) == ["vendor/b.md", "vendor/deep/c.md"]
    assert "non-commercial" in r.withheld["vendor/b.md"]
    assert r.counts["withheld"] == 2 and r.counts["read"] == 1


def test_withheld_files_still_add_up_in_the_counts(tmp_path):
    rec = record(licence_overrides={"vendor": "CC-BY-NC-4.0"})
    r = load(tmp_path, {"a.md": LONG, "vendor/b.md": LONG, "LICENSE": "MIT", "x.py": "1", "p.png": b"x"}, rec)
    assert sum(r.counts.values()) == 5


# --- what a Source carries ------------------------------------------------------------------

def test_source_fields(tmp_path):
    r = load(tmp_path, {"docs/a.md": LONG})
    (s,) = r.sources
    assert isinstance(s, Source)
    assert s.source_id == "s1/docs/a.md" and s.version == COMMIT
    assert s.loader_version == PLAIN_VERSION == "plain-1"
    assert s.licence == "MIT" and s.licence_basis == "detected"
    assert s.text == LONG and s.raw_sha256 == hashlib.sha256(LONG.encode()).hexdigest()


def test_version_falls_back_to_version_ref_without_a_commit(tmp_path):
    (s,) = load(tmp_path, {"a.md": LONG}, record(fetch=None)).sources
    assert s.version == "v1"


def test_override_licence_id_wins_for_its_path_and_the_longest_prefix_wins(tmp_path):
    rec = record(licence_overrides={"third": "CC-BY-SA-4.0", "third/deep": "Apache-2.0"})
    r = load(tmp_path, {"a.md": LONG, "third/b.md": LONG, "third/deep/c.md": LONG}, rec)
    lic = {s.source_id: s.licence for s in r.sources}
    # An override can only lower the verdict: Apache-2.0 would not, so the record's own MIT stays.
    assert lic == {"s1/a.md": "MIT", "s1/third/b.md": "CC-BY-SA-4.0", "s1/third/deep/c.md": "MIT"}


def test_licence_for_path(tmp_path):
    rec = record(licence_overrides={"third": "CC-BY-SA-4.0"})
    assert licence_for_path(rec, rules(tmp_path), "third/x.md") == "CC-BY-SA-4.0"
    assert licence_for_path(rec, rules(tmp_path), "other/x.md") == "MIT"
    assert licence_for_path(rec, rules(tmp_path), "thirdparty/x.md") == "MIT"


def test_text_is_canonical_and_raw_hash_is_of_the_original_bytes(tmp_path):
    raw = b"\xef\xbb\xbf" + ("caf" + DECOMPOSED_E + "\r\n" + LONG).encode()
    (s,) = load(tmp_path, {"a.txt": raw}).sources
    assert normalize_text(s.text) == s.text and "\r" not in s.text and DECOMPOSED_E not in s.text
    assert s.raw_sha256 == hashlib.sha256(raw).hexdigest()


# --- skipping -------------------------------------------------------------------------------

def test_named_skips_and_the_rest_still_loads(tmp_path):
    files = {f"ok{i}.md": LONG for i in range(10)}
    files.update({"bin.md": b"ab\x00cd", "bad.md": b"\xff\xfe\xfa bad", "empty.md": b"", "big.md": b"x" * 500,
                  "n.ipynb": "{}"})
    r = load(tmp_path, files, max_file_bytes=400, max_skip_share=1.0)
    assert len(r.sources) == 10
    assert r.skipped == {"bin.md": "binary", "bad.md": "not-utf8", "empty.md": "empty", "big.md": "too-large",
                         "n.ipynb": "no-reader"}


def test_a_text_that_normalises_to_blank_is_empty(tmp_path):
    r = load(tmp_path, {"a.md": LONG, "b.md": chr(0xFEFF) + "  \n"}, max_skip_share=1.0)
    assert r.skipped == {"b.md": "empty"}


def test_a_reader_that_raises_is_unreadable(tmp_path):
    def boom(raw):
        raise RuntimeError("no")
    r = load(tmp_path, {"a.md": LONG, "b.rst": LONG}, readers={".md": ("t-1", boom), ".rst": ("t-1", lambda b: b.decode())},
             max_skip_share=1.0)
    assert r.skipped == {"a.md": "unreadable"} and [s.loader_version for s in r.sources] == ["t-1"]


def test_a_registered_reader_is_used_with_its_version(tmp_path):
    (s,) = load(tmp_path, {"a.md": LONG}, readers={".md": ("md-7", lambda raw: "CLEAN " + raw.decode())}).sources
    assert s.text.startswith("CLEAN ") and s.loader_version == "md-7"


def test_over_the_skip_share_stops_with_the_result_attached(tmp_path):
    files = {"a.md": LONG, "b.md": b"", "c.md": b"", "d.md": LONG}
    with pytest.raises(SkipShareExceeded) as exc:
        load(tmp_path, files, max_skip_share=0.25)
    assert exc.value.result.skipped == {"b.md": "empty", "c.md": "empty"}
    assert len(exc.value.result.sources) == 2


def test_exactly_at_the_limit_passes(tmp_path):
    r = load(tmp_path, {"a.md": LONG, "b.md": LONG, "c.md": LONG, "d.md": b""}, max_skip_share=0.25)
    assert len(r.sources) == 3


def test_an_empty_folder_cannot_pass(tmp_path):
    with pytest.raises(SkipShareExceeded):
        load(tmp_path, {"p.png": b"x"}, max_skip_share=1.0)


def test_code_config_and_licence_files_are_counted_not_loaded(tmp_path):
    r = load(tmp_path, {"a.md": LONG, "LICENSE": "MIT", "x.py": "1", ".git/config": "x"})
    assert r.counts == {"read": 1, "licence": 1, "ignored": 0, "code-config": 1, "skipped": 0, "withheld": 0}
    assert r.code_config == ["x.py"]


# --- determinism ----------------------------------------------------------------------------

def test_two_runs_and_a_copy_elsewhere_give_equal_sources(tmp_path):
    files = {"b.md": LONG, "a.md": LONG + "x", "sub/c.txt": LONG + "y"}
    one = load(tmp_path, files)
    two = load_snapshot(tmp_path / "snap", record(), rules(tmp_path))
    shutil.copytree(tmp_path / "snap", tmp_path / "elsewhere" / "deeper")
    three = load_snapshot(tmp_path / "elsewhere" / "deeper", record(), rules(tmp_path))
    assert one.sources == two.sources == three.sources
    assert [s.source_id for s in one.sources] == ["s1/a.md", "s1/b.md", "s1/sub/c.txt"]
