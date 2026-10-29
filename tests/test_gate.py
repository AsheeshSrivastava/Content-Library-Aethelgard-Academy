"""Spec 02: the licence gate. Written before the code."""
from dataclasses import replace

import pytest

from ragcore.gate import RANK, LicenceRules, Verdict, check, check_path
from ragcore.register import FetchSpec, Licence, RegisterError, SourceRecord


def rules(tmp_path, text="allowed: [MIT, Apache-2.0]\nshare-alike: [CC-BY-SA-4.0]\nnon-commercial: [CC-BY-NC-4.0]\n"):
    path = tmp_path / "licences.yaml"
    path.write_text(text, encoding="utf-8")
    return LicenceRules.load(path)


def record(licence_id="MIT", covers="both", **kw):
    return SourceRecord(
        id="s1", title="T", owner="O", url="https://example.org", kind="docs", area="python",
        licence=Licence(id=licence_id, file_url="https://example.org/LICENSE", covers=covers),
        version_ref="a" * 40, fetched=None, content_hash="h", status="quarantine", **kw)


def test_rank_order():
    assert RANK["reference-only"] < RANK["limited"] < RANK["approved"]


@pytest.mark.parametrize("lic,covers,status,share", [
    ("MIT", "both", "approved", False),
    ("MIT", "code", "limited", False),
    ("MIT", "text", "limited", False),
    ("MIT", "unknown", "reference-only", False),
    ("CC-BY-SA-4.0", "both", "limited", True),
    ("CC-BY-SA-4.0", "unknown", "reference-only", False),
    ("CC-BY-NC-4.0", "both", "reference-only", False),
    ("Weird-1.0", "both", "reference-only", False),
    ("unknown", "both", "reference-only", False),
])
def test_matrix(tmp_path, lic, covers, status, share):
    verdict = check(record(lic, covers), rules(tmp_path))
    assert isinstance(verdict, Verdict)
    assert verdict.status == status
    assert verdict.share_alike is share
    assert verdict.reasons  # always says why


def test_gate_never_returns_rejected(tmp_path):
    for lic in ("MIT", "CC-BY-NC-4.0", "Weird-1.0"):
        assert check(record(lic), rules(tmp_path)).status != "rejected"


def test_id_in_two_classes_is_an_error(tmp_path):
    with pytest.raises(RegisterError, match="MIT"):
        rules(tmp_path, "allowed: [MIT]\nnon-commercial: [MIT]\n")


def test_unknown_class_is_an_error(tmp_path):
    with pytest.raises(RegisterError, match="extra"):
        rules(tmp_path, "extra: [MIT]\n")


def test_bad_yaml_and_non_mapping(tmp_path):
    with pytest.raises(RegisterError):
        rules(tmp_path, "allowed: [MIT\n")
    with pytest.raises(RegisterError):
        rules(tmp_path, "- MIT\n")


def test_empty_rules_make_everything_unknown(tmp_path):
    assert check(record("MIT"), rules(tmp_path, "")).status == "reference-only"


def test_licence_overrides_longest_prefix_wins(tmp_path):
    r = rules(tmp_path)
    rec = record("MIT", licence_overrides={"docs/": "CC-BY-SA-4.0", "docs/vendor/": "MIT"})
    assert check_path(rec, r, "docs/vendor/x.md").status == "approved"
    assert check_path(rec, r, "docs/guide.md").status == "limited"
    assert check_path(rec, r, "src/a.py").status == "approved"  # no prefix matches: record's own


def test_override_cannot_raise_the_ceiling(tmp_path):
    r = rules(tmp_path)
    rec = record("CC-BY-NC-4.0", licence_overrides={"docs/": "MIT"})
    assert check_path(rec, r, "docs/a.md").status == "reference-only"
    rec = record("MIT", "code", licence_overrides={"docs/": "MIT"})
    assert check_path(rec, r, "docs/a.md").status == "limited"


def test_override_can_lower_the_ceiling(tmp_path):
    rec = record("MIT", licence_overrides={"third_party/": "CC-BY-NC-4.0"})
    assert check_path(rec, rules(tmp_path), "third_party/x.py").status == "reference-only"


def test_prefix_matches_whole_path_parts_only(tmp_path):
    rec = record("MIT", licence_overrides={"docs": "CC-BY-NC-4.0"})
    r = rules(tmp_path)
    assert check_path(rec, r, "docs/a.md").status == "reference-only"
    assert check_path(rec, r, "docs-extra/a.md").status == "approved"
