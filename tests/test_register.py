"""Spec 02: the source register and status transitions. Written before the code."""
from dataclasses import replace
from datetime import date

import pytest

from ragcore.gate import LicenceRules
from ragcore.register import (
    FetchSpec, Licence, RegisterError, SourceRecord, TransitionError,
    load_record, load_register, save_record, set_status, to_contract_fields,
)

GOOD = """\
id: demo
title: Demo docs
owner: Someone
url: https://example.org/demo
kind: docs
area: python
licence:
  id: MIT
  file_url: https://example.org/LICENSE
  covers: both
  file_hash: lh
version:
  ref: {sha}
  fetched: 2026-10-20
content_hash: ch
status: quarantine
fetch:
  git: https://github.com/example/demo
  commit: {sha}
  paths: [docs]
"""
SHA = "a" * 40
ON = date(2026, 10, 29)


@pytest.fixture
def rules(tmp_path):
    p = tmp_path / "licences.yaml"
    p.write_text("allowed: [MIT]\nshare-alike: [CC-BY-SA-4.0]\nnon-commercial: [CC-BY-NC-4.0]\n", encoding="utf-8")
    return LicenceRules.load(p)


def write(tmp_path, text=None, name="demo.yaml", **replace_):
    text = (text or GOOD).format(sha=SHA)
    for old, new in replace_.items():
        text = text.replace(old, new)
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def test_load_valid_record(tmp_path):
    rec = load_record(write(tmp_path))
    assert isinstance(rec, SourceRecord)
    assert rec.id == "demo" and rec.status == "quarantine"
    assert rec.licence.covers == "both" and rec.licence.basis == "declared"
    assert rec.fetch.commit == SHA and rec.fetch.paths == ("docs",)
    assert rec.licence_overrides == {} and rec.max_mb == 200


@pytest.mark.parametrize("field", ["id", "title", "owner", "url", "kind", "area", "status"])
def test_missing_field_names_file_and_field(tmp_path, field):
    text = "\n".join(l for l in GOOD.format(sha=SHA).splitlines() if not l.startswith(field + ":"))
    p = tmp_path / "bad.yaml"
    p.write_text(text, encoding="utf-8")
    with pytest.raises(RegisterError, match=rf"bad\.yaml.*{field}"):
        load_record(p)


@pytest.mark.parametrize("old,new,field", [
    ("kind: docs", "kind: movie", "kind"),
    ("area: python", "area: cooking", "area"),
    ("covers: both", "covers: some", "covers"),
    ("status: quarantine", "status: maybe", "status"),
])
def test_bad_choice(tmp_path, old, new, field):
    with pytest.raises(RegisterError, match=field):
        load_record(write(tmp_path, **{old: new}))


def test_bad_yaml_and_non_mapping(tmp_path):
    p = tmp_path / "x.yaml"
    p.write_text("id: [oops\n", encoding="utf-8")
    with pytest.raises(RegisterError, match="x.yaml"):
        load_record(p)
    p.write_text("- a\n", encoding="utf-8")
    with pytest.raises(RegisterError, match="mapping"):
        load_record(p)


def test_duplicate_id(tmp_path):
    write(tmp_path, name="a.yaml")
    write(tmp_path, name="b.yaml")
    with pytest.raises(RegisterError, match="duplicate.*demo"):
        load_register(tmp_path)


def test_register_sorted_by_id(tmp_path):
    write(tmp_path, name="a.yaml", **{"id: demo": "id: zed"})
    write(tmp_path, name="b.yaml", **{"id: demo": "id: abc"})
    assert [r.id for r in load_register(tmp_path)] == ["abc", "zed"]


@pytest.mark.parametrize("bad", ["0", "-3", "true", "'big'", "1.5"])
def test_bad_max_mb(tmp_path, bad):
    with pytest.raises(RegisterError, match="max_mb"):
        load_record(write(tmp_path, GOOD + f"max_mb: {bad}\n"))


@pytest.mark.parametrize("files", ["[../x]", "[a/../x]", "['*.md']", "[a//b]", "[dir/]", "[/abs]"])
def test_bad_fetch_files(tmp_path, files):
    with pytest.raises(RegisterError, match="fetch.files"):
        load_record(write(tmp_path, GOOD + f"  files: {files}\n"))


def test_fetch_needs_exactly_one_of_git_or_web(tmp_path):
    both = GOOD.replace("  paths: [docs]\n", "  web: https://arxiv.org/html/1\n  licence_url: https://x.org/l\n")
    with pytest.raises(RegisterError, match="git.*web"):
        load_record(write(tmp_path, both))
    web_no_licence = GOOD.replace("  git: https://github.com/example/demo\n  commit: {sha}\n  paths: [docs]\n",
                                  "  web: https://arxiv.org/html/1\n")
    with pytest.raises(RegisterError, match="licence_url"):
        load_record(write(tmp_path, web_no_licence))


@pytest.mark.parametrize("commit", ["abc123", "main", "A" * 40, "a" * 39])
def test_commit_must_be_full_lowercase_sha(tmp_path, commit):
    with pytest.raises(RegisterError, match="commit"):
        load_record(write(tmp_path, GOOD.replace("commit: {sha}", f"commit: {commit}")))


def test_basis_choice_and_unknown_pairing(tmp_path):
    with pytest.raises(RegisterError, match="basis"):
        load_record(write(tmp_path, GOOD.replace("  covers: both\n", "  covers: both\n  basis: guessed\n")))
    # basis unknown only with licence unknown
    with pytest.raises(RegisterError, match="unknown"):
        load_record(write(tmp_path, GOOD.replace("  covers: both\n", "  covers: both\n  basis: unknown\n")))
    ok = GOOD.replace("id: MIT", "id: unknown").replace("  covers: both\n", "  covers: both\n  basis: unknown\n")
    assert load_record(write(tmp_path, ok)).licence.basis == "unknown"


def test_licence_overrides_loaded_and_checked(tmp_path):
    rec = load_record(write(tmp_path, GOOD + "licence_overrides:\n  docs/vendor: CC-BY-NC-4.0\n"))
    assert rec.licence_overrides == {"docs/vendor": "CC-BY-NC-4.0"}
    with pytest.raises(RegisterError, match="licence_overrides"):
        load_record(write(tmp_path, GOOD + "licence_overrides: [a]\n"))
    with pytest.raises(RegisterError, match="licence_overrides"):
        load_record(write(tmp_path, GOOD + "licence_overrides:\n  ../x: MIT\n"))


def test_save_then_load_roundtrip(tmp_path):
    rec = load_record(write(tmp_path, GOOD + "licence_overrides:\n  docs/v: MIT\nnotes: hello\n"))
    out = tmp_path / "out.yaml"
    save_record(rec, out)
    assert load_record(out) == rec


def test_roundtrip_web_record(tmp_path):
    text = GOOD.replace("  git: https://github.com/example/demo\n  commit: {sha}\n  paths: [docs]\n",
                        "  web: https://arxiv.org/html/1\n  licence_url: https://x.org/l\n")
    rec = load_record(write(tmp_path, text))
    out = tmp_path / "w.yaml"
    save_record(rec, out)
    assert load_record(out) == rec


# --- transitions (spec 02, rules 1 to 5) ---

@pytest.fixture
def rec(tmp_path):
    return load_record(write(tmp_path))


def test_rule1_rejected_is_final(rec, rules):
    rejected = set_status(rec, "rejected", rules, reviewer="Me", on=ON)
    with pytest.raises(TransitionError, match="final"):
        set_status(rejected, "approved", rules, reviewer="Me", on=ON)


def test_rule2_reviewer_required_and_stored(rec, rules):
    for blank in ("", "   "):
        with pytest.raises(TransitionError, match="reviewer"):
            set_status(rec, "rejected", rules, reviewer=blank, on=ON)
    out = set_status(rec, "approved", rules, reviewer=" Asheesh ", on=ON)
    assert out.reviewer == "Asheesh" and out.reviewed_on == ON and out.status == "approved"


def test_unknown_status_refused(rec, rules):
    with pytest.raises(TransitionError, match="Unknown"):
        set_status(rec, "maybe", rules, reviewer="Me", on=ON)


@pytest.mark.parametrize("field,text", [
    ("file_url", "licence file URL"), ("version_ref", "pinned version"),
    ("file_hash", "licence file hash"), ("content_hash", "content hash"),
])
@pytest.mark.parametrize("target", ["approved", "limited"])
def test_rule3_evidence_needed(rec, rules, field, text, target):
    broken = (replace(rec, licence=replace(rec.licence, **{field: ""})) if field in ("file_url", "file_hash")
              else replace(rec, **{field: ""}))
    with pytest.raises(TransitionError, match=text):
        set_status(broken, target, rules, reviewer="Me", on=ON)


def test_rule3_not_needed_for_reference_only_or_rejected(rec, rules):
    bare = replace(rec, content_hash="", version_ref="")
    assert set_status(bare, "reference-only", rules, reviewer="Me", on=ON).status == "reference-only"
    assert set_status(bare, "rejected", rules, reviewer="Me", on=ON).status == "rejected"


def test_rule4_ceiling_refused_without_reason(rec, rules):
    nc = replace(rec, licence=replace(rec.licence, id="CC-BY-NC-4.0"))
    with pytest.raises(TransitionError, match="override"):
        set_status(nc, "approved", rules, reviewer="Me", on=ON)
    with pytest.raises(TransitionError, match="override"):
        set_status(nc, "approved", rules, reviewer="Me", on=ON, override_reason="   ")


def test_rule4_override_with_reason_is_stored(rec, rules):
    nc = replace(rec, licence=replace(rec.licence, id="CC-BY-NC-4.0"))
    out = set_status(nc, "limited", rules, reviewer="Me", on=ON, override_reason=" owner permission email ")
    assert out.status == "limited" and out.override_reason == "owner permission email"


def test_rule4_within_ceiling_needs_no_reason(rec, rules):
    code_only = replace(rec, licence=replace(rec.licence, covers="code"))
    assert set_status(code_only, "limited", rules, reviewer="Me", on=ON).status == "limited"
    with pytest.raises(TransitionError):  # approved is above the "limited" ceiling
        set_status(code_only, "approved", rules, reviewer="Me", on=ON)


def test_set_status_does_not_mutate(rec, rules):
    set_status(rec, "approved", rules, reviewer="Me", on=ON)
    assert rec.status == "quarantine" and rec.reviewer == ""


def test_contract_fields(rec):
    assert to_contract_fields(rec) == {"licence": "MIT", "licence_basis": "declared"}
    unk = replace(rec, licence=replace(rec.licence, id="unknown", basis="unknown"))
    assert to_contract_fields(unk) == {"licence": "unknown", "licence_basis": "unknown"}
