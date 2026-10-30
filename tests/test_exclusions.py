"""Spec 14: corpus exclusions. Written before the code."""
from pathlib import Path

import pytest

from ragcore.contracts import Source
from ragcore.exclusions import Rule, apply_exclusions, load_exclusions, matching_rule


def write(tmp_path, text):
    path = tmp_path / "x.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def src(source_id):
    return Source(source_id, "v1", "text here", "0" * 64, "md-1", "MIT", "declared")


GOOD = """
python-docs:
  - exclude: "Doc/library/*"
    reason: "not taught"
  - include: "Doc/library/sys.rst"
  - include: "Doc/library/json.rst"
    reason: "kept on purpose"
other:
  - exclude: "a/b.md"
    reason: "noise"
"""


def test_load(tmp_path):
    ex = load_exclusions(write(tmp_path, GOOD))
    assert list(ex) == ["python-docs", "other"]
    assert ex["python-docs"][0] == Rule("exclude", "Doc/library/*", "not taught")
    assert [r.action for r in ex["python-docs"]] == ["exclude", "include", "include"]
    assert ex["python-docs"][2].reason == "kept on purpose" and ex["python-docs"][1].reason == ""


def test_an_empty_file_is_an_empty_mapping(tmp_path):
    assert load_exclusions(write(tmp_path, "")) == {}
    assert load_exclusions(write(tmp_path, "{}\n")) == {}


@pytest.mark.parametrize("text,fragment", [
    ("- not a mapping\n", "mapping"),
    ("a: 5\n", "list"),
    ("a: []\n", "at least one rule"),
    ("a:\n  - not a mapping\n", "rule 1"),
    ("a:\n  - exclude: x\n    include: y\n    reason: r\n", "exactly one"),
    ("a:\n  - reason: r\n", "exactly one"),
    ("a:\n  - exclude: x\n", "reason"),
    ("a:\n  - exclude: x\n    reason: '  '\n", "reason"),
    ("a:\n  - exclude: ''\n    reason: r\n", "glob"),
    ("a:\n  - exclude: 5\n    reason: r\n", "glob"),
    ("a:\n  - exclude: /abs/path\n    reason: r\n", "relative"),
    ("a:\n  - exclude: ../up\n    reason: r\n", "relative"),
    ("a:\n  - exclude: x\n    reason: r\n    colour: red\n", "unknown key"),
    ("a:\n  - include: x\n    reason: 5\n", "reason"),
    ("5: \n  - exclude: x\n    reason: r\n", "source id"),
])
def test_bad_files_are_refused_with_a_message(tmp_path, text, fragment):
    with pytest.raises(ValueError, match=fragment):
        load_exclusions(write(tmp_path, text))


def test_the_message_names_the_source_and_the_rule_position(tmp_path):
    with pytest.raises(ValueError, match=r"python-docs.*rule 2"):
        load_exclusions(write(tmp_path, "python-docs:\n  - exclude: a\n    reason: r\n  - exclude: b\n"))


def test_invalid_yaml(tmp_path):
    with pytest.raises(ValueError, match="YAML"):
        load_exclusions(write(tmp_path, "a: [unclosed\n"))


def rules():
    return [Rule("exclude", "Doc/library/*", "r"), Rule("include", "Doc/library/sys.rst", "")]


def test_last_matching_rule_wins_and_default_keeps():
    r = rules()
    assert matching_rule(r, "Doc/library/os.rst").action == "exclude"
    assert matching_rule(r, "Doc/library/sys.rst").action == "include"
    assert matching_rule(r, "Doc/tutorial/x.rst") is None


def test_star_crosses_slashes_and_question_mark_is_one_character():
    assert matching_rule([Rule("exclude", "a/*", "r")], "a/b/c/d.md")
    assert matching_rule([Rule("exclude", "a/?.md", "r")], "a/b.md")
    assert not matching_rule([Rule("exclude", "a/?.md", "r")], "a/bb.md")
    assert matching_rule([Rule("exclude", "a/[bc].md", "r")], "a/c.md")
    assert not matching_rule([Rule("exclude", "a/[bc].md", "r")], "a/d.md")


def test_a_glob_must_match_the_whole_path():
    assert not matching_rule([Rule("exclude", "b.md", "r")], "a/b.md")
    assert matching_rule([Rule("exclude", "a/b.md", "r")], "a/b.md")
    assert not matching_rule([Rule("exclude", "a/b.m", "r")], "a/b.md")


def test_matching_is_case_sensitive():
    assert not matching_rule([Rule("exclude", "A/*", "r")], "a/b.md")


def test_apply_keeps_order_and_reports_reasons():
    sources = [src("python-docs/Doc/library/os.rst"), src("python-docs/Doc/library/sys.rst"),
               src("python-docs/Doc/tutorial/a.rst"), src("other/x.md")]
    out = apply_exclusions(sources, {"python-docs": rules()})
    assert [s.source_id for s in out.kept] == ["python-docs/Doc/library/sys.rst", "python-docs/Doc/tutorial/a.rst", "other/x.md"]
    assert out.excluded == [("python-docs/Doc/library/os.rst", "r")]
    assert out.unused == []


def test_apply_with_no_rules_returns_everything():
    sources = [src("a/x.md"), src("b/y.md")]
    out = apply_exclusions(sources, {})
    assert out.kept == sources and out.excluded == [] and out.unused == []


def test_a_rule_that_matches_nothing_is_reported():
    out = apply_exclusions([src("a/x.md")], {"a": [Rule("exclude", "typo/*", "r"), Rule("exclude", "x.md", "r2")],
                                              "ghost": [Rule("exclude", "*", "r")]})
    assert out.unused == [("a", "typo/*"), ("ghost", "*")]
    assert out.kept == [] and out.excluded == [("a/x.md", "r2")]


def test_an_include_that_matches_nothing_is_also_reported():
    out = apply_exclusions([src("a/x.md")], {"a": [Rule("include", "nothing.md", "")]})
    assert out.unused == [("a", "nothing.md")] and len(out.kept) == 1


def test_the_relative_path_is_after_the_first_slash_only():
    out = apply_exclusions([src("s/a/b.md")], {"s": [Rule("exclude", "a/b.md", "r")]})
    assert out.kept == [] and out.excluded == [("s/a/b.md", "r")]


def test_a_source_id_without_a_slash_is_left_alone():
    out = apply_exclusions([src("loose")], {"loose": [Rule("exclude", "*", "r")]})
    assert len(out.kept) == 1


def test_the_shipped_file_loads_and_every_exclude_has_a_reason():
    path = Path(__file__).resolve().parent.parent / "configs" / "corpus-exclusions.yaml"
    ex = load_exclusions(path)
    assert ex, "the shipped exclusions file is empty"
    for source_id, rule_list in ex.items():
        for rule in rule_list:
            assert rule.action == "include" or rule.reason.strip(), (source_id, rule.glob)
