"""Spec 05: inspection script. Written before the code."""
import json
import unicodedata
from dataclasses import replace
from datetime import date

import pytest

from ragcore.fetch import snapshot_hash
from ragcore.inspect_sources import (
    LoadError, inspect_folder, look_notebook, main, markup_counts, write_report,
)
from ragcore.register import FetchSpec, Licence, SourceRecord, save_record

DECOMPOSED_E = "e" + chr(0x301)  # e + combining acute: not in NFC form
LONG = "This is a plain sentence about retrieval and sources that makes a file long enough to count. " * 3


def tree(root, files):
    for name, data in files.items():
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data if isinstance(data, bytes) else data.encode("utf-8"))
    return root


def nb(cells):
    return json.dumps({"cells": cells, "metadata": {}, "nbformat": 4, "nbformat_minor": 5})


def md_cell(text):
    return {"cell_type": "markdown", "metadata": {}, "source": [text]}


def code_cell(src, out=""):
    outputs = [{"output_type": "stream", "name": "stdout", "text": [out]}] if out else []
    return {"cell_type": "code", "metadata": {}, "source": [src], "outputs": outputs, "execution_count": 1}


# --- buckets ---------------------------------------------------------------------

def test_every_file_lands_in_exactly_one_bucket(tmp_path):
    tree(tmp_path, {
        "a.md": LONG, "pic.png": b"\x89PNG", "LICENSE": "MIT licence\n", "c.xyz": "who knows",
        "bin.md": b"abc\x00def", "bad.md": b"\xff\xfe\xfa bad bytes", "empty.md": b"", "blank.md": "  \n\t\n",
        ".git/config": "ignored", "sub/.git/x": "ignored too",
    })
    r = inspect_folder(tmp_path)
    assert r.files_seen == 8
    assert r.buckets == {"read": 1, "licence": 1, "ignored": 1, "code-config": 0, "skipped": 5}
    assert sum(r.buckets.values()) == r.files_seen
    assert r.skipped == {"c.xyz": "no-reader", "bin.md": "binary", "bad.md": "not-utf8",
                         "empty.md": "empty", "blank.md": "empty"}


def test_too_large(tmp_path):
    tree(tmp_path, {"big.md": "x" * 500, "ok.md": LONG})
    r = inspect_folder(tmp_path, max_file_bytes=400)
    assert r.skipped == {"big.md": "too-large"}


def test_reader_that_raises_is_a_skipped_row_not_a_crash(tmp_path):
    tree(tmp_path, {"a.md": LONG, "b.md": LONG})

    def boom(raw):
        raise RuntimeError("disk on fire")
    r = inspect_folder(tmp_path, loaders={".md": boom})
    assert r.skipped == {"a.md": "unreadable", "b.md": "unreadable"}


def test_ignored_files_do_not_count_in_the_skip_share(tmp_path):
    tree(tmp_path, {"a.md": LONG, "p1.png": b"x", "p2.jpg": b"x", "f.woff2": b"x", "s.css": "a{}", "x.lock": "l"})
    r = inspect_folder(tmp_path)
    assert r.buckets["ignored"] == 5 and r.skip_share == 0.0 and r.gate_passed


def test_code_and_config_files_have_their_own_bucket_outside_the_gate(tmp_path):
    tree(tmp_path, {"a.md": LONG, "x.py": "print(1)", "c.yml": "a: 1", "d.json": "{}", "Makefile": "all:",
                    ".gitignore": "x", "sub/.gitkeep": "", "INSTALL": "unknown"})
    r = inspect_folder(tmp_path)
    assert r.buckets == {"read": 1, "licence": 0, "ignored": 0, "code-config": 6, "skipped": 1}
    assert sum(r.buckets.values()) == r.files_seen
    assert r.skipped == {"INSTALL": "no-reader"}
    assert r.skip_share == 0.5
    assert sorted(r.code_config) == [".gitignore", "Makefile", "c.yml", "d.json", "sub/.gitkeep", "x.py"]
    assert "x.py" in r.render() and "x.py" in r.to_json()["code_config"]


def test_code_config_only_folder_passes_the_gate(tmp_path):
    tree(tmp_path, {"a.md": LONG, "x.py": "1", "y.py": "2", "z.py": "3"})
    r = inspect_folder(tmp_path)
    assert r.skip_share == 0.0 and r.gate_passed


def test_a_registered_loader_beats_code_config(tmp_path):
    tree(tmp_path, {"x.py": "print(1)" + " pad" * 40})
    r = inspect_folder(tmp_path, loaders={".py": lambda raw: raw.decode()})
    assert r.buckets["read"] == 1 and r.buckets["code-config"] == 0


# --- text problems and markup ----------------------------------------------------

def test_raw_problems_counted(tmp_path):
    tree(tmp_path, {
        "bom.md": b"\xef\xbb\xbf" + LONG.encode(), "crlf.md": LONG.replace(". ", ".\r\n"),
        "nfc.md": "caf" + "" + DECOMPOSED_E + " " + LONG, "clean.md": LONG,
    })
    r = inspect_folder(tmp_path)
    assert (r.raw["bom"], r.raw["crlf"], r.raw["not_nfc"]) == (1, 1, 1)
    assert r.raw_examples["bom"] == ["bom.md"]


def test_examples_are_capped_at_five(tmp_path):
    tree(tmp_path, {f"f{i}.md": LONG.replace(". ", ".\r\n") for i in range(8)})
    r = inspect_folder(tmp_path)
    assert r.raw["crlf"] == 8 and len(r.raw_examples["crlf"]) == 5


def test_markup_counts_outside_code_fences():
    text = ("<div class='x'>hi</div>\n<Callout type=\"info\">note</Callout>\nimport Tabs from '@theme/Tabs'\n"
            ".. note::\n\n   text\n\nSee :ref:`foo` and :class:`Bar`.\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\n"
            "```html\n<span>not counted</span>\n```\n")
    c = markup_counts(text)
    assert c["html_tags"] == 2          # <div ...> and </div>; the span is inside a code fence
    assert c["jsx_tags"] == 2           # <Callout ...> and </Callout>
    assert c["import_export"] == 1
    assert c["rst_directives"] == 1
    assert c["rst_roles"] == 2
    assert c["md_tables"] == 1


def test_markup_in_report_with_bounded_examples(tmp_path):
    tree(tmp_path, {"a.md": "<p>" + "x" * 300 + "</p>\n" + LONG})
    r = inspect_folder(tmp_path)
    assert r.markup["html_tags"] == 2
    example = r.markup_examples["html_tags"][0]
    assert example.startswith("a.md") and len(example.split(": ", 1)[1]) <= 120


def test_front_matter_tiny_and_stub(tmp_path):
    toc = ".. toctree::\n   :maxdepth: 2\n\n" + "".join(f"   page{i}\n" for i in range(40)) + "\nContents\n========\n"
    tree(tmp_path, {
        "fm.md": "---\ntitle: x\ntags: [a]\n---\n" + LONG, "nofm.md": LONG, "tiny.md": "short note",
        "index.rst": toc,
    })
    r = inspect_folder(tmp_path)
    assert r.front_matter == 1 and r.tiny == 1 and r.stubs == 1
    assert "index.rst" in r.stub_examples and "tiny.md" in r.tiny_examples


def test_sections_over_and_under_the_cap(tmp_path):
    tree(tmp_path, {"a.md": "# A\n" + "x" * 7000 + "\n# B\nshort\n"})
    r = inspect_folder(tmp_path, section_cap=6000)
    assert (r.sections_total, r.sections_over, r.sections_under) == (2, 1, 1)
    assert inspect_folder(tmp_path, section_cap=8000).sections_over == 0


def test_code_fence_lines_are_not_headings(tmp_path):
    tree(tmp_path, {"a.md": "# Real\n" + LONG + "\n```python\n# not a heading\nx = 1\n```\n" + LONG})
    assert inspect_folder(tmp_path).sections_total == 1


def test_rst_headings_split_sections(tmp_path):
    tree(tmp_path, {"a.rst": "Intro\n=====\n" + LONG + "\n\nSecond\n------\n" + LONG + "\n"})
    assert inspect_folder(tmp_path).sections_total == 2


def test_example_snippet_is_the_matching_line(tmp_path):
    tree(tmp_path, {"a.md": "intro line\n<p>x</p>\nmore\n" + LONG})
    assert inspect_folder(tmp_path).markup_examples["html_tags"] == ["a.md: <p>x</p>"]


def test_licence_names_count_only_at_the_root(tmp_path):
    tree(tmp_path, {"LICENSE": "MIT\n", "docs/license.md": LONG, "vendor/LICENSE": "other\n"})
    r = inspect_folder(tmp_path)
    assert r.buckets["licence"] == 1 and r.buckets["read"] == 1
    assert r.skipped == {"vendor/LICENSE": "no-reader"}


def test_short_underline_is_not_a_heading(tmp_path):
    tree(tmp_path, {"a.md": "# H\n" + LONG + "\nA long paragraph line here\n---\nmore text\n"})
    assert inspect_folder(tmp_path).sections_total == 1


def test_symlink_is_skipped_with_a_reason(tmp_path):
    import os
    (tmp_path / "real.md").write_text(LONG, encoding="utf-8")
    try:
        os.symlink(tmp_path / "real.md", tmp_path / "link.md")
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not available here")
    assert inspect_folder(tmp_path).skipped == {"link.md": "symlink"}


# --- notebooks -------------------------------------------------------------------

def test_notebook_cells_and_outputs(tmp_path):
    raw = nb([md_cell("# Title\n" + LONG), code_cell("print('hi')", out="OUTPUT_MARKER" * 50)])
    tree(tmp_path, {"n.ipynb": raw})
    r = inspect_folder(tmp_path)
    assert r.buckets["read"] == 1
    assert (r.notebooks["files"], r.notebooks["markdown_cells"], r.notebooks["code_cells"]) == (1, 1, 1)
    assert 0 < r.notebooks["output_share"] < 1
    text = look_notebook(raw.encode())
    assert "OUTPUT_MARKER" not in text and "print('hi')" in text and "```" in text and "# Title" in text


def test_bad_notebook_json_is_skipped(tmp_path):
    tree(tmp_path, {"bad.ipynb": "{not json", "list.ipynb": "[1,2]", "ok.md": LONG})
    r = inspect_folder(tmp_path)
    assert r.skipped == {"bad.ipynb": "unreadable", "list.ipynb": "unreadable"}


# --- duplicates ------------------------------------------------------------------

def test_exact_and_near_duplicates(tmp_path):
    tree(tmp_path, {
        "a.md": "Hello, World! " + LONG, "b.md": "Hello, World! " + LONG,
        "c.md": "hello world   " + LONG.upper().lower(), "d.md": "Something else entirely. " + LONG * 2,
    })
    r = inspect_folder(tmp_path)
    assert r.exact_duplicates == [["a.md", "b.md"]]
    assert r.near_duplicates == [["a.md", "b.md", "c.md"]]
    assert all("d.md" not in g for g in r.exact_duplicates + r.near_duplicates)


def test_identical_copies_are_exact_only_not_near(tmp_path):
    tree(tmp_path, {"a.md": LONG, "b.md": LONG})
    r = inspect_folder(tmp_path)
    assert r.exact_duplicates == [["a.md", "b.md"]] and r.near_duplicates == []


def test_no_duplicates_no_groups(tmp_path):
    tree(tmp_path, {"a.md": "one " + LONG, "b.md": "two " + LONG * 2})
    r = inspect_folder(tmp_path)
    assert r.exact_duplicates == [] and r.near_duplicates == []


def test_look_text_is_idempotent_under_normalisation(tmp_path):
    tree(tmp_path, {"a.md": "caf" + "" + DECOMPOSED_E + "" + LONG, "b.md": b"\xef\xbb\xbf" + LONG.encode()})
    assert inspect_folder(tmp_path).idempotence_failures == []


# --- gate ---------------------------------------------------------------------------

def test_gate_limit(tmp_path):
    tree(tmp_path, {"a.md": LONG, "b.md": LONG + "b", "c.md": LONG + "c", "d.xyz": "x"})
    r = inspect_folder(tmp_path)
    assert r.skip_share == 0.25 and r.gate_passed and "GATE PASSED" in r.render()
    tree(tmp_path, {"e.xyz": "x"})
    r = inspect_folder(tmp_path)
    assert not r.gate_passed and "GATE FAILED" in r.render() and "0.4" in r.render()
    assert inspect_folder(tmp_path, max_skip_share=0.5).gate_passed


def test_empty_folder_fails(tmp_path):
    r = inspect_folder(tmp_path)
    assert not r.gate_passed and "no candidate files" in r.render()
    assert not inspect_folder(tmp_path, max_skip_share=1.0).gate_passed  # no limit can make nothing pass


# --- report file, determinism, read-only ---------------------------------------------

def test_report_is_deterministic_and_folder_untouched(tmp_path):
    folder = tree(tmp_path / "snap", {"a.md": LONG, "z/b.md": LONG + "!", "n.ipynb": nb([md_cell(LONG)])})
    before = snapshot_hash(folder)
    one = inspect_folder(folder, source_id="demo")
    two = inspect_folder(folder, source_id="demo")
    assert one.render() == two.render()
    assert json.dumps(one.to_json(), sort_keys=True) == json.dumps(two.to_json(), sort_keys=True)
    assert snapshot_hash(folder) == before
    assert before in one.render() and "normaliser" in one.render().lower()


def test_write_report_names_carry_the_hash_prefix(tmp_path):
    folder = tree(tmp_path / "snap", {"a.md": LONG})
    r = inspect_folder(folder, source_id="demo")
    txt, js = write_report(r, tmp_path / "out")
    prefix = snapshot_hash(folder).split(":")[1][:12]
    assert txt.name == f"demo-{prefix}.txt" and js.name == f"demo-{prefix}.json"
    assert txt.read_text(encoding="utf-8") == r.render()
    assert json.loads(js.read_text(encoding="utf-8"))["source_id"] == "demo"


# --- loader mode ----------------------------------------------------------------------

def strip_tags(raw):
    import re
    return re.sub(r"<[^>]+>", "", raw.decode("utf-8"))


HTMLISH = "<p>" + LONG + "</p>\n<div>" + LONG + "</div>\n"


def run_with(tmp_path, loader, name="a.md", text=HTMLISH):
    tree(tmp_path, {name: text})
    return inspect_folder(tmp_path, loaders={"." + name.split(".")[-1]: loader})


def checks(report):
    return {check for _path, check in report.loader["failures"]}


def test_good_loader_passes_and_improves(tmp_path):
    r = run_with(tmp_path, strip_tags)
    assert r.loader["failures"] == [] and r.loader["regressions"] == []
    assert r.loader["used"][".md"] == "loader"
    assert r.compare[".md"]["markup_after"] < r.compare[".md"]["markup_before"]
    assert r.gate_passed


def test_loader_returning_empty_text(tmp_path):
    r = run_with(tmp_path, lambda raw: "")
    assert "empty-output" in checks(r) and not r.gate_passed


def test_loader_output_much_shorter(tmp_path):
    r = run_with(tmp_path, lambda raw: raw.decode()[:20])
    assert "too-short" in checks(r)


def test_loader_output_not_normalised(tmp_path):
    r = run_with(tmp_path, lambda raw: ("caf" + "" + DECOMPOSED_E + " ") * 60 + LONG)
    assert "not-normalised" in checks(r)


def test_loader_adds_markup(tmp_path):
    r = run_with(tmp_path, lambda raw: raw.decode() + "<b>x</b>" * 10)
    assert "more-markup" in checks(r)


def test_loader_not_deterministic(tmp_path):
    calls = []

    def flaky(raw):
        calls.append(1)
        return strip_tags(raw) + (" extra" * len(calls))
    assert "non-deterministic" in checks(run_with(tmp_path, flaky))


def test_loader_that_skips_a_readable_file_is_a_regression(tmp_path):
    def refuse(raw):
        raise LoadError("weird-format")
    r = run_with(tmp_path, refuse)
    assert r.skipped == {"a.md": "weird-format"}
    assert r.loader["regressions"] == [("a.md", "skipped-by-loader: weird-format")]
    assert not r.gate_passed


def test_loader_that_raises_something_else_is_unreadable_and_a_regression(tmp_path):
    def boom(raw):
        raise ValueError("oops")
    r = run_with(tmp_path, boom)
    assert r.skipped == {"a.md": "unreadable"} and r.loader["regressions"]


def test_declared_stub_skip_is_not_a_regression(tmp_path):
    def stub(raw):
        raise LoadError("stub")
    r = run_with(tmp_path, stub)
    assert r.skipped == {"a.md": "stub"} and r.loader["regressions"] == []


def test_extensions_without_a_loader_fall_back_to_the_look_reader(tmp_path):
    tree(tmp_path, {"a.md": HTMLISH, "b.rst": "Title\n=====\n" + LONG})
    r = inspect_folder(tmp_path, loaders={".md": strip_tags})
    assert r.loader["used"] == {".md": "loader", ".rst": "look"}
    assert "look reader" in r.render().lower()


def test_compare_flag_prints_before_and_after(tmp_path):
    tree(tmp_path, {"a.md": HTMLISH})
    r = inspect_folder(tmp_path, loaders={".md": strip_tags}, compare=True)
    assert "before" in r.render() and "after" in r.render()
    assert "before" not in inspect_folder(tmp_path, loaders={".md": strip_tags}).render()


# --- script entry point ------------------------------------------------------------------

def test_main_exit_codes(tmp_path, capsys):
    good = tree(tmp_path / "good", {"a.md": LONG})
    bad = tree(tmp_path / "bad", {"a.md": LONG, "b.xyz": "x"})
    out = str(tmp_path / "out")
    assert main([str(good), "--out", out]) == 0
    assert "GATE PASSED" in capsys.readouterr().out
    assert main([str(bad), "--out", out]) == 1
    assert main([str(bad), "--out", out, "--max-skip-share", "0.9"]) == 0
    assert main([str(tmp_path / "missing"), "--out", out]) == 2
    assert main([]) == 2
    assert list((tmp_path / "out").glob("*.txt"))


def test_main_finds_the_folder_from_a_record(tmp_path):
    sha = "a" * 40
    reg, snaps = tmp_path / "reg", tmp_path / "snaps"
    reg.mkdir()
    tree(snaps / "demo" / sha, {"a.md": LONG})
    rec = SourceRecord(
        id="demo", title="T", owner="O", url="https://example.org", kind="docs", area="python",
        licence=Licence(id="MIT", file_url="https://x/L", covers="both"), version_ref=sha, fetched=date(2026, 10, 29),
        content_hash="h", status="approved", fetch=FetchSpec(git="https://github.com/a/b"))
    save_record(rec, reg / "demo.yaml")
    assert main(["--source", "demo", "--register", str(reg), "--snapshots", str(snaps), "--out", str(tmp_path / "o")]) == 0
    assert main(["--source", "nope", "--register", str(reg), "--snapshots", str(snaps), "--out", str(tmp_path / "o")]) == 2
    assert [p.name for p in (tmp_path / "o").glob("demo-*.txt")]
