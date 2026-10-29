"""One file classifier shared by the inspection script and the snapshot loader (spec 06 rule 1)."""
import pytest

from ragcore.files import MAX_FILE_BYTES, classify_file, raw_problem


@pytest.mark.parametrize("rel, expected", [
    ("a.md", ("read", "")), ("sub/B.MDX", ("read", "")), ("n.ipynb", ("read", "")), ("x.HTM", ("read", "")),
    ("LICENSE", ("licence", "")), ("COPYING.txt", ("licence", "")), ("licence-extra.md", ("licence", "")),
    ("docs/LICENSE.md", ("read", "")),
    ("pic.PNG", ("ignored", "")), ("site.css", ("ignored", "")), ("a.lock", ("ignored", "")),
    ("x.py", ("code-config", "")), ("sub/c.YML", ("code-config", "")), ("Makefile", ("code-config", "")),
    ("sub/.gitkeep", ("code-config", "")), ("INSTALL", ("skipped", "no-reader")), ("data.jl", ("skipped", "no-reader")),
])
def test_classify_file(rel, expected):
    assert classify_file(rel) == expected


def test_symlink_is_skipped_before_anything_else():
    assert classify_file("LICENSE", symlink=True) == ("skipped", "symlink")
    assert classify_file("a.md", symlink=True) == ("skipped", "symlink")


def test_a_registered_extension_beats_code_config_and_no_reader():
    assert classify_file("x.py", loaders={".py"}) == ("read", "")
    assert classify_file("x.jl", loaders={".jl"}) == ("read", "")


def test_a_registered_extension_does_not_beat_licence_or_ignored():
    assert classify_file("LICENSE", loaders={""}) == ("licence", "")
    assert classify_file("a.png", loaders={".png"}) == ("ignored", "")


def test_raw_problem_order_and_limits():
    assert raw_problem(b"hello") == ""
    assert raw_problem(b"x" * 11, max_file_bytes=10) == "too-large"
    assert raw_problem(b"x" * 10, max_file_bytes=10) == ""
    assert raw_problem(b"  \n\t") == "empty"
    assert raw_problem(b"") == "empty"
    assert raw_problem(b"ab\x00cd") == "binary"
    assert raw_problem(b"\x00" * 11, max_file_bytes=10) == "too-large"
    assert MAX_FILE_BYTES == 2 * 1024 * 1024
