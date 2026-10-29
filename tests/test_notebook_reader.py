"""Spec 11: Jupyter notebook reader. Written before the code."""
import json

import pytest

from ragcore.errors import LoadError
from ragcore.loaders.notebook import NOTEBOOK_VERSION, read_ipynb
from ragcore.loaders.registry import DEFAULT_READERS

TICKS = "`" * 3


def nb(cells, *, language="python", nbformat=4, **extra):
    meta = {"language_info": {"name": language}} if language else {}
    return {"nbformat": nbformat, "nbformat_minor": 5, "metadata": meta, "cells": cells, **extra}


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def code(src, outputs=()):
    return {"cell_type": "code", "metadata": {}, "source": src, "execution_count": 1, "outputs": list(outputs)}


def read(notebook) -> str:
    return read_ipynb(json.dumps(notebook).encode("utf-8"))


def test_version_and_registration():
    assert NOTEBOOK_VERSION == "ipynb-1"
    assert DEFAULT_READERS[".ipynb"][0] == NOTEBOOK_VERSION


def test_markdown_and_code_cells_in_order():
    out = read(nb([md("# Title\n\nSome words."), code("x = 1\nprint(x)"), md("More words.")]))
    assert out == f"# Title\n\nSome words.\n\n{TICKS}python\nx = 1\nprint(x)\n{TICKS}\n\nMore words.\n"


def test_list_and_string_source_are_the_same():
    a = read(nb([md(["# T\n", "\n", "words\n"]), code(["a = 1\n", "b = 2"])]))
    b = read(nb([md("# T\n\nwords\n"), code("a = 1\nb = 2")]))
    assert a == b


def test_html_and_comments_leave_markdown_cells():
    out = read(nb([md("Hello <!-- hidden --> world <div>kept</div>")]))
    assert "hidden" not in out and "<div>" not in out and "kept" in out


def test_magics_and_shell_lines_stay_in_code():
    out = read(nb([code("!pip install -U sentence-transformers\n%matplotlib inline\nimport os")]))
    assert "!pip install -U sentence-transformers\n%matplotlib inline\nimport os" in out


def test_outputs_raw_cells_and_metadata_are_dropped():
    cells = [
        md("Words."),
        {"cell_type": "raw", "metadata": {}, "source": "raw secret"},
        code("print(1)", outputs=[{"output_type": "stream", "name": "stdout", "text": ["OUTPUT TEXT\n"]},
                                  {"output_type": "display_data", "data": {"image/png": "iVBORw0KGgo="}}]),
    ]
    notebook = nb(cells, extra_key="EXTRA")
    notebook["metadata"]["colab"] = {"name": "COLAB NAME"}
    out = read(notebook)
    for dropped in ("OUTPUT TEXT", "iVBOR", "raw secret", "COLAB NAME", "EXTRA"):
        assert dropped not in out


def test_the_same_notebook_with_and_without_outputs_reads_the_same():
    plain = nb([md("Words."), code("print(1)")])
    noisy = nb([md("Words."), code("print(1)", outputs=[{"output_type": "stream", "name": "stdout", "text": "1\n"}])])
    assert read(plain) == read(noisy)


def test_the_fence_grows_when_the_code_holds_backticks():
    out = read(nb([code(f"s = '{TICKS}'")]))
    assert out.startswith("````python\n") and out.rstrip().endswith("````")


def test_language_falls_back_to_the_kernel_and_odd_names_are_left_off():
    n = nb([code("a")], language=None)
    n["metadata"]["kernelspec"] = {"language": "julia"}
    assert read(n).startswith(f"{TICKS}julia\n")
    assert read(nb([code("a")], language="my lang!")).startswith(f"{TICKS}\n")
    assert read(nb([code("a")], language=None)).startswith(f"{TICKS}\n")


def test_empty_cells_are_skipped():
    assert read(nb([md("  \n"), code(""), code("  \n"), md("Words.")])) == "Words.\n"


def test_nothing_left_is_empty():
    assert read(nb([])) == ""
    assert read(nb([code("")])) == ""


def test_bad_input_is_named():
    with pytest.raises(LoadError, match="bad-notebook"):
        read_ipynb(b"not json")
    with pytest.raises(LoadError, match="bad-notebook"):
        read_ipynb(b'{"nbformat": 4}')
    with pytest.raises(LoadError, match="bad-notebook"):
        read_ipynb(b"[1, 2]")
    with pytest.raises(LoadError, match="old-notebook"):
        read(nb([md("x")], nbformat=3))


def test_a_markdown_fence_in_a_markdown_cell_is_untouched():
    text = f"Example:\n\n{TICKS}html\n<div>keep me</div>\n{TICKS}"
    assert f"<div>keep me</div>" in read(nb([md(text)]))


def test_unknown_cell_types_are_ignored():
    assert read(nb([{"cell_type": "mystery", "source": "x"}, md("Words.")])) == "Words.\n"


def test_deterministic_and_clean_ending():
    notebook = nb([md("# T"), code("x = 1"), md("end")])
    out = read(notebook)
    assert read(notebook) == out and out.endswith("\n") and not out.endswith("\n\n") and "\n\n\n" not in out


# --- found by the mutation check ------------------------------------------------------------

def test_the_notebook_language_wins_over_the_kernel_name():
    n = nb([code("a")], language="python")
    n["metadata"]["kernelspec"] = {"language": "julia"}
    assert read(n).startswith(f"{TICKS}python\n")


def test_cells_that_are_not_objects_are_ignored():
    assert read(nb([1, "x", None, md("Words.")])) == "Words.\n"


def test_blank_lines_around_code_are_trimmed_but_inner_ones_stay():
    assert read(nb([code("\n\na = 1\n\nb = 2\n\n")])) == f"{TICKS}python\na = 1\n\nb = 2\n{TICKS}\n"


def test_code_text_is_canonical():
    out = read(nb([code("a = 1\r\ncaf" + chr(0x65) + chr(0x301))]))
    assert "\r" not in out and "caf" + chr(0xE9) in out


def test_a_bom_in_the_file_is_fine():
    assert read_ipynb(b"\xef\xbb\xbf" + json.dumps(nb([md("Words.")])).encode()) == "Words.\n"
