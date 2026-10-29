"""Spec 08: the reStructuredText reader. Written before the code."""
import random
import re

import pytest

from ragcore.contracts import normalize_text
from ragcore.loaders.rst import read_rst

TICKS = "`" * 3


def rst(text):
    return read_rst(text.encode("utf-8"))


def words(text):
    return [w.lower() for w in re.findall(r"\w+", text)]


def is_subsequence(small, big):
    it = iter(big)
    return all(w in it for w in small)


def test_utf8_with_bom_is_fine_and_bad_bytes_raise():
    assert "Hello" in read_rst(b"\xef\xbb\xbfHello world")
    with pytest.raises(UnicodeDecodeError):
        read_rst(b"\xff\xfe\xfa bad")


# --- headings -------------------------------------------------------------------------------

def test_heading_levels_follow_the_order_styles_first_appear():
    text = "Title\n=====\n\nIntro text.\n\nSection\n-------\n\nBody.\n\nSub\n^^^\n\nDeep.\n\nSection two\n-----------\n\nMore.\n"
    assert rst(text) == "# Title\n\nIntro text.\n\n## Section\n\nBody.\n\n### Sub\n\nDeep.\n\n## Section two\n\nMore.\n"


def test_overline_and_underline_is_its_own_style():
    text = "=====\nTitle\n=====\n\nText.\n\nSub\n---\n\nOther\n=====\n\nEnd.\n"
    assert rst(text) == "# Title\n\nText.\n\n## Sub\n\n### Other\n\nEnd.\n"


def test_an_underline_shorter_than_the_text_is_not_a_heading():
    text = "A long title line\n==\n\nText.\n"
    assert rst(text) == text


def test_an_indented_line_is_not_a_heading():
    text = "Intro.\n\n   indented line\n--------------\n\nEnd.\n"
    assert "#" not in rst(text)


def test_an_underline_that_is_a_different_style_from_the_overline_does_not_pair_up():
    text = "=====\nTitle\n-----\n\nText.\n\nOther\n-----\n\nEnd.\n"
    assert rst(text) == "# Title\n\nText.\n\n# Other\n\nEnd.\n"


def test_an_overline_shorter_than_the_text_is_not_a_heading():
    assert rst("====\nLonger title here\n====\n\nText.\n") == "Longer title here\n====\n\nText.\n"


def test_a_short_rule_after_text_is_kept_as_text():
    assert rst("A long paragraph line\n-----\n\nEnd.\n") == "A long paragraph line\n-----\n\nEnd.\n"


def test_a_transition_is_removed():
    assert rst("First.\n\n----\n\nSecond.\n") == "First.\n\nSecond.\n"


def test_table_borders_are_left_alone():
    table = "=====  =====\nA      B\n=====  =====\n1      2\n=====  =====\n"
    assert rst(table) == table


def test_a_text_line_inside_a_paragraph_is_not_a_heading():
    text = "First line of a paragraph\nSecond line of the paragraph\n------------------------------\n\nEnd.\n"
    assert rst(text) == text


# --- literal blocks and code ----------------------------------------------------------------

def test_literal_block_after_double_colon():
    assert rst("Example::\n\n    x = 1\n\n    y = 2\n\nAfter.\n") == \
        f"Example:\n\n{TICKS}\nx = 1\n\ny = 2\n{TICKS}\n\nAfter.\n"


def test_double_colon_alone_or_after_a_space_disappears():
    assert rst("::\n\n    code\n") == f"{TICKS}\ncode\n{TICKS}\n"
    assert rst("Text ::\n\n    x\n") == f"Text\n\n{TICKS}\nx\n{TICKS}\n"


def test_double_colon_without_a_block_just_loses_a_colon():
    assert rst("Example::\n\nNot indented.\n") == "Example:\n\nNot indented.\n"


def test_code_block_directive_with_language_and_options():
    text = ".. code-block:: python\n   :linenos:\n\n   import os\n\n   print(1)\n\nAfter.\n"
    assert rst(text) == f"{TICKS}python\nimport os\n\nprint(1)\n{TICKS}\n\nAfter.\n"


@pytest.mark.parametrize("name", ["doctest", "testcode", "testsetup", "testoutput", "testcleanup", "productionlist",
                                  "grammar-snippet", "parsed-literal"])
def test_other_code_directives_become_plain_fences(name):
    assert rst(f".. {name}::\n\n   >>> 1 + 1\n   2\n") == f"{TICKS}\n>>> 1 + 1\n2\n{TICKS}\n"


def test_a_directive_argument_that_is_not_a_language_is_not_one():
    assert rst(".. doctest:: mygroup\n\n   >>> 1\n") == f"{TICKS}\n>>> 1\n{TICKS}\n"


def test_code_is_never_cleaned():
    text = ".. code-block:: rst\n\n   :func:`x` and ``y`` and .. note::\n   Title\n   =====\n"
    assert rst(text) == f"{TICKS}rst\n:func:`x` and ``y`` and .. note::\nTitle\n=====\n{TICKS}\n"


def test_an_existing_fence_is_left_alone():
    text = f"Text.\n\n{TICKS}\n:func:`x`::\n.. note::\n{TICKS}\n"
    assert rst(text) == text


# --- directives -----------------------------------------------------------------------------

@pytest.mark.parametrize("directive", [
    ".. toctree::\n   :maxdepth: 2\n\n   intro\n   usage\n",
    ".. index:: single: foo; bar\n",
    ".. module:: re\n   :synopsis: Regular expressions.\n",
    ".. currentmodule:: re\n",
    ".. sectionauthor:: Someone <a@b.c>\n",
    ".. include:: ../README.rst\n",
    ".. literalinclude:: example.py\n   :language: python\n",
    ".. image:: pic.png\n   :width: 10\n",
    ".. availability:: Unix, not Emscripten.\n",
    ".. audit-event:: open path\n",
    ".. highlight:: python\n",
    ".. only:: html\n\n   Hidden web text.\n",
    ".. raw:: html\n\n   <b>x</b>\n",
    ".. _my-label:\n",
    ".. |version| replace:: 3.12\n",
    ".. a plain comment\n   that continues here\n",
])
def test_dropped_directives_take_their_body_with_them(directive):
    assert rst(f"Before.\n\n{directive}\nAfter.\n") == "Before.\n\nAfter.\n"


def test_version_notes():
    assert rst(".. versionchanged:: 3.9\n   Added the *flags* parameter.\n\nText.\n") == \
        "3.9: Added the *flags* parameter.\n\nText.\n"
    assert rst("Text.\n\n.. versionadded:: 3.8\n\nMore.\n") == "Text.\n\nMore.\n"
    assert rst(".. deprecated-removed:: 3.1 3.5\n   Use something else.\n") == "3.1 3.5: Use something else.\n"


def test_object_descriptions_keep_signature_and_body():
    text = ".. function:: re.compile(pattern, flags=0)\n   :module: re\n\n   Compile a pattern.\n\n   Second paragraph.\n\nAfter.\n"
    assert rst(text) == "re.compile(pattern, flags=0)\n\nCompile a pattern.\n\nSecond paragraph.\n\nAfter.\n"
    assert rst(".. py:method:: Foo.bar()\n\n   Does bar.\n") == "Foo.bar()\n\nDoes bar.\n"
    assert rst(".. data:: False\n\n   The false value.\n") == "False\n\nThe false value.\n"


def test_admonitions_keep_their_body_and_title():
    assert rst(".. note::\n\n   Be careful.\n") == "Be careful.\n"
    assert rst(".. admonition:: Title here\n\n   Body text.\n") == "Title here\n\nBody text.\n"
    assert rst(".. seealso::\n\n   Module :mod:`os`.\n") == "Module `os`.\n"


def test_figure_keeps_its_caption_but_not_the_path():
    assert rst(".. figure:: pic.png\n   :width: 10\n\n   Caption text.\n") == "Caption text.\n"


def test_domain_prefixed_directives_use_the_name_after_the_prefix():
    assert rst("Before.\n\n.. py:module:: foo\n\nAfter.\n") == "Before.\n\nAfter.\n"


def test_unknown_directive_keeps_argument_and_body():
    assert rst(".. frobnicate:: widget\n   :flag:\n\n   Inner words.\n") == "widget\n\nInner words.\n"


def test_nested_directives_and_literal_blocks_inside_a_body():
    text = ".. note::\n\n   Run it::\n\n      go()\n\n   .. versionadded:: 3.2\n\n   Done.\n"
    assert rst(text) == f"Run it:\n\n{TICKS}\ngo()\n{TICKS}\n\nDone.\n"


def test_the_body_ends_at_the_first_less_indented_line():
    text = ".. note::\n\n   Inside.\n\nOutside.\n"
    assert rst(text) == "Inside.\n\nOutside.\n"


# --- inline ---------------------------------------------------------------------------------

def test_inline_literals_become_single_backticks():
    assert rst("Use ``torch.nn`` here.\n") == "Use `torch.nn` here.\n"


@pytest.mark.parametrize("src, out", [
    (":func:`re.compile`", "`re.compile`"),
    (":class:`~a.b.Thing`", "`Thing`"),
    (":meth:`.Foo.bar`", "`Foo.bar`"),
    (":exc:`!KeyError`", "`KeyError`"),
    (":py:class:`Thing`", "`Thing`"),
    (":c:func:`PyObject_New`", "`PyObject_New`"),
    (":ref:`the guide <guide-label>`", "the guide"),
    (":ref:`plain-label`", "plain-label"),
    (":term:`iterable`", "iterable"),
    (":doc:`Intro <intro>`", "Intro"),
    (":pep:`8`", "PEP 8"),
    (":rfc:`2616`", "RFC 2616"),
    (":issue:`123`", "123"),
    (":madeup:`stuff`", "stuff"),
])
def test_roles(src, out):
    assert rst(f"See {src} now.\n") == f"See {out} now.\n"


def test_a_title_in_a_code_role_is_taken_as_written_and_a_std_role_is_plain_text():
    assert rst("See :class:`~a.b <t>` and :std:ref:`the label <x>` now.\n") == "See `~a.b` and the label now.\n"


def test_hyperlinks_and_footnote_references():
    assert rst("Read `the docs <https://x.org/a>`_ and `other`_ and `third <u>`__.\n") == "Read the docs and other and third.\n"
    assert rst("A claim [1]_ and another [#]_ and [#note]_.\n") == "A claim and another and.\n"
    assert rst("Strict (see [#rfc-errata]_ ) subset [cit-2023]_.\n") == "Strict (see ) subset.\n"


def test_backticked_text_without_a_role_and_emphasis_are_kept():
    text = "A `Name` in *italics* and **bold** and |sub|.\n"
    assert rst(text) == text


def test_roles_inside_inline_literals_are_not_touched():
    assert rst("Write ``:func:`x` `` literally.\n").startswith("Write `:func:`x` ")


def test_field_lists_and_definition_lists_are_left_alone():
    text = ":param x: The value.\n:returns: Nothing.\n\nterm\n    Definition text.\n"
    assert rst(text) == text


# --- whitespace -----------------------------------------------------------------------------

def test_trailing_spaces_trimmed_and_blank_runs_collapsed():
    assert rst("a   \n\n\n\n\nb\t\n") == "a\n\nb\n"


def test_crlf_and_empty_results():
    assert "\r" not in rst("a\r\n\r\nb\r\n")
    assert rst(".. toctree::\n\n   a\n") == ""
    assert rst("") == ""


# --- properties -----------------------------------------------------------------------------

FIXTURES = [
    "Title\n=====\n\nIntro with :func:`os.path.join` and ``code``.\n\nExample::\n\n    x = 1\n\n.. versionadded:: 3.4\n\n"
    ".. function:: f(x)\n   :module: m\n\n   Does *f*. See :pep:`8`.\n\n   .. note::\n\n      Careful `link <u>`_.\n",
    "Plain paragraph only.\n\nAnother one.\n",
    ".. toctree::\n   :hidden:\n\n   a\n   b\n\nSection\n-------\n\n.. code-block:: python\n\n   print(1)\n\nEnd :ref:`x <y>`.\n",
]


@pytest.mark.parametrize("text", FIXTURES)
def test_properties_hold_on_fixtures(text):
    out = rst(text)
    assert normalize_text(out) == out
    assert rst(out) == out  # idempotent
    assert is_subsequence(words(out), words(text))


def test_code_survives_inside_fences():
    out = rst("Example::\n\n    a = 1\n\n    b = 2\n\n.. code-block:: c\n\n   int x;\n")
    assert "a = 1\n\nb = 2" in out and "int x;" in out


def test_properties_hold_on_random_mixes_of_constructs():
    pieces = ["Some prose here.\n", "Title\n=====\n", "Sub\n---\n", "\n", "\n\n\n", "text   \n", "Example::\n", "    code line\n",
              ".. note::\n", "   indented body\n", ".. toctree::\n", "   :hidden:\n", ".. versionadded:: 3.1\n",
              ".. function:: f(x)\n", ".. code-block:: python\n", "   x = 1\n", "See :func:`a.b` and ``lit``.\n",
              "A `link <u>`_ here.\n", ":pep:`8`\n", "----\n", ".. comment\n", "=====  =====\n", f"{TICKS}\n", "a [1]_ ref\n"]
    rng = random.Random(11)
    for _ in range(400):
        text = "".join(rng.choice(pieces) for _ in range(rng.randint(1, 14)))
        out = rst(text)
        assert normalize_text(out) == out, repr(text)
        assert is_subsequence(words(out), words(text)), repr(text)
        assert out == rst(text)
        assert rst(out) == out, repr(text)
