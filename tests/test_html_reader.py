"""Spec 10: HTML reader. Written before the code."""
import random
import re

import pytest

from ragcore.errors import LoadError
from ragcore.loaders.html import HTML_VERSION, read_html
from ragcore.loaders.registry import DEFAULT_READERS

TICKS = "`" * 3


def read(html: str) -> str:
    return read_html(html.encode("utf-8"))


def test_version_and_registration():
    assert HTML_VERSION == "html-1"
    assert DEFAULT_READERS[".html"][0] == HTML_VERSION and DEFAULT_READERS[".htm"][0] == HTML_VERSION


def test_headings_and_paragraphs():
    assert read("<h1>Guide</h1><p>Some  text\nhere.</p><h3>Deep</h3><p>More.</p>") == \
        "# Guide\n\nSome text here.\n\n### Deep\n\nMore.\n"


@pytest.mark.parametrize("tag", ["script", "style", "head", "nav", "footer", "template", "noscript", "svg", "iframe"])
def test_dropped_elements_vanish_with_content(tag):
    assert read(f"<p>keep</p><{tag}>secret words</{tag}><p>this</p>") == "keep\n\nthis\n"


def test_title_becomes_a_heading_only_without_an_h1():
    assert read("<html><head><title>My page</title></head><body><p>Body.</p></body></html>") == "# My page\n\nBody.\n"
    assert read("<head><title>My page</title></head><h1>Real</h1><p>Body.</p>") == "# Real\n\nBody.\n"


def test_lists_and_br():
    assert read("<ul><li>one</li><li>two<br>three</li></ul>") == "- one\n- two\nthree\n"


def test_table_cells_joined_in_a_row():
    assert read("<table><tr><th>a</th><th>b</th></tr><tr><td>1</td><td>2</td></tr></table>") == "a | b\n1 | 2\n"


def test_pre_becomes_a_fence_unchanged():
    out = read("<p>x</p><pre>def f():\n    return  1\n\n\nz</pre>")
    assert out == f"x\n\n{TICKS}\ndef f():\n    return  1\n\n\nz\n{TICKS}\n"


def test_fence_grows_when_the_code_has_backticks():
    out = read(f"<pre>a {TICKS} b</pre>")
    assert out.startswith("`" * 4 + "\n") and out.rstrip().endswith("`" * 4)


def test_inline_code_links_and_images():
    assert read('<p>Use <code>x &lt; y</code> from <a href="http://e.com">the docs</a><img src="a.png" alt="logo"></p>') == \
        "Use `x < y` from the docslogo\n"


def test_entities_are_decoded_but_not_parsed():
    assert read("<p>&lt;div&gt; &amp; &quot;q&quot; a &lt; b</p>") == '<div> & "q" a < b\n'


def test_a_bare_less_than_in_text_stays():
    assert read("<p>if a < b then c</p>") == "if a < b then c\n"


def test_template_syntax_is_removed_but_not_in_pre():
    out = read('{% extends "base.html" %}{# note #}<p>Hello {{ name }} world</p><pre>{{ keep }}</pre>')
    assert out == f"Hello world\n\n{TICKS}\n{{{{ keep }}}}\n{TICKS}\n"


def test_nothing_left_is_empty():
    assert read("<script>x</script>") == ""
    assert read("") == ""


def test_the_real_templates_are_stubs_through_the_registry():
    reader = DEFAULT_READERS[".html"][1]
    page = '{% extends "base.html" %}{% block x %}<title>{{ page.title }} | {{ site }}</title>{% endblock %}'
    with pytest.raises(LoadError) as e:
        reader(page.encode())
    assert str(e.value) in ("stub", "empty")


def test_a_bom_is_fine():
    assert read_html(b"\xef\xbb\xbf<p>hi</p>") == "hi\n"


def test_crlf_is_normalised():
    assert read("<p>a</p>\r\n<p>b</p>") == "a\n\nb\n"


PIECES = ["<h2>Title</h2>", "<p>Some words here.</p>", "<ul><li>a</li><li>b</li></ul>", "<pre>x = 1\n\ny</pre>",
          "<p>A <code>c</code> and <a href='u'>link</a>.</p>", "<script>no</script>", "<p>a &lt; b</p>",
          "<table><tr><td>1</td><td>2</td></tr></table>", "<p>line<br>break</p>", "<nav>menu</nav>"]


def test_random_mixes_are_deterministic_and_clean():
    rng = random.Random(7)
    for _ in range(200):
        html = "".join(rng.choice(PIECES) for _ in range(rng.randint(1, 8)))
        out = read(html)
        assert read(html) == out
        assert out == "" or out.endswith("\n") and not out.endswith("\n\n") and "\n\n\n" not in out and "\x00" not in out


def test_prose_is_never_lost_and_dropped_words_never_appear():
    rng = random.Random(3)
    for _ in range(100):
        html = "".join(rng.choice(PIECES) for _ in range(rng.randint(1, 8)))
        out = read(html)
        assert "menu" not in out and "no" not in out.split()
        kept = re.sub(r"<(script|nav)>.*?</\1>", "", html)
        words = re.findall(r"\w+", re.sub(r"<[^>]+>", " ", kept))
        pos = 0
        for w in re.findall(r"\w+", out):  # The output words are a subsequence of the input words.
            pos = words.index(w, pos) + 1


# --- found by the mutation check ------------------------------------------------------------

def test_nested_dropped_elements_end_together():
    assert read("<nav>a<svg>b</svg>c</nav><p>keep</p>") == "keep\n"


def test_a_self_closing_dropped_tag_does_not_hide_the_page():
    assert read("<script src='x'/><p>keep</p>") == "keep\n"


def test_pre_keeps_its_first_line_indentation():
    assert read("<pre>  indented\n    more</pre>") == f"{TICKS}\n  indented\n    more\n{TICKS}\n"


def test_extra_spaces_and_blank_lines_collapse():
    assert read("<p>a <span> </span> b</p>") == "a b\n"
    assert read("<p>a</p><div></div><div></div><div></div><p>b</p>") == "a\n\nb\n"


def test_an_empty_list_item_is_dropped():
    assert read("<ul><li></li><li>x</li></ul>") == "- x\n"


def test_cells_outside_a_row_still_join():
    assert read("<table><td>a</td><td>b</td></table>") == "a | b\n"
