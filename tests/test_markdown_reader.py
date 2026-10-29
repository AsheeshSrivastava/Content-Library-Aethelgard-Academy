"""Spec 07: the Markdown and MDX reader. Written before the code."""
import random
import re

import pytest

from ragcore.contracts import normalize_text
from ragcore.loaders.markdown import read_md, read_mdx

TICKS = "`" * 3


def md(text):
    return read_md(text.encode("utf-8"))


def mdx(text):
    return read_mdx(text.encode("utf-8"))


def words(text):
    return re.findall(r"\w+", text)


def is_subsequence(small, big):
    it = iter(big)
    return all(w in it for w in small)


# --- decoding -------------------------------------------------------------------------------

def test_utf8_with_bom_is_fine_and_bad_bytes_raise():
    assert "Hello" in read_md(b"\xef\xbb\xbfHello world")
    with pytest.raises(UnicodeDecodeError):
        read_md(b"\xff\xfe\xfa bad")


# --- front matter ---------------------------------------------------------------------------

def test_front_matter_removed_and_title_becomes_a_heading():
    out = md("---\ntitle: Pipelines\nsidebar_position: 3\ntags: [a, b]\n---\nBody text here.\n")
    assert out == "# Pipelines\n\nBody text here.\n"


def test_title_not_added_when_the_body_already_starts_with_a_heading():
    out = md("---\ntitle: Pipelines\n---\n\n# Pipelines, in full\n\nText.\n")
    assert out == "# Pipelines, in full\n\nText.\n"


def test_front_matter_without_title_is_just_dropped():
    assert md("---\nsidebar_position: 3\n---\nBody.\n") == "Body.\n"


def test_unclosed_or_non_yaml_front_matter_is_left_alone():
    assert md("---\ntitle: x\nBody without end.\n").startswith("---")
    text = "---\nthis: is: not: [valid\n---\nBody.\n"
    assert md(text) == text


def test_a_horizontal_rule_in_the_middle_is_not_front_matter():
    text = "Intro line.\n\n---\n\nMore text.\n"
    assert md(text) == text


def test_quoted_and_numeric_titles():
    assert md('---\ntitle: "Hello: world"\n---\nBody.\n').startswith("# Hello: world\n")
    assert md("---\ntitle: 2026\n---\nBody.\n").startswith("# 2026\n")


# --- MDX imports and exports ----------------------------------------------------------------

def test_mdx_esm_lines_removed():
    text = 'import { Callout } from "nextra/components"\nimport Foo from \'../foo\';\nimport "./style.css"\n' \
           'export const meta = { a: 1 }\n\nReal text stays.\n'
    assert mdx(text) == "Real text stays.\n"


def test_import_in_md_is_prose():
    text = "import the data first, then run it.\n"
    assert md(text) == text
    assert md('import Foo from "x"\n\nText.\n').startswith("import Foo")


def test_mdx_prose_starting_with_import_stays():
    text = "import the data first, then run it.\n"
    assert mdx(text) == text


# --- JSX components -------------------------------------------------------------------------

def test_wrapper_component_tags_go_and_inner_text_stays():
    out = mdx('<Callout type="info" emoji="x">\nRemember the order.\n</Callout>\n')
    assert out == "Remember the order.\n"
    assert md("<Tip warning={true}>\nCareful here.\n</Tip>\n") == "Careful here.\n"


def test_self_closing_components_and_multiline_tags_disappear_with_props():
    out = mdx('Before.\n\n<Youtube id="abc"/>\n\n<PageHero\n    title="Big title"\n    lede="a > b"\n/>\n\nAfter.\n')
    assert out == "Before.\n\nAfter.\n"


def test_attribute_with_a_space_around_equals_and_nested_braces():
    assert mdx('<Callout type= "info">\nHi.\n</Callout>\n') == "Hi.\n"
    assert mdx("<Chart data={{a: {b: 1}}}>\nCaption.\n</Chart>\n") == "Caption.\n"


def test_generic_type_in_prose_is_not_a_component():
    text = "Use List<String> for this and Map<K, V> too.\n"
    assert md(text) == text


# --- HTML -----------------------------------------------------------------------------------

def test_html_tags_removed_text_kept():
    out = md('<div class="components">\n<details>\n<summary>More</summary>\nHidden text.\n</details>\n</div>\n')
    assert words(out) == ["More", "Hidden", "text"]
    assert "<" not in out


def test_img_keeps_alt_and_iframe_and_video_vanish():
    out = md('Intro.\n\n<img src="a.png" alt="A diagram of flow">\n\n<img src="b.png">\n\n'
             '<iframe width="100%"\n  src="https://x"></iframe>\n\n<video src="v.mp4"></video>\n\nOutro.\n')
    assert out == "Intro.\n\nA diagram of flow\n\nOutro.\n"


def test_br_becomes_a_line_break_and_inline_html_keeps_text():
    assert md("one<br>two<br/>three\n") == "one\ntwo\nthree\n"
    assert md('Text with <abbr title="Large Language Model">LLM</abbr> inside.\n') == "Text with LLM inside.\n"
    assert md('See <a href="https://x.org">the guide</a> now.\n') == "See the guide now.\n"


def test_unknown_lower_case_tags_in_prose_are_left_alone():
    text = "Set GITHUB_TOKEN=<token> and edit versioned_docs/version-<highest>/ for it.\n"
    assert md(text) == text


def test_comments_removed_single_and_multi_line():
    assert md("A <!-- hidden --> B\n") == "A  B\n"
    assert md("A\n<!--\nmulti\nline\n-->\nB\n") == "A\n\nB\n"
    assert mdx("A {/* note */} B\n") == "A  B\n"


# --- code is never touched ------------------------------------------------------------------

def test_fenced_code_is_untouched():
    code = f'{TICKS}jsx\nimport X from "y"\n<Callout type="info">\n<div class="a">x</div>\n<!-- c -->\n  trailing   \n\n\n\n{TICKS}\n'
    out = mdx(f"Text.\n\n{code}\nAfter.\n")
    assert code in out


def test_tilde_and_longer_fences_and_unclosed_fence():
    code = "~~~\n<Tip>\n~~~\n"
    assert code in md(f"A\n\n{code}\nB\n")
    long = f"{TICKS}`\n{TICKS}\n<Tip>\n{TICKS}`\n"
    assert long in md(long)
    unclosed = f"Intro\n\n{TICKS}\n<Tip>\nstill code\n"
    assert "<Tip>" in md(unclosed)


def test_inline_code_is_untouched():
    text = "Use `dspy.<OptimizerName>` and ``a <div> `tick` b`` and `<token>`.\n"
    assert md(text) == text


def test_an_indented_fence_inside_a_list_is_still_code():
    text = f"1. Step one:\n\n   {TICKS}python\n   <Tip>\n   import os\n   {TICKS}\n"
    assert "   <Tip>\n" in md(text)


# --- whitespace and canonical form ----------------------------------------------------------

def test_trailing_spaces_trimmed_and_blank_runs_collapsed_outside_code():
    assert md("a   \n\n\n\n\nb\t\n") == "a\n\nb\n"


def test_tables_are_kept():
    table = "| a | b |\n|---|---|\n| 1 | 2 |\n"
    assert md(table) == table


def test_crlf_input_is_normalised_by_the_reader_path():
    assert "\r" not in md("a\r\n\r\nb\r\n")


def test_empty_result_is_an_empty_string():
    assert md("---\ntitle: x\n---\n").strip() in ("", "# x")
    assert md("<!-- only a comment -->\n").strip() == ""
    assert mdx("import X from 'y'\n").strip() == ""


# --- properties -----------------------------------------------------------------------------

FIXTURES = [
    '---\ntitle: T\ntags: [a]\n---\nimport X from "y"\n\n<Callout type="info">\nText `<code>` here.\n</Callout>\n\n'
    f'{TICKS}py\nimport os\n{TICKS}\n\n<div class="a">\n<img src="x.png" alt="pic">\nMore <abbr title="t">words</abbr>.\n</div>\n',
    "Plain prose only.\n\nSecond paragraph with | pipes | in it.\n",
    "# Heading\n\n<!-- c -->\n<details>\n<summary>S</summary>\nBody\n</details>\n\n| a | b |\n|---|---|\n| 1 | 2 |\n",
    "<Tabs>\n<Tab>\nOne\n</Tab>\n<Tab>\nTwo\n</Tab>\n</Tabs>\n",
]


@pytest.mark.parametrize("text", FIXTURES)
def test_properties_hold_on_fixtures(text):
    out = mdx(text)
    assert normalize_text(out) == out
    assert mdx(out) == out  # idempotent
    assert is_subsequence(words(out), words(text))  # no prose invented or reordered
    for fence in re.findall(rf"{TICKS}.*?{TICKS}", text, re.S):
        assert fence in out
    for span in re.findall(r"`[^`\n]+`", text):
        assert span in out


def test_properties_hold_on_random_mixes_of_constructs():
    pieces = ["Some prose here.\n", "# A heading\n", "<Callout>\n", "</Callout>\n", '<div class="x">\n', "</div>\n",
              "`<inline>`\n", f"{TICKS}\n<Tip>\n{TICKS}\n", "<!-- c -->\n", "\n", "\n\n\n", "text   \n",
              '<img src="a" alt="alt words">\n', "| a | b |\n", "import X from 'y'\n", "<br>\n", "x < y and y > z\n"]
    rng = random.Random(7)
    for _ in range(300):
        text = "".join(rng.choice(pieces) for _ in range(rng.randint(1, 14)))
        out = mdx(text)
        assert normalize_text(out) == out, text
        assert mdx(out) == out, text
        assert is_subsequence(words(out), words(text)), text
        assert out == mdx(text)  # deterministic
        for span in re.findall(r"`[^`\n]+`", text):
            assert span in out, text


# --- gaps found by the mutation check ---------------------------------------------------------

def test_front_matter_that_is_not_a_mapping_is_left_alone():
    text = "---\njust some words\n---\nBody text.\n"
    assert md(text) == text


def test_iframe_and_video_vanish_with_their_fallback_text():
    assert md('Intro.\n\n<iframe src="x">Fallback words</iframe>\n\n<video src="v">No video support.</video>\n\nEnd.\n') \
        == "Intro.\n\nEnd.\n"


def test_a_fence_is_closed_only_by_its_own_character():
    text = f"{TICKS}\n<Tip>\n~~~\n<Tip>\n{TICKS}\n"
    assert md(text) == text


def test_three_backticks_with_more_on_the_same_line_are_inline_code_not_a_fence():
    assert md(f"{TICKS}inline{TICKS} and <Tip>text</Tip>\n") == f"{TICKS}inline{TICKS} and text\n"
