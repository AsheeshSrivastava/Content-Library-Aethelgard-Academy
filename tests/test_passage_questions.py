"""Passage-level question file: loading, per-question checks and the mix rule (spec 16 sections 2 to 4)."""
import pytest

from ragcore.passage_questions import (KINDS, PassageQuestion, check_mix, check_questions, load_passage_questions)

QUOTE = "The retry decorator runs the wrapped function again when it raises a timeout error, up to three times."
SOURCES = {"lib/docs/retry-guide.md": f"# Retry guide\n\n{QUOTE}\n\nOther text about caching.\n",
           "lib/docs/cache.md": "# Cache\n\nCaches store results so that repeated calls are quick and cheap to make.\n"}


def pq(q="What happens after a call fails because it took too long?", source="retry-guide", quote=QUOTE, kind="fact"):
    return PassageQuestion(q, source, quote, kind)


def problems(*questions, sources=SOURCES):
    return check_questions(list(questions), sources)


def write(tmp_path, text, name="a.yaml"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


# --- loading ---------------------------------------------------------------------------------------

def test_kinds():
    assert KINDS == ("fact", "how-to", "code", "number", "context")


def test_a_file_loads(tmp_path):
    path = write(tmp_path, f"questions:\n- q: Why?\n  source: retry-guide\n  quote: '{QUOTE}'\n  kind: fact\n")
    assert load_passage_questions(path) == [PassageQuestion("Why?", "retry-guide", QUOTE, "fact")]


def test_a_folder_loads_every_file_in_name_order(tmp_path):
    for name, q in (("b.yaml", "Second?"), ("a.yaml", "First?")):
        write(tmp_path, f"questions:\n- q: {q}\n  source: s\n  quote: 'x'\n  kind: fact\n", name)
    assert [x.q for x in load_passage_questions(tmp_path)] == ["First?", "Second?"]


@pytest.mark.parametrize("body, message", [
    ("", "questions"),
    ("questions: []\n", "questions"),
    ("questions:\n- q: ''\n  source: s\n  quote: x\n  kind: fact\n", "needs q"),
    ("questions:\n- q: a\n  source: s\n  quote: x\n", "kind"),
    ("questions:\n- q: a\n  source: s\n  quote: x\n  kind: other\n", "kind"),
    ("questions:\n- q: a\n  source: s\n  kind: fact\n", "quote"),
    ("questions:\n- q: a\n  source: s\n  quote: x\n  kind: fact\n  extra: 1\n", "unknown key"),
    ("questions: [1\n", "not valid YAML"),
])
def test_bad_files_are_refused(tmp_path, body, message):
    with pytest.raises(ValueError, match=message):
        load_passage_questions(write(tmp_path, body))


def test_an_empty_folder_is_refused(tmp_path):
    with pytest.raises(ValueError, match="no question files"):
        load_passage_questions(tmp_path)


# --- one question ----------------------------------------------------------------------------------

def test_a_good_question_has_no_problems():
    assert problems(pq()) == []


def test_the_source_must_be_found_and_be_one_file():
    assert any("no source" in p for p in problems(pq(source="nope")))
    two = {"a/guide.md": QUOTE, "b/guide.md": QUOTE}
    assert any("2 sources" in p for p in problems(pq(source="guide"), sources=two))
    assert problems(pq(source="a/guide.md"), sources=two) == []


def test_an_exact_source_id_wins_over_longer_ids_that_contain_it():
    src = {"lib/retry-guide.md": QUOTE, "lib/retry-guide.md.orig": QUOTE}
    assert problems(pq(source="lib/retry-guide.md"), sources=src) == []
    assert any("2 sources" in p for p in problems(pq(source="retry-guide.md"), sources=src))

def test_the_quote_must_be_in_the_source_exactly_once():
    assert any("not found" in p for p in problems(pq(quote=QUOTE.replace("retry", "retries"))))
    twice = {"lib/retry-guide.md": QUOTE + "\n\n" + QUOTE}
    assert any("2 times" in p for p in problems(pq(), sources=twice))


@pytest.mark.parametrize("quote", ["short quote", "x" * 401])
def test_quote_length_is_limited(quote):
    src = {"lib/retry-guide.md": quote}
    assert any("40 to 400" in p for p in problems(pq(quote=quote), sources=src))


@pytest.mark.parametrize("size", [40, 400])
def test_quote_length_limits_are_allowed(size):
    quote = ("alpha beta " * 100)[:size]
    src = {"lib/retry-guide.md": quote}
    assert problems(pq(quote=quote), sources=src) == []


def test_a_question_may_not_repeat_four_words_of_the_quote():
    copied = "What happens when it raises a timeout error in a retry?"
    assert any("4 words" in p for p in problems(pq(q=copied)))
    three = "Which decorator raises a timeout?"
    assert problems(pq(q=three)) == []


def test_word_runs_ignore_case_and_punctuation():
    assert any("4 words" in p for p in problems(pq(q="Does THE retry, decorator: runs? it")))


def test_a_question_may_not_name_the_file():
    assert any("names the file" in p for p in problems(pq(q="What does the retry-guide say about failures?")))
    assert problems(pq(q="What does the guide say about failures?")) == []


def test_file_names_of_six_letters_are_checked_and_shorter_ones_are_not():
    six = {"lib/retire.md": QUOTE}
    assert any("names the file" in p for p in problems(pq(source="retire.md", q="Why did retire fail?"), sources=six))
    five = {"lib/retir.md": QUOTE}
    assert problems(pq(source="retir.md", q="Why did retir fail?"), sources=five) == []


def test_each_problem_names_its_question():
    assert problems(pq(source="nope", q="Unique words here please?"))[0].startswith("'Unique words here please?'")


def test_duplicates_are_found():
    out = problems(pq(), pq(quote=QUOTE))
    assert any("same question" in p for p in out)
    other = pq(q="Why does a call get repeated?")
    assert any("same quote" in p for p in problems(pq(), other))
    assert not any("same" in p for p in problems(pq(), pq(q="Another question entirely?", quote=SOURCES["lib/docs/cache.md"][9:60], source="cache")))


def test_the_same_question_with_other_case_and_punctuation_is_a_duplicate():
    assert any("same question" in p for p in problems(pq(q="Why, really?"), pq(q="why really")))


# --- the mix of the whole set ----------------------------------------------------------------------

def make_set(kinds, sources=None):
    pool = sources or [f"s{n}" for n in range(20)]
    return [PassageQuestion(f"q{i}", pool[i % len(pool)], "x", kind) for i, kind in enumerate(kinds)]


def good_kinds():
    return ["code"] * 40 + ["number"] * 20 + ["context"] * 25 + ["fact"] * 40 + ["how-to"] * 25


def test_a_good_mix_has_no_problems():
    assert check_mix(make_set(good_kinds())) == []


def test_each_mix_rule_is_checked():
    base = good_kinds()
    assert any("code" in p for p in check_mix(make_set(["fact"] * 30 + base[30:])))
    assert any("number" in p for p in check_mix(make_set(["fact"] * 20 + base[:40] + base[60:])))
    assert any("context" in p for p in check_mix(make_set([k if k != "context" else "fact" for k in base])))
    assert any("above 40" in p for p in check_mix(make_set(["fact"] * 61 + ["code"] * 40 + ["number"] * 20 + ["context"] * 29)))


def test_mix_limits_are_exact():
    # 150 questions: code 25 % is 38 (37.5), number 10 % is 15, context 15 % is 23 (22.5), a kind at 40 % is 60.
    ok = ["code"] * 38 + ["number"] * 15 + ["context"] * 23 + ["fact"] * 60 + ["how-to"] * 14
    assert check_mix(make_set(ok)) == []
    assert any("code" in p for p in check_mix(make_set(["code"] * 37 + ["fact"] * 1 + ok[38:])))
    assert any("number" in p for p in check_mix(make_set(ok[:38] + ["fact"] * 1 + ["number"] * 14 + ok[53:])))


def test_source_limits_are_checked():
    assert any("above 12" in p for p in check_mix(make_set(good_kinds(), sources=["only"] * 3 + [f"s{i}" for i in range(20)])))
    exact = ["big"] * 12 + [f"s{i}" for i in range(138)]
    assert check_mix(make_set(good_kinds(), sources=[f"s{i}" for i in range(15)])) == []
    assert not any("above 12" in p for p in check_mix(make_set(good_kinds(), sources=exact)))
    few = [f"s{i}" for i in range(14)]
    assert any("at least 15" in p for p in check_mix(make_set(good_kinds(), sources=few)))


def test_a_small_set_is_refused_for_the_mix():
    assert any("about 150" in p for p in check_mix(make_set(["code"] * 10)))
    assert not any("about 150" in p for p in check_mix(make_set(good_kinds()[:100])))

def test_the_mix_counts_register_sources_not_files():
    qs = [PassageQuestion(f"q{i}", f"python-peps/pep-{i:04d}.rst", "x", k) for i, k in enumerate(good_kinds())]
    assert any("1 sources are covered" in p for p in check_mix(qs)) and any("above 12" in p for p in check_mix(qs))


def test_a_file_name_is_grouped_through_the_loaded_sources():
    srcs = {f"src{n}/page{i}.md": "x" for n in range(20) for i in range(8)}
    names = [f"page{i}.md" for i in range(8)]
    qs = [PassageQuestion(f"q{i}", f"src{i % 20}/page{i % 8}.md", "x", k) for i, k in enumerate(good_kinds())]
    assert check_mix(qs, srcs) == []
    short = [PassageQuestion(f"q{i}", f"page{i % 8}.md", "x", k) for i, k in enumerate(good_kinds())]
    assert check_mix(short, srcs) != []  # ambiguous names fall back to the text itself