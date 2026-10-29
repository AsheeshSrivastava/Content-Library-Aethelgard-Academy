"""Spec 12: question files, question checks and scoring. Written before the code."""
import pytest

from ragcore.evalset import (Hit, Question, QuestionSet, Report, check_questions, evaluate, evaluate_sets,
                             load_question_dir, load_question_file)


def write(tmp_path, text, name="q.yaml"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


# --- loading --------------------------------------------------------------------------------

def test_load_question_file(tmp_path):
    path = write(tmp_path, "# note\nquestions:\n- q: How do I install it?\n  expect: [install, faq]\noutside:\n- What is the weather?\n",
                 "ruff.yaml")
    s = load_question_file(path)
    assert s.name == "ruff" and s.questions == (Question("How do I install it?", ("install", "faq")),)
    assert s.outside == ("What is the weather?",)


def test_outside_is_optional(tmp_path):
    assert load_question_file(write(tmp_path, "questions:\n- {q: 'A?', expect: [a]}\n")).outside == ()


@pytest.mark.parametrize("text", [
    "", "[]", "questions: {}", "questions:\n- {q: 'A?'}", "questions:\n- {q: 'A?', expect: []}",
    "questions:\n- {q: 'A?', expect: a}", "questions:\n- {q: '  ', expect: [a]}", "questions:\n- {q: 'A?', expect: ['']}",
    "questions:\n- {q: 'A?', expect: [1]}", "questions:\n- {q: 'A?', expect: [a]}\noutside: x",
    "questions:\n- {q: 'A?', expect: [a]}\noutside: [1]", "questions:\n- {q: 'A?', expect: [a]}\noutside: ['  ']",
    "questions:\n- x", "questions: [", "questions:\n- {q: 5, expect: [a]}",
])
def test_bad_files_are_refused_with_the_file_name(tmp_path, text):
    with pytest.raises(ValueError, match="q.yaml"):
        load_question_file(write(tmp_path, text))


def test_load_question_dir_is_sorted_and_ignores_other_files(tmp_path):
    write(tmp_path, "questions:\n- {q: 'B?', expect: [b]}\n", "b.yaml")
    write(tmp_path, "questions:\n- {q: 'A?', expect: [a]}\n", "a.yaml")
    write(tmp_path, "nope", "notes.txt")
    assert [s.name for s in load_question_dir(tmp_path)] == ["a", "b"]


def test_load_question_dir_must_find_something(tmp_path):
    with pytest.raises(ValueError, match="no question files"):
        load_question_dir(tmp_path)


# --- checks ---------------------------------------------------------------------------------

IDS = [f"lib/docs/page{i}.md" for i in range(8)] + ["lib/faq.md", "other/guide.md"]


def qs(*questions, outside=()):
    return QuestionSet("t", tuple(Question(q, tuple(e)) for q, e in questions), tuple(outside))


def kinds(problems):
    return sorted(p.kind for p in problems)


def test_a_sound_set_has_no_problems():
    assert check_questions([qs(("A?", ["faq"]), ("B?", ["guide", "page3"]))], IDS) == []


def test_a_piece_matching_nothing_is_reported_but_the_question_still_reachable():
    problems = check_questions([qs(("A?", ["faq", "missing"]))], IDS)
    assert kinds(problems) == ["no-match"] and "missing" in problems[0].message and problems[0].set == "t"


def test_a_question_with_no_matching_piece_is_unreachable():
    assert kinds(check_questions([qs(("A?", ["nope"]))], IDS)) == ["no-match", "unreachable"]


def test_a_broad_piece_is_reported():
    assert kinds(check_questions([qs(("A?", ["lib/"]))], IDS)) == ["broad"]
    assert check_questions([qs(("A?", ["lib/"]))], IDS, broad_share=0.9) == []


def test_broad_share_is_strictly_greater_than():
    ids = ["a/x", "b/x", "c/y", "d/y"]  # "x" appears in exactly half
    assert check_questions([qs(("A?", ["x"]))], ids, broad_share=0.5) == []
    assert kinds(check_questions([qs(("A?", ["x"]))], ids, broad_share=0.49)) == ["broad"]


def test_a_duplicate_question_is_reported_across_sets():
    a, b = qs(("Same?", ["faq"])), qs(("Same?", ["guide"]))
    assert kinds(check_questions([a, b], IDS)) == ["duplicate"]


def test_an_outside_question_that_is_also_answerable_is_reported():
    assert kinds(check_questions([qs(("A?", ["faq"]), outside=["A?"])], IDS)) == ["outside-is-answerable"]


def test_no_source_ids_means_every_piece_is_a_problem():
    assert kinds(check_questions([qs(("A?", ["faq"]))], [])) == ["no-match", "unreachable"]


# --- scoring --------------------------------------------------------------------------------

def fake(results):
    """A retriever that answers from a dict of query -> [(source_id, score)]."""
    def retrieve(query, n):
        return [Hit(s, score) for s, score in results.get(query, [])][:n]
    return retrieve


QUESTIONS = (Question("one", ("alpha",)), Question("two", ("beta",)), Question("three", ("gamma",)),
             Question("four", ("delta",)))


def test_first_top_k_and_misses():
    r = evaluate(QUESTIONS, (), fake({
        "one": [("x/alpha.md", 0.9), ("x/z.md", 0.5)],              # first
        "two": [("x/z.md", 0.8), ("x/y.md", 0.7), ("x/beta.md", 0.6)],  # third
        "three": [("x/z.md", 0.8), ("x/y.md", 0.7), ("x/w.md", 0.6), ("x/gamma.md", 0.5)],  # fourth: outside k=3
        "four": [],
    }), k=3)
    assert (r.total, r.first, r.top_k) == (4, 1, 2)
    assert r.misses == [("three", ["x/z.md", "x/y.md", "x/w.md"]), ("four", [])]


def test_scores_use_the_best_right_hit_within_depth():
    results = {"one": [("x/z.md", 0.9), ("x/alpha.md", 0.4), ("y/alpha2.md", 0.8)],
               "two": [("x/z.md", 0.9)] * 3 + [("x/beta.md", 0.3)]}
    r = evaluate(QUESTIONS[:2], (), fake(results), k=3, depth=4)
    assert r.right_scores == [0.8, 0.3] and r.right_questions == ["one", "two"]
    r = evaluate(QUESTIONS[:2], (), fake(results), k=3, depth=3)
    assert r.right_scores == [0.8] and r.right_questions == ["one"]


def test_the_retriever_is_asked_for_at_least_depth_results():
    asked = []

    def retrieve(query, n):
        asked.append(n)
        return []
    evaluate(QUESTIONS[:1], ("out",), retrieve, k=3, depth=10)
    evaluate(QUESTIONS[:1], (), retrieve, k=20, depth=10)
    assert asked == [10, 1, 20]


def test_outside_scores_and_threshold():
    r = evaluate(QUESTIONS[:2], ("o1", "o2"), fake({
        "one": [("x/alpha.md", 0.8)], "two": [("x/beta.md", 0.6)], "o1": [("x/q.md", 0.3)], "o2": []}))
    assert r.outside_scores == [0.3, 0.0] and r.outside_questions == ["o1", "o2"]
    assert r.lowest_right == 0.6 and r.highest_outside == 0.3
    assert r.suggested_threshold == pytest.approx(0.45)


def test_overlapping_scores_have_no_threshold():
    r = evaluate(QUESTIONS[:1], ("o",), fake({"one": [("x/alpha.md", 0.5)], "o": [("x/q.md", 0.5)]}))
    assert r.suggested_threshold is None
    assert "overlap" in r.summary()


def test_no_threshold_without_both_kinds_of_scores():
    assert evaluate(QUESTIONS[:1], (), fake({"one": [("x/alpha.md", 0.5)]})).suggested_threshold is None
    assert evaluate((), ("o",), fake({})).suggested_threshold is None


def test_an_empty_run_does_not_divide_by_zero():
    r = evaluate((), (), fake({}))
    assert r.total == 0 and "0 of 0 (0%)" in r.summary()


def test_summary_lines():
    r = evaluate(QUESTIONS[:2], ("o1",), fake({"one": [("x/alpha.md", 0.8)], "o1": [("x/q.md", 0.3)]}), k=3)
    text = r.summary()
    assert "Right lesson first for 1 of 2 (50%)" in text and "top 3 for 1 of 2 (50%)" in text
    assert "missed: two -> nothing" in text
    assert "weakest right match 0.800 (one)" in text and "strongest match for a question outside" in text
    assert "suggested threshold: 0.550" in text


def test_a_piece_may_match_anywhere_in_the_path():
    r = evaluate(QUESTIONS[:1], (), fake({"one": [("src/alphabet/readme.md", 0.9)]}))
    assert r.first == 1


def test_evaluate_sets_gives_one_report_per_set_and_a_total():
    a = QuestionSet("a", QUESTIONS[:1], ("oa",))
    b = QuestionSet("b", QUESTIONS[1:2], ())
    reports = evaluate_sets([a, b], fake({"one": [("x/alpha.md", 0.9)], "two": [], "oa": [("x/q.md", 0.1)]}))
    assert list(reports) == ["a", "b", "all"]
    assert (reports["a"].first, reports["b"].first, reports["all"].first) == (1, 0, 1)
    assert reports["all"].total == 2 and reports["all"].outside_scores == [0.1]
    assert isinstance(reports["all"], Report)


def test_a_retriever_that_returns_too_many_is_cut_at_depth():
    results = {"one": [("x/z.md", 0.9)] * 3 + [("x/alpha.md", 0.4)]}
    r = evaluate(QUESTIONS[:1], (), lambda query, n: [Hit(s, sc) for s, sc in results["one"]], k=2, depth=3)
    assert r.right_scores == [] and r.top_k == 0


def test_evaluate_sets_adds_up_every_field():
    a = QuestionSet("a", QUESTIONS[:2], ("oa",))
    b = QuestionSet("b", QUESTIONS[2:], ("ob",))
    reports = evaluate_sets([a, b], fake({"one": [("x/alpha.md", 0.9)], "two": [("x/z.md", 0.5), ("x/beta.md", 0.4)],
                                          "three": [], "four": [("x/delta.md", 0.7)], "oa": [("x/q.md", 0.1)]}))
    total = reports["all"]
    assert (total.total, total.first, total.top_k) == (4, 2, 3)
    assert total.misses == [("three", [])]
    assert total.right_questions == ["one", "two", "four"] and total.right_scores == [0.9, 0.4, 0.7]
    assert total.outside_questions == ["oa", "ob"] and total.outside_scores == [0.1, 0.0]
