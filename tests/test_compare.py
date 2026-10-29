"""Paired comparison of runs (spec 13 sections 4.6 and 5)."""
import pytest

from ragcore.compare import QResult, markdown_table, paired, score_questions, split_sets, tally, verdict
from ragcore.evalset import Hit, Question, QuestionSet


def retriever_from(table):
    return lambda q, n: [Hit(s, 1.0 - i / 100) for i, s in enumerate(table[q])][:n]


QS = [Question("q1", ("alpha",)), Question("q2", ("beta",)), Question("q3", ("gamma",)), Question("q4", ("delta",))]
TABLE = {
    "q1": ["x/alpha", "y"],                      # first
    "q2": ["y", "x/beta", "z"],                  # second -> top 3, not first
    "q3": ["y", "z", "w", "v", "u", "gamma"],    # rank 6 -> not top 3, inside 10
    "q4": ["y", "z"],                            # missing
}


def test_score_questions_fields():
    r = score_questions(QS, retriever_from(TABLE))
    assert [(x.first, x.top_k) for x in r] == [(True, True), (False, True), (False, False), (False, False)]
    assert [round(x.rr, 4) for x in r] == [1.0, 0.5, round(1 / 6, 4), 0.0]
    assert [x.recall10 for x in r] == [True, True, True, False]
    assert r[0].q == "q1"


def test_depth_limits_recall_and_rr():
    r = score_questions(QS, retriever_from(TABLE), depth=5)
    assert r[2].recall10 is False and r[2].rr == 0.0


def test_tally():
    t = tally(score_questions(QS, retriever_from(TABLE)))
    assert t["total"] == 4 and t["first"] == 1 and t["top_k"] == 2 and t["recall10"] == 3
    assert t["mrr"] == pytest.approx((1 + 0.5 + 1 / 6) / 4)


def test_tally_of_nothing():
    assert tally([]) == {"total": 0, "first": 0, "top_k": 0, "recall10": 0, "mrr": 0.0}


def res(q, first, top):
    return QResult(q, first, top, 1.0 if first else 0.0, top)


def test_paired_counts_only_differences():
    a = [res("q1", True, True), res("q2", False, True), res("q3", False, False), res("q4", True, True)]
    b = [res("q1", True, True), res("q2", True, True), res("q3", False, True), res("q4", False, False)]
    p = paired(a, b)
    assert p == {"only_a_first": 1, "only_b_first": 1, "only_a_top_k": 1, "only_b_top_k": 1,
                 "net_first": 0, "net_top_k": 0}


def test_paired_net_is_b_minus_a():
    a = [res("q1", False, False)] * 1
    b = [res("q1", True, True)]
    p = paired(a, b)
    assert p["net_first"] == 1 and p["net_top_k"] == 1


def test_paired_needs_the_same_questions():
    with pytest.raises(ValueError, match="same questions"):
        paired([res("q1", True, True)], [res("q2", True, True)])


def make(first, top, n=100):
    return [QResult(f"q{i}", i < first, i < top, 0.0, i < top) for i in range(n)]


@pytest.mark.parametrize("d_first,d_top,expected", [
    (5, 0, "beats"), (4, 0, "close"), (0, 3, "beats"), (0, 2, "close"), (0, 0, "close"),
    (5, -2, "beats"), (5, -3, "worse"), (-2, 3, "beats"), (-3, 3, "worse"), (-2, -2, "close"),
    (-3, 0, "worse"), (0, -3, "worse"),
])
def test_verdict_rule(d_first, d_top, expected):
    base = make(50, 70, 200)
    assert verdict(make(50 + d_first, 70 + d_top, 200), base) == expected


def test_verdict_clearly_worse():
    assert verdict(make(40, 60, 200), make(50, 70, 200)) == "worse"


def test_verdict_needs_same_questions():
    with pytest.raises(ValueError):
        verdict(make(5, 5, 10), make(5, 5, 11))


def test_split_sets_primary_excludes_rag_questions():
    sets = [QuestionSet("a.yaml", ()), QuestionSet("rag-questions.yaml", ()), QuestionSet("b.yaml", ())]
    primary, secondary = split_sets(sets)
    assert [s.name for s in primary] == ["a.yaml", "b.yaml"] and [s.name for s in secondary] == ["rag-questions.yaml"]


def test_split_sets_accepts_names_without_extension():
    primary, secondary = split_sets([QuestionSet("rag-questions", ()), QuestionSet("x", ())])
    assert [s.name for s in secondary] == ["rag-questions"] and [s.name for s in primary] == ["x"]


def test_markdown_table_lists_every_run_with_verdict():
    base = make(50, 70, 200)
    table = markdown_table({"rules": base, "fixed": make(57, 70, 200)}, baseline="rules")
    lines = table.splitlines()
    assert lines[0].startswith("| Run |") and len(lines) == 4
    assert "rules" in lines[2] and "baseline" in lines[2]
    assert "fixed" in lines[3] and "beats" in lines[3] and "57" in lines[3]


def test_markdown_table_unknown_baseline():
    with pytest.raises(ValueError, match="baseline"):
        markdown_table({"a": make(1, 1, 5)}, baseline="zzz")


def test_rank_just_outside_top_k_is_not_top_k():
    r = score_questions([Question("q", ("a",))], retriever_from({"q": ["x", "y", "z", "a"]}))[0]
    assert (r.first, r.top_k, r.recall10, r.rr) == (False, False, True, 0.25)


def test_rank_beyond_depth_does_not_count_for_rr_or_recall():
    long_list = lambda q, n: [Hit(s, 0.5) for s in ["x", "y", "z", "a"]]  # a retriever that ignores n
    r = score_questions([Question("q", ("a",))], long_list, k=3, depth=3)[0]
    assert (r.top_k, r.recall10, r.rr) == (False, False, 0.0)
