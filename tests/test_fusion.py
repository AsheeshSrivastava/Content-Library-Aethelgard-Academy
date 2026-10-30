"""Reciprocal rank fusion (spec 15 section 3.2)."""
import pytest

from ragcore.fusion import rrf


def test_scores_add_across_lists():
    fused = rrf([[1, 2, 3], [3, 1, 4]])
    assert [row for row, _ in fused] == [1, 3, 2, 4]
    scores = dict(fused)
    assert scores[1] == pytest.approx(1 / 61 + 1 / 62)
    assert scores[3] == pytest.approx(1 / 63 + 1 / 61)
    assert scores[2] == pytest.approx(1 / 62) and scores[4] == pytest.approx(1 / 63)


def test_k_changes_the_scores():
    assert dict(rrf([[7]], k=10))[7] == pytest.approx(1 / 11)


def test_an_empty_list_lets_the_other_pass_through_in_order():
    assert [row for row, _ in rrf([[], [9, 4, 6]])] == [9, 4, 6]
    assert [row for row, _ in rrf([[9, 4, 6], []])] == [9, 4, 6]


def test_all_empty_gives_nothing():
    assert rrf([]) == [] and rrf([[], []]) == []


def test_ties_go_to_the_better_best_rank_then_the_row_position():
    # 5 is first in list one and absent from list two; 7 is first in list two. Same score, same best rank: row 5 first.
    assert [row for row, _ in rrf([[5], [7]])] == [5, 7]
    # 9 is first in one list and third in the other, 8 second in both: 1/61+1/63 is a little above 2/62, so 9 leads.
    assert [row for row, _ in rrf([[9, 8, 2], [3, 8, 9]])][:2] == [9, 8]
    # Equal scores, equal best rank: the lower row position first.
    fused = rrf([[4, 6], [6, 4]])
    assert [row for row, _ in fused] == [4, 6]


def test_a_row_repeated_in_one_list_counts_once_at_its_first_rank():
    assert dict(rrf([[1, 2, 1]]))[1] == pytest.approx(1 / 61)


def test_the_number_of_results_is_limited():
    fused = rrf([[1, 2, 3, 4], [4, 3, 2, 1]], n=2)
    assert len(fused) == 2
    with pytest.raises(ValueError, match="n"):
        rrf([[1]], n=0)


def test_k_must_be_at_least_one():
    with pytest.raises(ValueError, match="k"):
        rrf([[1]], k=0)


def test_same_input_same_output():
    lists = [[3, 1, 2, 9], [2, 3, 8, 1]]
    assert rrf(lists) == rrf(lists)
