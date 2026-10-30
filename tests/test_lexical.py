"""Keyword index (spec 15 section 3.1): tokens that keep code names, and BM25 checked by hand."""
import math

import pytest

from ragcore.lexical import Bm25Index, tokenize


# --- tokenize -----------------------------------------------------------------------------------

def test_plain_words_are_lowercased_numbers_kept():
    assert tokenize("Retrieval finds 3 passages!") == ["retrieval", "finds", "3", "passages"]


def test_snake_case_keeps_the_whole_name_and_its_parts():
    assert tokenize("call load_snapshot now") == ["call", "load_snapshot", "load", "snapshot", "now"]


def test_dotted_names_keep_the_whole_name_and_its_parts():
    assert tokenize("use os.path.join") == ["use", "os.path.join", "os", "path", "join"]


def test_camel_case_and_acronyms_are_split():
    assert tokenize("HFEncoder") == ["hfencoder", "hf", "encoder"]
    assert tokenize("parseHTTPResponse") == ["parsehttpresponse", "parse", "http", "response"]


def test_dunder_names_keep_the_whole_name():
    assert tokenize("__init__") == ["__init__", "init"]


def test_a_sentence_end_does_not_stick_to_the_word():
    assert tokenize("It ends here.") == ["it", "ends", "here"]


def test_letters_and_digits_split_in_camel_runs():
    assert tokenize("py312") == ["py312", "py", "312"]


def test_nothing_to_index():
    assert tokenize("") == [] and tokenize("  -- ** !! ") == []


# --- BM25 ---------------------------------------------------------------------------------------

DOCS = ["apple banana", "banana banana cherry", "cherry"]


def idf(n, df):
    return math.log(1 + (n - df + 0.5) / (df + 0.5))


def test_single_term_score_matches_the_hand_calculation():
    index = Bm25Index(DOCS)
    (row, score), = index.search("apple")
    # N=3, df=1, tf=1, doc length 2 = average length 2, so the length factor is 1 and tf part is (k1+1)/(1+k1)=1.
    assert row == 0 and score == pytest.approx(idf(3, 1), rel=1e-6)


def test_repeated_term_and_longer_document_score_as_the_formula_says():
    index = Bm25Index(DOCS)
    hits = dict(index.search("banana"))
    avgdl = (2 + 3 + 1) / 3
    k = 1.2 * (1 - 0.75 + 0.75 * 3 / avgdl)
    assert hits[1] == pytest.approx(idf(3, 2) * 2 * 2.2 / (2 + k), rel=1e-6)
    assert hits[0] == pytest.approx(idf(3, 2), rel=1e-6)
    assert [r for r, _ in index.search("banana")] == [1, 0]


def test_scores_of_several_terms_add_up():
    index = Bm25Index(DOCS)
    both = dict(index.search("apple banana"))
    assert both[0] == pytest.approx(idf(3, 1) + idf(3, 2), rel=1e-6)


def test_query_terms_count_once_even_if_repeated():
    index = Bm25Index(DOCS)
    assert index.search("apple apple apple") == index.search("apple")


def test_unknown_terms_and_empty_queries_return_nothing():
    index = Bm25Index(DOCS)
    assert index.search("zebra") == [] and index.search("") == [] and index.search(" -- ") == []


def test_an_index_without_documents_returns_nothing():
    assert Bm25Index([]).search("apple") == []


def test_documents_without_tokens_are_fine():
    index = Bm25Index(["", "apple", " -- "])
    assert [r for r, _ in index.search("apple")] == [1]


def test_equal_scores_are_ordered_by_row_position():
    index = Bm25Index(["apple", "apple", "apple", "other words here"])
    assert [r for r, _ in index.search("apple")] == [0, 1, 2]


def test_the_number_of_results_is_limited():
    index = Bm25Index(["apple"] * 10)
    assert [r for r, _ in index.search("apple", n=3)] == [0, 1, 2]
    with pytest.raises(ValueError, match="n"):
        index.search("apple", n=0)


def test_an_exact_identifier_beats_a_loose_word():
    docs = ["The snapshot is a folder of files.", "Call load_snapshot to read it.", "Nothing relevant here at all."]
    index = Bm25Index(docs)
    assert [r for r, _ in index.search("load_snapshot")][0] == 1
    assert {r for r, _ in index.search("snapshot")} == {0, 1}


def test_same_input_same_output():
    index = Bm25Index(DOCS * 5)
    assert index.search("banana cherry") == index.search("banana cherry")


def test_settings_are_checked():
    for bad in ({"k1": -1}, {"b": -0.1}, {"b": 1.5}):
        with pytest.raises(ValueError):
            Bm25Index(DOCS, **bad)
