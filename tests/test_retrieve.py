"""Retriever over a run (spec 13 section 4.5): cosine search, parent mapping, one hit per file."""
import numpy as np
import pytest

from ragcore.runs import Run
from ragcore.retrieve import make_retriever, search_passages


class AxisEmbedder:
    """Queries 'q0', 'q1', ... become unit axis vectors; the query vector of other text is zero."""
    id = "axis"
    fingerprint = "axis"
    dim = 4

    def __init__(self):
        self.kinds = []

    def embed(self, texts, *, kind):
        self.kinds.append(kind)
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            if t.startswith("q"):
                out[i, int(t[1:])] = 1.0
        return out


def make_run(rows, vecs):
    full = [{"passage_id": f"p{i}", "source_id": s, "start": 0, "end": 1, "headings": [], "parent": par}
            for i, (s, par) in enumerate(rows)]
    return Run("r", {"inputs": {}}, full, np.array(vecs, dtype=np.float32))


def unit(*xs):
    v = np.array(xs, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_best_cosine_first():
    run = make_run([("a", None), ("b", None), ("c", None)],
                   [unit(1, 0, 0, 0), unit(0, 1, 0, 0), unit(1, 1, 0, 0)])
    hits = search_passages(run, np.array([1, 0, 0, 0], dtype=np.float32))
    assert [h[0]["passage_id"] for h in hits] == ["p0", "p2", "p1"]
    assert hits[0][1] == pytest.approx(1.0) and hits[1][1] == pytest.approx(2 ** -0.5, abs=1e-5)


def test_n_raw_limits_passages():
    run = make_run([("a", None)] * 5, [unit(1, i, 0, 0) for i in range(5)])
    assert len(search_passages(run, np.array([1, 0, 0, 0], dtype=np.float32), n_raw=2)) == 2


def test_children_of_one_parent_count_once_with_best_score():
    run = make_run([("a", "P"), ("a", "P"), ("b", None)],
                   [unit(1, 1, 0, 0), unit(1, 0, 0, 0), unit(1, 2, 0, 0)])
    hits = search_passages(run, np.array([1, 0, 0, 0], dtype=np.float32))
    assert [(h[0]["parent"] or h[0]["passage_id"]) for h in hits] == ["P", "p2"]
    assert hits[0][1] == pytest.approx(1.0)


def test_retriever_collapses_to_one_hit_per_file_in_score_order():
    run = make_run([("a", None), ("b", None), ("a", None), ("c", None)],
                   [unit(1, 1, 0, 0), unit(1, 2, 0, 0), unit(1, 0, 0, 0), unit(0, 1, 0, 0)])
    hits = make_retriever(run, AxisEmbedder())("q0", 10)
    assert [h.source_id for h in hits] == ["a", "b", "c"]
    assert hits[0].score == pytest.approx(1.0)  # the better of a's two passages


def test_retriever_respects_n_and_embeds_queries_as_queries():
    run = make_run([("a", None), ("b", None), ("c", None)], [unit(1, 0, 0, 0), unit(1, 1, 0, 0), unit(1, 2, 0, 0)])
    emb = AxisEmbedder()
    assert len(make_retriever(run, emb)("q0", 2)) == 2
    assert emb.kinds == ["query"]


def test_file_collapse_happens_before_n_so_duplicates_do_not_fill_the_list():
    run = make_run([("a", None)] * 4 + [("b", None)],
                   [unit(1, 0, 0, 0), unit(1, 0.01, 0, 0), unit(1, 0.02, 0, 0), unit(1, 0.03, 0, 0), unit(1, 1, 0, 0)])
    assert [h.source_id for h in make_retriever(run, AxisEmbedder(), n_raw=200)("q0", 2)] == ["a", "b"]


def test_wrong_query_width_refused():
    run = make_run([("a", None)], [unit(1, 0, 0, 0)])
    with pytest.raises(ValueError, match="width"):
        search_passages(run, np.ones(3, dtype=np.float32))


def test_zero_n_raw_refused():
    run = make_run([("a", None)], [unit(1, 0, 0, 0)])
    with pytest.raises(ValueError):
        search_passages(run, np.ones(4, dtype=np.float32), n_raw=0)


def test_empty_run_returns_nothing():
    run = make_run([], np.zeros((0, 4)))
    assert make_retriever(run, AxisEmbedder())("q0", 3) == []


def test_equal_scores_keep_row_order():
    run = make_run([(f"s{i}", None) for i in range(400)], [unit(1, 0, 0, 0)] * 400)
    hits = search_passages(run, np.array([1, 0, 0, 0], dtype=np.float32), n_raw=150)
    assert [h[0]["passage_id"] for h in hits] == [f"p{i}" for i in range(150)]  # the cut keeps the first rows
