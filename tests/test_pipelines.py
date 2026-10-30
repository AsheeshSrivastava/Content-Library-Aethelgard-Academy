"""The five stage 3 pipelines V, L, H, VR, HR (spec 15 section 3.4), on a tiny run with fake models."""
import math

import numpy as np
import pytest

from ragcore.evalset import Hit
from ragcore.rerank import FakeReranker
from ragcore.retrieve import PIPELINES, keyword_index, make_pipeline, make_retriever
from ragcore.runs import Run

QUERY = "retry timeout"


class MapEmbedder:
    """Every query gets the vector (1, 0, 0, 0); passages are not embedded here (the run already holds vectors)."""
    id = "map"
    fingerprint = "map"
    dim = 4

    def __init__(self):
        self.calls = 0

    def embed(self, texts, *, kind):
        self.calls += 1
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        out[:, 0] = 1.0
        return out


def vec(cosine):
    return [cosine, math.sqrt(1 - cosine * cosine), 0.0, 0.0]


def setup(spec, headings=None):
    """spec: (source id, text, cosine to the query). Returns (run, texts)."""
    rows = [{"passage_id": f"p{i}", "source_id": s, "start": 0, "end": len(t), "headings": (headings or {}).get(i, []),
             "parent": None} for i, (s, t, _) in enumerate(spec)]
    run = Run("r", {"inputs": {}}, rows, np.array([vec(c) for _, _, c in spec], dtype=np.float32))
    return run, [t for _, t, _ in spec]


SPEC = [
    ("A", "unrelated words here", 0.9),
    ("B", "retry timeout handling", 0.8),
    ("C", "retry only", 0.7),
    ("D", "nothing", 0.1),
]


def files(hits):
    return [h.source_id for h in hits]


def pipeline(variant, spec=SPEC, **kw):
    run, texts = setup(spec, kw.pop("headings", None))
    return make_pipeline(variant, run, texts, embedder=kw.pop("embedder", MapEmbedder()),
                         reranker=kw.pop("reranker", FakeReranker()), **kw)


def test_the_five_variants_are_named():
    assert PIPELINES == ("V", "L", "H", "VR", "HR")


def test_v_is_the_existing_vector_retriever():
    run, texts = setup(SPEC)
    embedder = MapEmbedder()
    assert make_pipeline("V", run, texts, embedder=embedder)(QUERY, 10) == make_retriever(run, embedder)(QUERY, 10)
    assert files(pipeline("V")(QUERY, 10)) == ["A", "B", "C", "D"]


def test_l_orders_by_keyword_score_and_drops_files_without_a_match():
    hits = pipeline("L", embedder=None)(QUERY, 10)
    assert files(hits) == ["B", "C"]
    assert hits[0].score > hits[1].score > 0


def test_l_needs_no_embedder():
    assert files(pipeline("L", embedder=None)("retry", 10))[0] in ("B", "C")


def test_h_fuses_both_lists():
    hits = pipeline("H")(QUERY, 10)
    assert files(hits) == ["B", "C", "A", "D"]
    assert hits[0].score == pytest.approx(1 / 62 + 1 / 61)


def test_h_with_no_keyword_match_is_the_vector_list():
    assert files(pipeline("H")("!!! ???", 10)) == ["A", "B", "C", "D"]
    assert files(pipeline("L", embedder=None)("!!! ???", 10)) == []


def test_empty_query_does_not_crash():
    for variant in PIPELINES:
        assert isinstance(pipeline(variant)("", 10), list)


def test_vr_reranks_only_the_first_depth_passages():
    hits = pipeline("VR", depth=2)(QUERY, 10)
    assert files(hits) == ["B", "A"]  # C and D were beyond the depth and are dropped.
    assert hits[0].score == 2.0


def test_hr_reranks_the_fused_list():
    assert files(pipeline("HR", depth=3)(QUERY, 10)) == ["B", "C", "A"]


def test_hr_depth_can_change_the_answer():
    spec = [("A", "retry timeout", 0.9), ("B", "retry timeout", 0.8), ("C", "retry timeout", 0.7)]
    assert files(pipeline("HR", spec=spec, depth=2)(QUERY, 10)) == ["A", "B"]


def test_fewer_candidates_than_depth_is_fine():
    assert files(pipeline("VR", depth=30)(QUERY, 10)) == ["B", "C", "A", "D"]


def test_reranker_ties_keep_the_earlier_rank():
    spec = [("A", "retry", 0.9), ("B", "retry", 0.8), ("C", "retry", 0.7)]
    assert files(pipeline("VR", spec=spec)("retry", 10)) == ["A", "B", "C"]


def test_one_hit_per_file_keeps_the_best_passage():
    spec = [("A", "x", 0.9), ("B", "retry timeout", 0.8), ("A", "retry timeout retry", 0.7), ("C", "z", 0.1)]
    assert files(pipeline("V", spec=spec)(QUERY, 10)) == ["A", "B", "C"]
    assert files(pipeline("L", spec=spec, embedder=None)(QUERY, 10)) == ["A", "B"]  # A is found through its second passage, where "retry" twice outscores B.
    assert files(pipeline("VR", spec=spec)(QUERY, 10)) == ["B", "A", "C"]


def test_n_limits_the_answer():
    assert files(pipeline("H")(QUERY, 2)) == ["B", "C"]


def test_headings_are_searchable():
    spec = [("A", "text without the word", 0.9), ("B", "text without the word", 0.8)]
    hits = pipeline("L", spec=spec, embedder=None, headings={1: ["Guide", "Retry on timeout"]})(QUERY, 10)
    assert files(hits) == ["B"]


def test_each_variant_repeats_exactly():
    run, texts = setup(SPEC)
    for variant in PIPELINES:
        one = make_pipeline(variant, run, texts, embedder=MapEmbedder(), reranker=FakeReranker())
        two = make_pipeline(variant, run, texts, embedder=MapEmbedder(), reranker=FakeReranker())
        assert one(QUERY, 10) == two(QUERY, 10) == one(QUERY, 10)


def test_a_shared_index_gives_the_same_answer():
    run, texts = setup(SPEC)
    index = keyword_index(run, texts)
    alone = make_pipeline("H", run, texts, embedder=MapEmbedder())
    shared = make_pipeline("H", run, texts, embedder=MapEmbedder(), index=index)
    assert alone(QUERY, 10) == shared(QUERY, 10)


def test_l_does_not_embed_the_query():
    embedder = MapEmbedder()
    pipeline("L", embedder=embedder)(QUERY, 10)
    assert embedder.calls == 0


def test_missing_parts_are_refused():
    run, texts = setup(SPEC)
    with pytest.raises(ValueError, match="embedder"):
        make_pipeline("V", run, texts)
    with pytest.raises(ValueError, match="reranker"):
        make_pipeline("VR", run, texts, embedder=MapEmbedder())
    with pytest.raises(ValueError, match="unknown variant"):
        make_pipeline("X", run, texts, embedder=MapEmbedder())
    with pytest.raises(ValueError, match="texts"):
        make_pipeline("V", run, texts[:2], embedder=MapEmbedder())
    with pytest.raises(ValueError, match="depth"):
        make_pipeline("VR", run, texts, embedder=MapEmbedder(), reranker=FakeReranker(), depth=0)


def test_runs_with_parents_are_refused():
    run, texts = setup(SPEC)
    run.rows[1]["parent"] = "p0"
    with pytest.raises(ValueError, match="parent"):
        make_pipeline("H", run, texts, embedder=MapEmbedder())


def test_hits_are_hit_objects():
    assert all(isinstance(h, Hit) for h in pipeline("HR")(QUERY, 3))
