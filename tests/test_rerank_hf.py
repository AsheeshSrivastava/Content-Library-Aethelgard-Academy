"""Real-model checks for HFReranker (spec 15 section 3.3). Run on purpose: `uv run --group embed pytest -m model`.

Needs the `embed` group, the downloaded model, and the golden scores from scripts/golden_rerank.py."""
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.model

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = Path.home() / ".ragcore" / "golden" / "bge-reranker-v2-m3.json"
ID = "bge-reranker-v2-m3"


@pytest.fixture(scope="module")
def spec():
    from ragcore.rerank import load_reranker_specs
    return load_reranker_specs(ROOT / "configs" / "rerankers.yaml")[ID]


@pytest.fixture(scope="module")
def reranker(spec):
    from ragcore.rerank import HFReranker
    return HFReranker(spec)


@pytest.fixture(scope="module")
def golden():
    if not GOLDEN.is_file():
        pytest.skip(f"{GOLDEN} missing; run scripts/golden_rerank.py")
    return json.loads(GOLDEN.read_text(encoding="utf-8"))


def score_pairs(model, pairs):
    return [model.score(q, [p])[0] for q, p in pairs]


def test_matches_reference_scores(reranker, golden):
    got = score_pairs(reranker, golden["pairs"])
    assert max(abs(a - b) for a, b in zip(got, golden["scores"])) <= 1e-2 * max(1.0, max(map(abs, golden["scores"])))


def test_fp16_keeps_the_order_of_50_pairs(spec, reranker, golden):
    from ragcore.rerank import HFReranker
    full = HFReranker(spec, dtype="float32")
    query = golden["pairs"][0][0]
    passages = [f"{p} (variant {i})" for i in range(10) for _, p in golden["pairs"]][:50]
    a = reranker.score(query, passages)
    b = full.score(query, passages)
    rank = lambda xs: sorted(range(len(xs)), key=lambda i: -xs[i])
    # Pairs whose fp32 scores are within 0.05 may swap; everything else must keep its place.
    top_a, top_b = rank(a), rank(b)
    swaps = [i for i in range(50) if top_a[i] != top_b[i] and abs(b[top_a[i]] - b[top_b[i]]) > 0.05]
    assert swaps == []


def test_batch_size_does_not_change_order(spec, reranker, golden):
    from ragcore.rerank import HFReranker, rerank
    small = HFReranker(spec, batch_pairs=2)
    query = golden["pairs"][2][0]
    passages = [p for _, p in golden["pairs"]] * 3
    assert rerank(small, query, passages) == rerank(reranker, query, passages)


def test_order_independent_of_input_order(reranker, golden):
    query = golden["pairs"][0][0]
    passages = [p for _, p in golden["pairs"]]
    forward = reranker.score(query, passages)
    backward = reranker.score(query, passages[::-1])[::-1]
    assert max(abs(a - b) for a, b in zip(forward, backward)) <= 0.05


def test_long_passage_is_cut_and_counted(reranker):
    before = reranker.truncated
    out = reranker.score("what is a word", ["word " * 3000, "short"])
    assert len(out) == 2
    assert reranker.truncated == before + 1


def test_empty_input(reranker):
    assert reranker.score("q", []) == []
