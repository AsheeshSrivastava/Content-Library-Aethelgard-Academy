"""The reranker pin, the ordering rule and the fake reranker (spec 15 section 3.3). No GPU needed."""
from pathlib import Path

import pytest

from ragcore.rerank import FakeReranker, RerankerSpec, load_reranker_specs, rerank

ROOT = Path(__file__).resolve().parent.parent
GOOD = {
    "repo": "BAAI/bge-reranker-v2-m3",
    "revision": "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
    "license": "Apache-2.0",
    "max_pair_tokens": 1024,
    "trust_remote_code": False,
}


def write(tmp_path, text):
    path = tmp_path / "rerankers.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def yaml_of(**changes):
    raw = {**GOOD, **changes}
    lines = ["r1:"]
    for key, value in raw.items():
        if value is not None:
            lines.append(f"  {key}: {str(value).lower() if isinstance(value, bool) else value}")
    return "\n".join(lines) + "\n"


def test_real_file_loads_and_is_pinned():
    specs = load_reranker_specs(ROOT / "configs" / "rerankers.yaml")
    spec = specs["bge-reranker-v2-m3"]
    assert isinstance(spec, RerankerSpec)
    assert spec.repo == "BAAI/bge-reranker-v2-m3"
    assert spec.revision == "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
    assert spec.max_pair_tokens == 1024


def test_valid_spec(tmp_path):
    spec = load_reranker_specs(write(tmp_path, yaml_of()))["r1"]
    assert (spec.id, spec.license, spec.max_pair_tokens) == ("r1", "Apache-2.0", 1024)


@pytest.mark.parametrize("changes, message", [
    ({"revision": "main"}, "40-character"),
    ({"revision": "953DC6F6F85A1B2DBFCA4C34A2796E7DD08D41E0"}, "40-character"),
    ({"max_pair_tokens": 0}, "positive"),
    ({"max_pair_tokens": "big"}, "positive"),
    ({"trust_remote_code": True}, "remote code"),
    ({"repo": ""}, "repo"),
    ({"license": ""}, "license"),
    ({"extra": 1}, "unknown key"),
])
def test_bad_spec_is_refused(tmp_path, changes, message):
    with pytest.raises(ValueError, match=message):
        load_reranker_specs(write(tmp_path, yaml_of(**changes)))


@pytest.mark.parametrize("missing", ["repo", "revision", "license", "max_pair_tokens", "trust_remote_code"])
def test_missing_key_is_refused(tmp_path, missing):
    with pytest.raises(ValueError, match=f"missing {missing}"):
        load_reranker_specs(write(tmp_path, yaml_of(**{missing: None})))


def test_empty_or_broken_file_is_refused(tmp_path):
    with pytest.raises(ValueError, match="mapping"):
        load_reranker_specs(write(tmp_path, ""))
    with pytest.raises(ValueError, match="not valid YAML"):
        load_reranker_specs(write(tmp_path, "a: [1\n"))


def test_fake_scores_count_query_words_in_passage():
    fake = FakeReranker()
    scores = fake.score("retry timeout function", ["retry the function", "nothing here", "Timeout, retry, function!"])
    assert scores == [2.0, 0.0, 3.0]
    assert fake.truncated == 0


def test_fake_counts_each_query_word_once():
    assert FakeReranker().score("retry retry", ["retry retry retry"]) == [1.0]


def test_rerank_orders_by_score_descending():
    assert rerank(FakeReranker(), "retry timeout", ["a", "retry", "retry timeout"]) == [2, 1, 0]


def test_rerank_ties_keep_input_order():
    assert rerank(FakeReranker(), "retry", ["retry a", "b", "retry c", "d"]) == [0, 2, 1, 3]


def test_rerank_empty_list():
    assert rerank(FakeReranker(), "retry", []) == []


def test_rerank_rejects_wrong_score_count():
    class Short:
        truncated = 0

        def score(self, query, passages):
            return [1.0]

    with pytest.raises(ValueError, match="scores"):
        rerank(Short(), "q", ["a", "b"])
