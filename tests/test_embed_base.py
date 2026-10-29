"""Spec 13, unit 13b: model pins, the embedder protocol, the fake embedder and the throughput probe. Tests first."""
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from ragcore.embed.base import (FakeEmbedder, ModelSpec, TokenStates, estimate_seconds, load_model_specs,  # noqa: E402
                                probe)

MODELS = Path(__file__).resolve().parent.parent / "configs" / "models.yaml"
SHA = "a" * 40

GOOD = f"""
m1:
  repo: org/model
  revision: {SHA}
  license: MIT
  dim: 8
  max_tokens: 512
  pooling: cls
  trust_remote_code: false
"""


def write(tmp_path, text):
    path = tmp_path / "models.yaml"
    path.write_text(text, encoding="utf-8")
    return path


# --- model pins -----------------------------------------------------------------------------

def test_the_real_pins_load_and_look_right():
    specs = load_model_specs(MODELS)
    assert set(specs) == {"bge-m3", "jina-v2-base-en"}
    bge, jina = specs["bge-m3"], specs["jina-v2-base-en"]
    assert (bge.pooling, bge.dim, bge.max_tokens, bge.trust_remote_code) == ("cls", 1024, 8192, False)
    assert (jina.pooling, jina.dim, jina.max_tokens, jina.trust_remote_code) == ("mean", 768, 8192, True)
    assert len(bge.revision) == 40 and len(jina.revision) == 40
    assert jina.code_repo == "jinaai/jina-bert-implementation" and len(jina.code_revision) == 40
    assert set(jina.code_sha256) == {"configuration_bert.py", "modeling_bert.py"}
    assert bge.code_repo is None


def test_a_good_minimal_file_loads(tmp_path):
    spec = load_model_specs(write(tmp_path, GOOD))["m1"]
    assert isinstance(spec, ModelSpec) and spec.id == "m1" and spec.dim == 8 and spec.code_repo is None


@pytest.mark.parametrize("old,new,message", [
    ("pooling: cls", "pooling: max", "pooling"),
    (f"revision: {SHA}", "revision: main", "revision"),
    (f"revision: {SHA}", f"revision: {SHA[:39]}", "revision"),
    ("dim: 8", "dim: 0", "dim"),
    ("max_tokens: 512", "max_tokens: -1", "max_tokens"),
    ("trust_remote_code: false", "trust_remote_code: true", "code"),
    ("license: MIT\n", "license: MIT\n  colour: blue\n", "colour"),
    ("  repo: org/model\n", "", "repo"),
    ("  license: MIT\n", "", "license"),
])
def test_bad_pins_are_refused(tmp_path, old, new, message):
    with pytest.raises(ValueError, match=message):
        load_model_specs(write(tmp_path, GOOD.replace(old, new)))


def test_remote_code_needs_a_pinned_reviewed_copy(tmp_path):
    base = GOOD.replace("trust_remote_code: false", "trust_remote_code: true")
    code = f"  code:\n    repo: org/impl\n    revision: {SHA}\n    sha256:\n      a.py: {'B' * 64}\n"
    assert load_model_specs(write(tmp_path, base + code))["m1"].code_sha256 == {"a.py": "B" * 64}
    for bad in (code.replace(SHA, "main"), code.replace("B" * 64, "xyz"), code.split("    sha256")[0],
                code.replace("      a.py: " + "B" * 64 + "\n", "")):
        with pytest.raises(ValueError, match="code"):
            load_model_specs(write(tmp_path, base + bad))


def test_code_section_without_remote_code_is_refused(tmp_path):
    code = f"  code:\n    repo: org/impl\n    revision: {SHA}\n    sha256:\n      a.py: {'B' * 64}\n"
    with pytest.raises(ValueError, match="code"):
        load_model_specs(write(tmp_path, GOOD + code))


def test_an_empty_or_odd_file_is_refused(tmp_path):
    for text in ("", "[]", "m1: 5"):
        with pytest.raises(ValueError):
            load_model_specs(write(tmp_path, text))


# --- the fake embedder ----------------------------------------------------------------------

def test_fake_vectors_are_float32_unit_length_and_deterministic():
    e = FakeEmbedder(dim=16)
    a = e.embed(["the cat sat", "dogs bark loudly"], kind="passage")
    assert a.shape == (2, 16) and a.dtype == np.float32
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0, atol=1e-5)
    assert np.array_equal(a, FakeEmbedder(dim=16).embed(["the cat sat", "dogs bark loudly"], kind="passage"))


def test_fake_output_order_and_batch_independence():
    e = FakeEmbedder()
    texts = ["alpha beta", "gamma", "delta epsilon zeta"]
    both = e.embed(texts, kind="passage")
    for i, t in enumerate(texts):
        assert np.allclose(both[i], e.embed([t], kind="passage")[0], atol=1e-6)
    assert np.allclose(e.embed(texts[::-1], kind="passage")[::-1], both, atol=1e-6)


def test_fake_similar_texts_are_closer_than_different_ones():
    e = FakeEmbedder(dim=64)
    a, b, c = e.embed(["red apple pie recipe", "apple pie recipe red", "quantum field theory"], kind="passage")
    assert a @ b > a @ c + 0.2


def test_fake_kind_is_checked_and_queries_equal_passages():
    e = FakeEmbedder()
    with pytest.raises(ValueError, match="kind"):
        e.embed(["x"], kind="document")
    assert np.array_equal(e.embed(["same text"], kind="query"), e.embed(["same text"], kind="passage"))


def test_fake_empty_inputs():
    e = FakeEmbedder(dim=8)
    assert e.embed([], kind="passage").shape == (0, 8)
    v = e.embed(["", "   "], kind="passage")
    assert v.shape == (2, 8) and not np.isnan(v).any() and not v.any()


def test_fake_fingerprint_changes_with_the_settings():
    assert FakeEmbedder(dim=8).fingerprint != FakeEmbedder(dim=16).fingerprint
    assert FakeEmbedder(dim=8).fingerprint == FakeEmbedder(dim=8).fingerprint
    assert FakeEmbedder(dim=8).dim == 8 and FakeEmbedder(dim=8).id == "fake"


def test_fake_count_tokens():
    e = FakeEmbedder()
    assert e.count_tokens("one two three") == 3 and e.count_tokens("") == 0


# --- token states (for late chunking) ---------------------------------------------------------

def test_fake_token_states_have_offsets_into_the_text():
    e = FakeEmbedder(dim=8)
    text = "Alpha, beta  gamma!"
    states = e.token_states(text)
    assert isinstance(states, TokenStates) and states.vectors.shape == (3, 8) and states.vectors.dtype == np.float32
    assert [text[s:t] for s, t in states.offsets] == ["Alpha", "beta", "gamma"]
    assert states.truncated is False


def test_fake_token_vectors_depend_on_their_neighbours():
    e = FakeEmbedder(dim=16)
    a = e.token_states("river bank flows")
    b = e.token_states("money bank loans")
    assert not np.allclose(a.vectors[1], b.vectors[1])
    assert np.allclose(e.token_states("river bank flows").vectors, a.vectors)


def test_fake_whole_text_mean_of_token_states_matches_the_normal_vector():
    e = FakeEmbedder(dim=16)
    text = "one two three four five"
    states = e.token_states(text)
    mean = states.vectors.mean(axis=0)
    mean /= np.linalg.norm(mean)
    assert np.allclose(mean, e.embed([text], kind="passage")[0], atol=1e-5)


def test_fake_token_states_of_blank_text_are_empty():
    states = FakeEmbedder(dim=8).token_states("  \n ")
    assert states.vectors.shape == (0, 8) and states.offsets == []


# --- probe and estimate -----------------------------------------------------------------------

def test_probe_counts_tokens_and_time():
    clock = iter([10.0, 12.0])
    result = probe(FakeEmbedder(), ["a b c", "d e"], clock=lambda: next(clock))
    assert result == {"texts": 2, "tokens": 5, "seconds": 2.0, "tokens_per_second": 2.5}


def test_probe_rejects_nothing_to_measure():
    with pytest.raises(ValueError, match="nothing"):
        probe(FakeEmbedder(), [])
    with pytest.raises(ValueError, match="nothing"):
        probe(FakeEmbedder(), ["", " "])


def test_probe_with_a_zero_clock_difference_does_not_divide_by_zero():
    clock = iter([5.0, 5.0])
    assert probe(FakeEmbedder(), ["a b"], clock=lambda: next(clock))["tokens_per_second"] == float("inf")


def test_estimate_seconds():
    assert estimate_seconds(1_000_000, 2_000.0) == 500.0
    with pytest.raises(ValueError):
        estimate_seconds(10, 0.0)
    with pytest.raises(ValueError):
        estimate_seconds(-1, 10.0)
