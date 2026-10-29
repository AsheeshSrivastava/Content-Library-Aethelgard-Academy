"""Real-model checks for HFEncoder (spec 13 section 4). Run on purpose: `uv run --group embed pytest -m model`.

Needs the `embed` group, a GPU, the downloaded models, and the golden vectors from scripts/golden_vectors.py."""
import json
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.model

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = Path.home() / ".ragcore" / "golden"
IDS = ["bge-m3", "jina-v2-base-en"]


@pytest.fixture(scope="module", params=IDS)
def model_id(request):
    return request.param


@pytest.fixture(scope="module")
def specs():
    from ragcore.embed.base import load_model_specs
    return load_model_specs(ROOT / "configs" / "models.yaml")


@pytest.fixture(scope="module")
def encoder(model_id, specs):
    from ragcore.embed.hf import HFEncoder
    return HFEncoder(specs[model_id])


@pytest.fixture(scope="module")
def golden(model_id):
    path = GOLDEN / f"{model_id}.json"
    if not path.is_file():
        pytest.skip(f"{path} missing; run scripts/golden_vectors.py")
    return json.loads(path.read_text(encoding="utf-8"))


def cos(a, b):
    return (a * b).sum(axis=-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))


def test_matches_reference_vectors(encoder, golden):
    got = encoder.embed(golden["texts"], kind="passage")
    assert got.shape == (len(golden["texts"]), encoder.dim)
    assert cos(got, np.array(golden["vectors"], dtype=np.float32)).min() >= 0.999


def test_vectors_are_unit_length_float32(encoder, golden):
    got = encoder.embed(golden["texts"], kind="query")
    assert got.dtype == np.float32
    assert np.allclose(np.linalg.norm(got, axis=1), 1.0, atol=1e-4)


def test_fp16_close_to_fp32(model_id, specs, encoder, golden):
    from ragcore.embed.hf import HFEncoder
    full = HFEncoder(specs[model_id], dtype="float32")
    assert cos(encoder.embed(golden["texts"], kind="passage"), full.embed(golden["texts"], kind="passage")).min() >= 0.995


def test_batch_does_not_change_vectors(encoder, golden):
    texts = golden["texts"]
    together = encoder.embed(texts, kind="passage")
    alone = np.vstack([encoder.embed([t], kind="passage") for t in texts])
    assert cos(together, alone).min() >= 0.999


def test_same_input_same_output(encoder, golden):
    assert np.array_equal(encoder.embed(golden["texts"], kind="passage"), encoder.embed(golden["texts"], kind="passage"))


def test_long_text_is_cut_and_counted(encoder):
    before = encoder.truncated
    text = "word " * (encoder.spec.max_tokens + 500)
    out = encoder.embed([text, "short"], kind="passage")
    assert out.shape == (2, encoder.dim)
    assert encoder.truncated == before + 1


def test_token_states_offsets_point_into_the_text(encoder):
    text = "Chunking splits long documents into passages."
    states = encoder.token_states(text)
    assert states.vectors.shape == (len(states.offsets), encoder.dim)
    assert states.offsets[0][0] == 0
    assert all(0 <= s < e <= len(text) for s, e in states.offsets)
    assert all(a[1] <= b[1] for a, b in zip(states.offsets, states.offsets[1:]))


def test_token_states_refuses_over_limit(encoder):
    with pytest.raises(ValueError):
        encoder.token_states("word " * (encoder.spec.max_tokens + 10))


def test_bad_kind_refused(encoder):
    with pytest.raises(ValueError):
        encoder.embed(["x"], kind="document")


def test_empty_input(encoder):
    assert encoder.embed([], kind="passage").shape == (0, encoder.dim)
