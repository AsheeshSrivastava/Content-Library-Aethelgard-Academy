"""Real-model checks for late chunking (spec 13 section 4.3). Run on purpose: `uv run --group embed pytest -m model`."""
from pathlib import Path

import numpy as np
import pytest

from ragcore.embed.late import MARGIN, late_vectors

pytestmark = pytest.mark.model

ROOT = Path(__file__).resolve().parent.parent
IDS = ["bge-m3", "jina-v2-base-en"]
SENTENCE = "Retrieval finds passages that answer a question, and ranking orders them by cosine similarity. "


@pytest.fixture(scope="module", params=IDS)
def encoder(request):
    from ragcore.embed.base import load_model_specs
    from ragcore.embed.hf import HFEncoder
    return HFEncoder(load_model_specs(ROOT / "configs" / "models.yaml")[request.param])


def test_token_spans_match_the_token_count_and_point_into_the_text(encoder):
    text = "Alpha beta\n\n    # ...or...\n\nGamma delta epsilon. " * 5
    spans = encoder.token_spans(text)
    assert len(spans) == encoder.count_tokens(text)
    assert all(0 <= s < e <= len(text) for s, e in spans) and spans == sorted(spans)


def test_token_spans_work_past_the_model_limit(encoder):
    text = SENTENCE * 1200
    assert len(encoder.token_spans(text)) > encoder.max_tokens


def test_one_passage_source_late_matches_normal_for_a_mean_pooling_model(encoder):
    if encoder.spec.pooling != "mean":
        pytest.skip("late pooling is a mean; it differs from CLS pooling by design (ADR 0007 point 30)")
    text = SENTENCE * 3
    late = late_vectors(encoder, text, [(0, len(text))]).vectors
    normal = encoder.embed([text], kind="passage")
    assert float((late * normal).sum()) >= 0.999


def test_late_vector_stays_close_to_the_normal_vector_of_the_same_passage(encoder):
    text = SENTENCE * 3 + "Zebras graze near quartz mountains in the early morning light. " + SENTENCE * 3
    start = text.index("Zebras")
    span = (start, start + len("Zebras graze near quartz mountains in the early morning light."))
    late = late_vectors(encoder, text, [span]).vectors[0]
    normal = encoder.embed([text[span[0]:span[1]]], kind="passage")[0]
    assert abs(float(np.linalg.norm(late)) - 1.0) < 1e-4
    assert float(late @ normal) > 0.5  # Same words, some added context.


def test_a_document_longer_than_the_model_limit_goes_through_in_windows(encoder):
    text = "".join(f"Section {i}: " + SENTENCE * 3 + "\n\n" for i in range(700))
    assert len(encoder.token_spans(text)) > encoder.max_tokens
    spans = [(text.index(f"Section {i}:"), text.index(f"Section {i}:") + 60) for i in (3, 350, 690)]
    result = late_vectors(encoder, text, spans)
    assert result.windows >= 2 and result.fallback_long == [] and result.fallback_empty == []
    assert result.window_tokens == encoder.max_tokens - MARGIN
    assert np.allclose(np.linalg.norm(result.vectors, axis=1), 1.0, atol=1e-3)
    assert encoder.truncated == 0  # No window was cut by the model.
