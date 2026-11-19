"""Spec 19 test 12: `load_models` on the real pinned models. Run on purpose with `-m model` (needs the GPU and the
cached model files; the Jina reviewed-code folder must exist)."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import ragcore
from ragcore import Retriever, open_index
from ragcore.chunking import chunk_source
from ragcore.embed.base import load_model_specs
from ragcore.embed.hf import HFEncoder
from ragcore.index_sync import sync
from ragcore.rerank import HFReranker, load_reranker_specs
from test_index_retriever import CONFIG, SOURCES, URLS

pytestmark = pytest.mark.model
PINS = Path(ragcore.__file__).parent / "pins"


@pytest.fixture(scope="module")
def loaded():
    return ragcore.load_models()


def test_real_models_match_the_direct_route(loaded):
    embedder, reranker, setup = loaded
    direct_embedder = HFEncoder(load_model_specs(PINS / "models.yaml")["jina-v2-base-en"])
    texts = ["Paragraph about bananas.", "Something about cherries."]
    a, b = embedder.embed(texts, kind="passage"), direct_embedder.embed(texts, kind="passage")
    assert float(np.min(np.sum(a * b, axis=1))) > 0.999
    direct_reranker = HFReranker(load_reranker_specs(PINS / "rerankers.yaml")["bge-reranker-v2-m3"])
    got = reranker.score("which fruit is yellow", texts)
    want = direct_reranker.score("which fruit is yellow", texts)
    assert np.allclose(got, want, atol=1e-2)
    assert setup.embedder_id == "jina-v2-base-en" and setup.reranker_id == "bge-reranker-v2-m3"


def test_search_on_a_tiny_index_with_the_real_models(loaded, tmp_path):
    embedder, reranker, setup = loaded
    tiny = replace(setup, chunker="heading-pack", chunker_version=CONFIG.version, chunk_config="70/400")
    path = tmp_path / "index.db"
    with open_index(path, tiny) as store:
        sync(SOURCES, store, lambda s: chunk_source(s, CONFIG), embedder, urls=URLS)
    with Retriever.open(path, tiny, embedder, reranker) as r:
        results = r.search("bananas", k=3)
    assert results and results[0].source_id == "alpha/b.md"
