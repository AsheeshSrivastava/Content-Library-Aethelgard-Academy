"""Spec 19: the public `load_models` and the pins that travel inside the package."""
import argparse
import os
import re
import subprocess
import sys
import tomllib
from importlib import resources
from pathlib import Path

import pytest

import ragcore
from ragcore import IndexSetup, IndexStoreError, Retriever, open_index
from ragcore.chunking import ChunkConfig
from ragcore.embed.base import load_model_specs
from ragcore.llamaindex_chunker import llamaindex_version
from ragcore.rerank import FakeReranker, load_reranker_specs
from test_index_retriever import FakeEmbedder

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


class SpyEncoder:
    calls: list = []

    def __init__(self, spec, **kwargs):
        self.spec, self.kwargs = spec, kwargs
        SpyEncoder.calls.append(("embedder", spec.id, kwargs))


class SpyReranker:
    calls: list = []

    def __init__(self, spec, **kwargs):
        self.spec, self.kwargs = spec, kwargs
        SpyReranker.calls.append(("reranker", spec.id, kwargs))


@pytest.fixture(autouse=True)
def patched_models(monkeypatch):
    SpyEncoder.calls, SpyReranker.calls = [], []
    monkeypatch.setattr("ragcore.embed.hf.HFEncoder", SpyEncoder)
    monkeypatch.setattr("ragcore.rerank.HFReranker", SpyReranker)


def pin_dir() -> Path:
    return Path(str(resources.files("ragcore") / "pins"))


# --- 1 surface and version ---------------------------------------------------------------------------------------

def test_surface_has_seven_names_including_load_models():
    assert set(ragcore.__all__) == {"Retriever", "Result", "IndexSetup", "open_index", "IndexStoreError",
                                    "load_models", "__version__"}
    assert callable(ragcore.load_models)


def test_version_is_0_2_0_in_both_places():
    assert PYPROJECT["project"]["version"] == ragcore.__version__ == "0.2.0"


# --- 3 and 4: what load_models builds -----------------------------------------------------------------------------

def test_defaults_build_the_pinned_models_and_setup():
    embedder, reranker, setup = ragcore.load_models()
    emb = load_model_specs(pin_dir() / "models.yaml")["jina-v2-base-en"]
    rer = load_reranker_specs(pin_dir() / "rerankers.yaml")["bge-reranker-v2-m3"]
    assert isinstance(embedder, SpyEncoder) and embedder.spec == emb
    assert isinstance(reranker, SpyReranker) and reranker.spec == rer
    assert isinstance(setup, IndexSetup)
    assert setup.embedder_id == "jina-v2-base-en"
    assert setup.embedder_revision == emb.revision + "+" + emb.code_revision
    assert setup.reranker_id == "bge-reranker-v2-m3" and setup.reranker_revision == rer.revision
    assert setup.chunker == "llamaindex" and setup.chunk_config == "1500/6000"
    assert setup.chunker_version == llamaindex_version(ChunkConfig())


def test_other_embedder_id_is_used():
    embedder, _, setup = ragcore.load_models("bge-m3", "bge-reranker-v2-m3")
    assert embedder.spec.id == "bge-m3" and setup.embedder_id == "bge-m3"


def test_local_files_only_reaches_both_models():
    ragcore.load_models()
    ragcore.load_models(local_files_only=True)
    flags = [c[2].get("local_files_only", False) for c in SpyEncoder.calls + SpyReranker.calls]
    assert flags == [False, True, False, True]


# --- 5 bad ids ----------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("embedder_id,reranker_id", [
    ("nope", "bge-reranker-v2-m3"), ("fake", "bge-reranker-v2-m3"), ("", "bge-reranker-v2-m3"),
    ("Jina-V2-Base-En", "bge-reranker-v2-m3"), ("jina-v2-base-en", "nope"), ("jina-v2-base-en", "fake"),
    ("jina-v2-base-en", " bge-reranker-v2-m3")])
def test_unknown_ids_are_refused_and_name_the_valid_ones(embedder_id, reranker_id):
    with pytest.raises(ValueError, match=r"jina-v2-base-en|bge-reranker-v2-m3"):
        ragcore.load_models(embedder_id, reranker_id)
    assert SpyEncoder.calls == [] and SpyReranker.calls == []


# --- 6 and 7: the setup cannot drift -------------------------------------------------------------------------------

def test_setup_equals_the_one_the_index_command_builds(monkeypatch):
    from ragcore import index_cli, models
    monkeypatch.setattr(index_cli.metadata, "version", lambda name: models.LLAMA_INDEX_PIN)
    args = argparse.Namespace(models=str(pin_dir() / "models.yaml"), rerankers=str(pin_dir() / "rerankers.yaml"),
                              embedder="jina-v2-base-en", reranker="bge-reranker-v2-m3")
    assert index_cli.real_services().setup(args) == ragcore.load_models()[2]


def test_llama_index_constant_equals_the_pyproject_pin():
    from ragcore.models import LLAMA_INDEX_PIN
    assert f"llama-index-core=={LLAMA_INDEX_PIN}" in PYPROJECT["project"]["optional-dependencies"]["variants"]


# --- 8 missing extra ------------------------------------------------------------------------------------------------

def test_missing_torch_says_to_install_the_embed_extra(monkeypatch):
    def boom(spec, **kwargs):
        raise ModuleNotFoundError("No module named 'torch'", name="torch")
    monkeypatch.setattr("ragcore.embed.hf.HFEncoder", boom)
    with pytest.raises(ImportError, match=r"ragcore\[embed\]"):
        ragcore.load_models()


def test_reviewed_code_error_is_not_hidden(monkeypatch):
    def boom(spec, **kwargs):
        raise RuntimeError("reviewed code file is missing")
    monkeypatch.setattr("ragcore.embed.hf.HFEncoder", boom)
    with pytest.raises(RuntimeError, match="reviewed code"):
        ragcore.load_models()


# --- 9 pins inside the package ---------------------------------------------------------------------------------------

def test_pins_live_in_the_package_and_not_in_configs():
    assert (pin_dir() / "models.yaml").is_file() and (pin_dir() / "rerankers.yaml").is_file()
    assert not (ROOT / "configs" / "models.yaml").exists()
    assert not (ROOT / "configs" / "rerankers.yaml").exists()
    assert (ROOT / "configs" / "corpus-exclusions.yaml").is_file()
    assert set(load_model_specs(pin_dir() / "models.yaml")) == {"bge-m3", "jina-v2-base-en"}
    assert set(load_reranker_specs(pin_dir() / "rerankers.yaml")) == {"bge-reranker-v2-m3"}


def test_no_code_still_points_at_the_old_pin_paths():
    pattern = re.compile(r"configs\W+(models|rerankers)\.yaml")
    offenders = [str(p.relative_to(ROOT)) for folder in ("src", "scripts", "tests") for p in (ROOT / folder).rglob("*.py")
                 if p.name != Path(__file__).name and pattern.search(p.read_text(encoding="utf-8"))]
    assert offenders == []


# --- 10 round trip ---------------------------------------------------------------------------------------------------

def test_index_made_with_the_setup_reopens_and_a_changed_pin_is_refused(tmp_path):
    from dataclasses import replace
    _, _, setup = ragcore.load_models()
    path = tmp_path / "index.db"
    open_index(path, setup).close()
    with Retriever.open(path, setup, FakeEmbedder(), FakeReranker()) as r:
        assert r.search("bananas", k=1) == []
    with pytest.raises(IndexStoreError):
        open_index(path, replace(setup, reranker_revision="0" * 40), read_only=True)
    with pytest.raises(IndexStoreError):
        open_index(path, replace(setup, embedder_revision="0" * 40), read_only=True)


# --- import weight ---------------------------------------------------------------------------------------------------

def test_importing_ragcore_still_pulls_no_heavy_libraries():
    code = ("import sys, ragcore; "
            "bad = [m for m in ('torch', 'transformers', 'llama_index') if m in sys.modules]; sys.exit(1 if bad else 0)")
    done = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True,
                          env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
    assert done.returncode == 0, done.stderr
