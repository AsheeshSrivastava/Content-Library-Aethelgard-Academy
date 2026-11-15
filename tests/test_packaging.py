"""Unit 18 (spec 18): the installable package, its public surface, and the keyword-only retriever."""
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

import ragcore
from ragcore.index_retriever import Retriever
from ragcore.rerank import FakeReranker
from test_index_retriever import SETUP, FakeEmbedder, fill, path  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
SURFACE = {"Retriever", "Result", "IndexSetup", "open_index", "IndexStoreError", "__version__"}


def test_public_surface_is_exactly_the_promised_names():
    assert set(ragcore.__all__) == SURFACE
    for name in SURFACE:
        assert hasattr(ragcore, name)


def test_version_matches_pyproject():
    assert ragcore.__version__ == PYPROJECT["project"]["version"] == "0.1.0"


def test_importing_ragcore_does_not_pull_heavy_libraries():
    code = ("import sys, ragcore; "
            "bad = [m for m in ('torch', 'transformers', 'llama_index') if m in sys.modules]; "
            "sys.exit(1 if bad else 0)")
    done = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True,
                          env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
    assert done.returncode == 0, done.stderr


def test_project_is_a_package_with_a_build_backend_and_extras():
    assert PYPROJECT["build-system"]["build-backend"] == "hatchling.build"
    assert PYPROJECT["tool"]["uv"]["package"] is True
    assert sorted(PYPROJECT["project"]["dependencies"]) == sorted(["pyyaml>=6.0.3", "numpy>=2.0"])
    extras = PYPROJECT["project"]["optional-dependencies"]
    assert extras["variants"] == ["llama-index-core==0.14.25"]
    assert any(item.startswith("torch") for item in extras["embed"])
    assert any(item.startswith("transformers") and "<5" in item for item in extras["embed"])


def test_cuda_index_is_not_applied_to_the_default_install():
    sources = PYPROJECT["tool"]["uv"]["sources"]["torch"]
    sources = sources if isinstance(sources, list) else [sources]
    assert all("group" in s or "extra" in s for s in sources), "the CUDA index must be limited to one group or extra"


# --- keyword-only retriever (spec 18 section 4) ----------------------------------------------------------------------

class Boom:
    def __getattr__(self, name):
        raise AssertionError(f"the keyword-only retriever touched {name}")


def test_keyword_variant_needs_no_models_and_ranks_from_one(path):
    with Retriever.open(path, SETUP, None, None, variant="L") as r:
        results = r.search("bananas paragraph", k=3)
    assert results and [x.rank for x in results] == list(range(1, len(results) + 1))
    assert results[0].source_id == "alpha/b.md"


def test_keyword_variant_never_calls_the_models(path):
    with Retriever.open(path, SETUP, Boom(), Boom(), variant="L") as r:
        assert r.search("cherries", k=2)


def test_keyword_variant_respects_max_per_file(path):
    with Retriever.open(path, SETUP, None, None, variant="L") as r:
        results = r.search("paragraph", k=10, max_per_file=1)
    assert len({x.source_id for x in results}) == len(results)


def test_default_variant_is_unchanged(path):
    with Retriever.open(path, SETUP, FakeEmbedder(), FakeReranker()) as r:
        assert r.search("apples paragraph", k=2)


def test_bad_variant_and_missing_models_are_refused(path):
    with pytest.raises(ValueError):
        Retriever.open(path, SETUP, None, None, variant="X")
    with pytest.raises(ValueError):
        Retriever.open(path, SETUP, None, None)  # default HR needs both models
    with pytest.raises(ValueError):
        Retriever.open(path, SETUP, FakeEmbedder(), None, variant="HR")
