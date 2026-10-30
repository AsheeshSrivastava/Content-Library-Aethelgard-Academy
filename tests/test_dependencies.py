"""The LlamaIndex pin (ADR 0010, spec 17 section 3 rule 8): the exact version that made the comparison results."""
import tomllib
from importlib import metadata
from pathlib import Path

import pytest

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"
PINNED = "0.14.25"


def _variants() -> list[str]:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["dependency-groups"]["variants"]


def test_llama_index_core_is_pinned_exactly():
    assert _variants() == [f"llama-index-core=={PINNED}"]


def test_installed_llama_index_matches_the_pin():
    try:
        installed = metadata.version("llama-index-core")
    except metadata.PackageNotFoundError:
        pytest.skip("llama-index-core is not installed (uv sync --group variants)")
    assert installed == PINNED


def test_numpy_is_a_normal_dependency():
    project = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]["dependencies"]
    assert any(item.startswith("numpy") for item in project)
