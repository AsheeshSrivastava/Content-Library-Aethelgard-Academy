# Spec 18: Installable package and public surface

Status: Accepted by the owner 2026-11-15 (licence Apache-2.0 chosen with the go), follows ADR 0012. Written before the code.

## 1. Purpose
Make `pip install "ragcore @ git+https://github.com/AsheeshSrivastava/Content-Library-Aethelgard-Academy@v0.1.0"` work on Python 3.12 to 3.14 with only pyyaml and numpy, and give callers one small, stable import surface.

## 2. pyproject.toml
1. `[build-system]` hatchling; wheel packages `src/ragcore`; `[tool.uv] package = true`.
2. `version = "0.1.0"`; `license` and `LICENSE` file as the owner chooses; `requires-python = ">=3.12"` (3.14 is a tested target).
3. `dependencies = ["pyyaml>=6.0.3", "numpy>=2.0"]` unchanged.
4. `[project.optional-dependencies]`: `embed = ["torch>=2.7", "transformers>=4.50,<5"]`, `variants = ["llama-index-core==0.14.25"]`. The existing `[dependency-groups]` `embed` and `variants` stay for `uv sync --group ...` and point at the same lists.
5. CUDA index: the `pytorch-cu128` source is limited to the dependency group `embed` (`group = "embed"`), so the owner's `uv sync --group embed` and all scripts work unchanged, while the extra `ragcore[embed]` gets the normal PyPI torch wheel. (uv also ignores `tool.uv.sources` of a dependency, so the LMS is unaffected either way.)
6. The wheel carries no data, eval files or scripts.

## 3. Public surface (`ragcore/__init__.py`)
Exports exactly: `Retriever`, `Result`, `IndexSetup`, `open_index`, `IndexStoreError`, `__version__` (read from package metadata, falling back to the string in `pyproject.toml` for a source checkout). `import ragcore` must not import torch, transformers or llama_index.

## 4. Retriever variant
`Retriever(store, embedder=None, reranker=None, *, variant="HR", ...)` and `Retriever.open(..., variant="HR")`.
- `HR`: needs an embedder and a reranker, as now; behaviour unchanged.
- `L`: keywords only through `make_ranker("L", ...)`; embedder and reranker may be None and are never called; scores are the keyword scores, ranks from 1; `max_per_file` works the same.
- Any other variant, or HR without an embedder or reranker, raises `ValueError` at construction.

## 5. Tests (written first)
1. Public surface: the names in section 3 exist and `ragcore.__all__` equals them.
2. `import ragcore` in a subprocess leaves torch, transformers and llama_index out of `sys.modules`.
3. `L` search on a small index with `embedder=None, reranker=None` returns the same order as `make_pipeline("L", ...)` on the same run, and never touches fakes passed in.
4. HR default unchanged: the existing retriever tests and the spec 17 check test pass untouched.
5. Bad variant / HR without models raise `ValueError`.
6. `__version__` equals the `pyproject.toml` version.
7. Build check (script, not pytest): build a wheel, install it in a clean 3.14 venv with only the core dependencies, run section 3 and test 3 there.
