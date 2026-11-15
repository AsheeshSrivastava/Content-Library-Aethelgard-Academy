# ADR 0012: Package rag-core as an installable library for the LMS

## Status: Accepted (owner, 2026-11-15)

## Context
The Aethelgard LMS will use rag-core as its retrieval backend and do generation only. It runs on Python 3.14 and will pin rag-core from git. Today rag-core is not installable (`package = false`, version 0.0.1), the CUDA 12.8 torch index is pinned for everyone, and `Retriever` only runs the full hybrid-plus-reranker pipeline (variant HR), which needs torch. Facts checked 2026-11-15: the core dependencies (pyyaml, numpy) install on Python 3.14.7 and `import ragcore.index_retriever` works there without importing torch (torch is imported inside the embedder and reranker constructors only).

## Options considered
1. Install straight from the repo with `pythonpath = src` tricks (no packaging).
2. Package it (build backend, extras, small public surface, tag) and let the LMS pin `ragcore @ git+https://...@v0.1.0`.
3. Publish to PyPI.

## Decision
Option 2. Build backend `hatchling` with `src/ragcore`. Core dependencies stay pyyaml and numpy. Extras: `embed` (torch, transformers<5) and `variants` (llama-index-core==0.14.25). The public surface is `Retriever`, `Result`, `IndexSetup`, `open_index`, `IndexStoreError` and `__version__`, exported from `ragcore/__init__.py`; everything else (compare, stage3, stage4, runs, scripts) is internal and not promised to callers. `Retriever` gains a `variant` argument: `"HR"` (default, unchanged) or `"L"` (keywords only, no embedder or reranker, no torch). Version 0.1.0, git tag `v0.1.0`. The CUDA torch index pin moves out of the default path. A LICENSE file (Apache-2.0, owner's choice) is added.

## Why
- The LMS needs a pinned, reproducible install and must never import experiment code.
- A keyword-only mode lets the LMS run in CPU-only or small environments (and a later Hugging Face deployment) before a GPU model is available.
- Nothing about ranking changes for HR: same code path, same results (test 10 of spec 17 keeps passing).

## Trade-offs accepted
- Exported names become a promise: removing one needs a version bump.
- `L` is lower quality than HR (stage 3 report); it is a fallback, not the recommended mode.
- Revisit if the LMS needs more than search (for example indexing from inside the LMS) or if PyPI publishing is wanted.
