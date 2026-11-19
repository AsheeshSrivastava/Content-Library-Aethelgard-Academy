# Spec 19: `load_models`, a public way to build the pinned models

Status: Accepted by the owner 2026-11-19 (with the ADR 0012 amendment; option "move the pins" chosen). Follows ADR 0012 (amended) and spec 18. The decision itself (D1) was approved by the owner on 2026-11-19 in the LMS spec 27; this spec is the rag-core side of it.

## 1. Purpose
A caller that installed `ragcore` (the LMS) must be able to open an index and search it in the hybrid-plus-reranker mode without rag-core's repository files. Today the embedder and reranker are built by `ragcore.stage.make_embedder` and `ragcore.stage3.make_reranker`, which are internal and read `configs/*.yaml` from the source checkout; those files are not in the installed package. One public function builds the models and the matching `IndexSetup` from pins that travel inside the package.

## 2. What we know
Confirmed (read in code on 2026-11-19): the pins are in `configs/models.yaml` and `configs/rerankers.yaml`; `index_cli.real_services().setup` builds the `IndexSetup` from them; `IndexSetup.llama_index_version` is read from the installed `llama-index-core`, which the LMS does not install (extra `variants`); `llamaindex_version(config)` needs no LlamaIndex; the wheel builds from `src/ragcore` only (spec 18 section 2.6). Jina needs reviewed remote code in a local folder (`code.local_copy`), checked by SHA-256 before use (`verify_reviewed_code`).
Unknown until the build check: nothing material. The real-model path is already covered by the `model`-marked tests.

## 3. Interface (one new public name)
```python
def load_models(embedder_id: str = "jina-v2-base-en", reranker_id: str = "bge-reranker-v2-m3", *,
                local_files_only: bool = False) -> tuple[Embedder, Reranker, IndexSetup]
```
Exported from `ragcore/__init__.py` and added to `__all__` (public surface becomes 7 names, spec 18 section 3 is amended). The returned objects are the same ones `Retriever.open(path, setup, embedder, reranker)` takes. The `IndexSetup` equals the one the index command builds for the same ids: same revisions, chunker `llamaindex`, chunker version `llamaindex-1/<tokens>/1500/6000`, chunk config `1500/6000`.

## 4. Requirements
- R1. Pins move inside the package: `src/ragcore/pins/models.yaml` and `src/ragcore/pins/rerankers.yaml` (moved with `git mv`, content unchanged, so no pin is edited). `corpus-exclusions.yaml` stays in `configs/`. All code, scripts and tests that read the two files use the new path (one constant per file). The wheel contains them. Older ADRs and specs that name `configs/models.yaml` are not rewritten; spec 19 is the pointer.
- R2. `load_models` lives in a new module `ragcore/models.py`. Importing `ragcore` still imports no torch, transformers or llama_index (spec 18 test 2). Torch and transformers load inside the function.
- R3. Ids: only ids present in the bundled pins are accepted (today `jina-v2-base-en`, `bge-m3`; reranker `bge-reranker-v2-m3`). Anything else, including `fake`, raises `ValueError` naming the valid ids. Tests use the fake classes directly or patch the two HF classes; the public function never builds a fake.
- R4. The `IndexSetup` comes from one function `make_setup(embedder_spec, reranker_spec, llama_index_version)` used by both `load_models` and the index command, so the two cannot drift. `load_models` passes the pinned LlamaIndex version (the exact pin in `pyproject.toml`, `0.14.25`), because the index file records the version it was built with and the LMS never builds an index. A test fails if the constant and the `variants` pin in `pyproject.toml` differ. The index command keeps using the installed version.
- R5. Failures are named, never silent. Missing extra (torch or transformers not installed): `ImportError` saying to install `ragcore[embed]`. Missing or changed reviewed Jina code folder: the existing `RuntimeError` from `verify_reviewed_code`, unchanged. Model files unavailable: the library's own error passes through. `local_files_only=True` is passed to both model classes (no download), for hosts with no network.
- R6. Nothing else changes: no ranking code, no index schema, no `Retriever` behaviour. Existing tests pass untouched except the path constants and the surface test.
- R7. Version `0.2.0` in `pyproject.toml` and the source fallback in `__init__.py`. The git tag `v0.2.0` is made only when the owner says (commit and tag push are separate owner decisions).

## 5. Tests (written first, seen failing)
1. Surface: `__all__` equals the seven names; `load_models` exists; version matches `pyproject.toml` and is `0.2.0`.
2. `import ragcore` leaves torch, transformers and llama_index out of `sys.modules` (existing test, still passes).
3. With the two HF classes patched, `load_models()` returns the patched embedder and reranker built from the bundled specs (right repo and revision passed), and an `IndexSetup` with the right fields.
4. `local_files_only=True` reaches both constructors; the default is False.
5. Unknown embedder id, unknown reranker id, `fake`, empty string: `ValueError` naming valid ids.
6. `make_setup` parity: the setup from `load_models` equals the setup the index command's `setup` builds (installed LlamaIndex version patched to the pin).
7. LlamaIndex constant equals the `variants` pin in `pyproject.toml`.
8. Missing torch (import patched to fail): `ImportError` mentioning `ragcore[embed]`.
9. Both pin files exist inside the package, load with the existing loaders, and no `configs/models.yaml` or `configs/rerankers.yaml` remains.
10. Retriever round trip: an index built with the setup from `load_models` (fakes patched in) opens with the same setup, and a changed reranker revision is refused with `IndexStoreError`.
11. Build check (script, not pytest, as in spec 18 test 7): build the wheel, confirm both pin files are inside, install in a clean venv with only core dependencies, run tests 1, 2 and 5 there.
12. Local only (`model` marker): `load_models()` on the real models, one search on a tiny index, result order equals the direct route.

## 6. Edge cases for the chaos check
Repeated calls (each builds new models; the caller keeps one); ids with different case or spaces; no GPU (CPU, float32, as today); cache folder not writable; a pin file edited by hand (the existing validators refuse a non-commit revision).

## 7. Traceability
ADR 0012 amendment (2026-11-19) -> sections 3, R1 to R7. Spec 18 section 3 -> section 3 here. LMS spec 27 D1 and R10 -> sections 3, R4, R7.

## 8. Order of work
1. Owner's go on this spec and the ADR amendment. 2. Tests red. 3. Code. 4. Checks, review. 5. Local commit. 6. On the owner's word: push the branch, open the PR, then tag `v0.2.0` after the merge.
