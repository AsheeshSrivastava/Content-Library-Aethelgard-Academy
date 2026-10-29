# Spec 11: Jupyter notebook reader

Status: Accepted under the owner's "then the notebook reader" (2026-10-29). Builds on specs 05, 06, 07 and [ADR 0004](../adr/0004-loaders-and-inspection-script.md).

## Evidence (2026-10-29)
The 23 snapshots hold 2 `.ipynb` files: `retrieve_rerank_simple_wikipedia.ipynb` (sentence-transformers, 124 KB, nbformat 4.1, 17 code cells with 14 `stream` outputs and 1 `display_data`, 1 markdown cell, Colab metadata) and `vector_store_guide.ipynb` (llamaindex, 9 KB, nbformat 4.2, 12 markdown and 8 code cells, no outputs). Cell `source` is a list of lines in most cells and a single string in some. Most of the 124 KB is cell outputs and widget metadata, not text a learner reads.

## Purpose
`read_ipynb(raw: bytes) -> str` in `src/ragcore/loaders/notebook.py`, version `ipynb-1`, registered for `.ipynb` behind the stub wrapper. A notebook becomes Markdown text the chunker already understands: prose cells as prose, code cells as fenced code.

## Rules
1. **Parse** as UTF-8 JSON. Not JSON, or not an object with a `cells` list: `LoadError("bad-notebook")`. `nbformat` under 4: `LoadError("old-notebook")`.
2. **Markdown cells** are cleaned by the Markdown reader (spec 07, so HTML and comments go) and kept in order.
3. **Code cells** become a fenced block with the notebook language (`metadata.language_info.name`, else `kernelspec.language`; left off if it is not a plain word). The code is unchanged, including `!pip install` and `%magic` lines. The fence is longer than any backtick run in the code.
4. **Raw cells, outputs, attachments and all metadata are dropped.** Outputs change from run to run, can be huge (base64 images, progress bars) and are not the author's words.
5. Cells whose text is empty after cleaning are skipped. Cells are joined by one blank line; the text ends with one newline.
6. `source` may be a string or a list of strings; both give the same text.
7. **Deterministic**; the same notebook with different outputs gives the same text.
8. Nothing readable left returns an empty string (the registry wrapper names it `empty` or `stub`).

## Checks promised
Code survives (every code cell appears unchanged inside a fence), outputs never appear, deterministic, markdown cell words are a subsequence of the output words in order.

## Acceptance tests
A before/after test per rule; list and string `source` equivalence; a notebook with outputs and the same notebook without give the same text; bad JSON and old format named; fence growth; unknown language; mutation check; real-data check on the 2 notebooks.

## Results (2026-10-29)
- Tests: 20 in `test_notebook_reader.py`, plus the registry and loader tests updated for `.ipynb`; full suite 529 passed, 2 skipped. Mutation check (19 mutants): 5 survivors at first, each closed with a test (language priority, non-object cells, trimming blank lines around code, canonical text, BOM); re-run left none.
- Real data (loader mode): both notebooks load and both sources pass the gate. The 124 KB sentence-transformers notebook becomes 6,202 characters of input text read as 6,186 (outputs and widget metadata are the rest of the file); the llamaindex notebook 5,513 to 5,497.
