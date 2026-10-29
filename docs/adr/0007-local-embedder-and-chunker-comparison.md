# ADR 0007: Local embedder and chunker comparison design

## Status: Accepted (2026-10-29, owner)

## Context
The order in AGENTS.md is contracts, loaders, chunking comparison, then embedder and reranker choice. Contracts (specs 01 to 03), loaders (06 to 08, 10, 11), the baseline chunker (09) and the measuring stick (12) are done. The owner decided on 2026-10-29 to compare two local embedders, `BAAI/bge-m3` and `jinaai/jina-embeddings-v2-base-en`, with these chunking ideas: late chunking, semantic chunking steered by each embedder, parent-child, fixed windows, merging small sections, and a LlamaIndex splitter as a chunker-only variant (ADR 0002). Earlier library3 results (rules 73% first, parent-child 80%, hybrid-rerank 83% on 377 AI questions with a paid embedder) are the only evidence so far and used a different library and embedder.

## Decision
1. **Embedders run locally** through `transformers` (one small wrapper, pooling chosen by config: first-token for bge-m3, mean for jina). Model revisions are pinned by commit hash and recorded in every run. Torch and transformers live in an optional dependency group `embed`, so the normal tests do not need them.
2. **Search is exact and flat**: normalised float32 vectors in one `.npy` file per run, cosine by matrix product. No vector database and no approximate index until a choice is made, so no search noise mixes into the comparison.
3. **One run = one folder** under `~/.ragcore/runs/<run_id>/` (outside the repo). `run_id` is a hash of everything that can change the result: chunker id and settings, embedder id and revision, embedding mode, source list with snapshot versions, normaliser version, dtype. The folder holds the manifest, the passage list, the vectors and the scores. A run that exists is never recomputed.
4. **Late chunking is an embedding mode, not a chunker.** Spans stay exact slices from a chunker; only how their vectors are made changes (whole document through the model, then each span's token vectors averaged). This keeps ADR 0003 unchanged and lets late chunking be tested with any chunker.
5. **Parent-child needs no new contract field.** The run keeps the child to parent map (both are exact slices); search ranks children and returns each parent once.
6. **Semantic chunking is a chunker that takes an embedder.** Its version string carries the embedder and revision, so "jina steers the cuts" and "bge-m3 steers the cuts" are different, comparable chunkers.
7. **LlamaIndex** is used only to find boundaries; every node is mapped back to an exact slice of our text. Nodes that cannot be mapped are counted; more than 1% unmapped fails the variant. It lives in the optional group `variants` (owner agreed, 2026-10-29; this replaces the variant branch named in ADR 0002 for this case).
8. **Pre-registration.** The metrics, the primary question set, the ranking rule and the decision rule are written in spec 13 before any run, and not changed after seeing results.
9. **Ranking is by file.** Hits are collapsed to the best-scoring hit per source file before "first" and "top 3" are counted, so small chunks cannot win by filling the top 3 with one file, and big chunks cannot win by hiding behind it.
10. **The primary question set excludes `rag-questions.yaml`** (it was written from the same lessons as the old pieces and favours them; library3 plan rev 2 excluded it for the same reason). It is reported as a secondary set.

Scope: this is the design of the experiment. The production vector store (SQLite layout, vector keys, schema version) stays a separate decision, listed under "Coming up" in the ADR index.

## Why
Chunker, embedder and embedding mode interact; only a fixed ruler and a recorded run for every cell of the grid let the owner see which part helps. Local models cost nothing per run, so the grid can be run in full; the cost is GPU time, which is printed as an estimate before each stage.

## Trade-offs accepted
- Questions are file-level, so a chunker that finds the right paragraph inside the right file scores the same as one that finds the right file. A small quote-anchored set (AGENTS.md anchors) is a later stage, not part of this decision.
- `jina-embeddings-v2-base-en` requires `trust_remote_code`: code from its Hugging Face repo runs on the owner's PC. Mitigation: pin the commit, read the modelling files once before the first run, then run offline. Known risk: that code has broken with new `transformers` versions before; a smoke test comes first and the version can be pinned.
- English-only jina next to multilingual bge-m3: fine for this English library, noted in results.
- Brute-force search is slow at millions of vectors; at about 35,000 passages it is milliseconds.
- bge-m3 dense vectors are trained from the first token; averaging token vectors (late chunking) departs from that training. The run measures it; a poor late-chunking result for bge-m3 is a finding, not a bug.

## Reopen only if
A model cannot be loaded or reproduced, a run is not repeatable to within the stated tolerance, or the results show the file-level ruler cannot tell the variants apart.
