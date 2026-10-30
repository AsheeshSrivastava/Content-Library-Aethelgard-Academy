# ADR 0009: Hybrid search with our own keyword index, and a local cross-encoder reranker

## Status: Accepted (2026-10-30, by the owner)

## Context
Stage 1 and 2 of the comparison (spec 13, report `docs/reports/retrieval-comparison-report.md`) narrowed the choice to two cells, jina-v2-base-en with `llamaindex` and bge-m3 with `merge-small`, and found the chunkers within about 6 questions of each other out of 548. Vector search alone misses exact terms (function names, flags, error codes) that matter in documentation, and the owner decided the final retrieval is hybrid search followed by the local reranker `BAAI/bge-reranker-v2-m3`. This ADR fixes how, so that stage 3 can be measured the same way as stages 1 and 2.

## Decision
1. **Keyword index:** our own BM25 (k1 1.2, b 0.75) over the passage text plus its heading path, built in memory from the same passages as the run. Tokens are lower-case words and numbers; identifiers with `_`, `.` or camelCase also add their parts; no stemming, no stop words (code and API names must stay intact).
2. **Combining:** reciprocal rank fusion (k 60) of the vector list and the keyword list, 200 candidates each. Fusion works on ranks, so the very different score scales of cosine and BM25 do not matter.
3. **Reranker:** `BAAI/bge-reranker-v2-m3` (a cross-encoder, local, fp16, pinned to a full commit hash in `configs/rerankers.yaml`, no remote code). It scores the query against each of the first 30 fused passages and orders them by that score. After it, results are collapsed to one hit per source file, as in spec 13 section 4.5.
4. **Measurement:** five variants per cell (vector only = baseline, keyword only, hybrid, vector + reranker, hybrid + reranker), the same questions and the same pre-registered paired rule as spec 13 section 5. Settings above are fixed now and not tuned.
5. Everything runs on the owner's computer. No paid service.

## Why
Rank fusion needs no weights to tune and no score calibration. Our own BM25 adds no dependency, is deterministic, and is small enough to test completely. A local cross-encoder reads question and passage together, which is the only part of the pipeline that can judge a passage rather than match vectors, and it costs only GPU time. The five variants let us see which step helps, not only whether the whole pipeline does.

## Alternatives
- **SQLite FTS5 or a search server (OpenSearch, Elasticsearch):** the right home once there is a store; the store decision ("before the store" in `docs/adr/README.md`) is later, and a server is more than 22,000 passages need. The index code is behind one function, so it can be swapped.
- **The `rank_bm25` package:** slow on large lists and fixes its own tokenising, which matters for code terms.
- **Weighted score mixing instead of rank fusion:** needs a weight tuned on the questions, which breaks the rule that nothing is tuned per cell.
- **Paid cloud reranking (as in library3):** the owner found it too expensive and asked for the local model.

## Trade-offs accepted
- The keyword index is rebuilt in memory for each evaluation (a few seconds at this size); persistence waits for the store decision.
- Reranking 30 passages per question costs GPU time per query (measured and reported, not assumed).
- The reranker sees passage text, so the run folder alone is not enough: the passages are rebuilt from the corpus, which must match the run's recorded corpus hash or the evaluation stops.
