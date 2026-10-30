# ADR 0010: Final chunker and embedder: jina-v2-base-en with the `llamaindex` chunker

## Status: Accepted (2026-10-30, by the owner)

## Context
The comparison (ADR 0007, spec 13; report `docs/reports/retrieval-comparison-report.md`) ended with two cells within a few questions of each other: jina-v2-base-en with `llamaindex`, and bge-m3 with `merge-small`. Retrieval is hybrid search plus the local reranker `BAAI/bge-reranker-v2-m3` (ADR 0009). The 548 file-level questions (stage 3) and the 137 passage-level questions (stage 4, spec 16, reviewed by Gemini and Codex, frozen with hash DF6A47B5...) were run on both cells with the final method (HR).

Key numbers with HR, right passage first / in top 3:
- File-level, 548 questions: jina 373 / 468, bge-m3 379 / 466.
- Passage-level, 137 questions: jina 100 / 121, bge-m3 96 / 127. Recall within 10: jina 132, bge-m3 134.
- Quotes cut by the chunking: jina 1 of 137, bge-m3 0 of 137.
- Reranker pairs cut at the 1,024-token limit (stage 4): jina 0, bge-m3 83 of 4,110.
- Time per question with HR: jina 0.13 s, bge-m3 0.17 s.

## Decision
1. The corpus is chunked with the `llamaindex` chunker (default settings from the run) and embedded with `jinaai/jina-embeddings-v2-base-en` (entry `jina-v2-base-en` in `configs/models.yaml`, revision 322d4d7e2f35, 768 dimensions, Apache-2.0). The same model embeds the questions at search time: passages and questions must use the same model and revision, so a change of model means re-embedding everything.
2. Search is hybrid (own BM25 plus vectors, rank fusion) followed by the local reranker, exactly as in ADR 0009.
3. The corpus language is English only, so the many-language strength of bge-m3 is not needed.

## Why
The two cells are a near tie, so the reasons are the small ones that all point one way: jina puts the right passage first more often on the harder passage-level questions (+4), no reranker pair is cut at the length limit, it is slightly faster, and English-only removes bge-m3's main extra. bge-m3 is ahead only in the top 3 (+6 passage-level, 2 within 10), and the pre-registered paired rule gave no clear winner (jina baseline, bge-m3 marked worse on the between-cell test).

## Alternatives
- **bge-m3 with `merge-small`:** better top 3 on the passage-level set and a hair ahead on file-level right-first. Its passages are longer, and 83 reranker pairs were cut at the limit. A longer pair limit could help it; that was not tried.
- **Keep both and decide later:** doubles storage and index work for a difference this small.
- **Other chunkers (`rules`, `fixed`, parent-child, semantic):** within 6 of 548 questions or worse in stage 1; late chunking lost everywhere.

## Trade-offs accepted
- If the corpus ever becomes multi-language, this choice must be reopened (supersede this ADR).
- The choice rests on question sets of 548 and 137; a different set could swap the order. Re-run the comparison, with the frozen sets, if the corpus changes a lot.
- A longer reranker pair limit stays an untested idea, not part of this decision.
