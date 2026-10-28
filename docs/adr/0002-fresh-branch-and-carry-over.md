# ADR 0002: Fresh orphan branch `rag-core`, with a short carry-over list

## Status: Accepted (2026-10-28, owner)

## Context
library3 grew by experiments. Its contracts have gaps found on 2026-10-28: text is not normalized (BOM kept, no NFC), no chunker version is stored, re-indexing deletes every vector of a source, unsupported files (notebooks, HTML, txt) are dropped without a report, IDs are only partly content-derived. The owner also wants a local embedder and reranker chosen from scratch.

## Options considered
1. Keep patching library3.
2. New branch with history copied.
3. Orphan branch with no history, carrying only chosen parts.

## Decision
Option 3, as an orphan branch in a worktree at `.worktrees/rag-core`. library3 stays untouched as the reference until rag-core beats it on the same questions. No code imports from `aethelgard.library`.

Carried over (details in docs/carry-over.md): source register and licence gate; pinned fetch with hashes and host allow-list; contract strengths (exact-slice pieces, separate chunk and vector tables); the eval harness, question sets, snapshots and result files as a measuring stick; the rst heading reader.
Taken from production-rag-with-LangChain-and-vector-databases: ADR discipline, data contract ideas, skip report and guard rail, HTML extraction, the systems-architect-mentor skill. GitHub loader: best of both.

## Why
The measuring stick is what lets us compare embedders and chunkers fairly; the contracts must be fixed before the next re-chunk and re-embed because they change offsets and hashes.

## Trade-offs accepted
Re-porting some code. Old scores do not transfer to a new embedder, so only the questions carry over. Two code lines exist for a while.
