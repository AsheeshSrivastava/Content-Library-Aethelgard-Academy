# Architecture Decision Records

| # | Decision | Status |
|---|---|---|
| [0001](0001-record-architecture-decisions.md) | Record architecture decisions | Accepted |
| [0002](0002-fresh-branch-and-carry-over.md) | Fresh orphan branch `rag-core`, short carry-over list | Accepted |
| [0003](0003-contracts-canonical-text-and-ids.md) | Canonical text, exact-slice passages, two IDs | Proposed |
| [0004](0004-loaders-and-inspection-script.md) | Loader set, inspection script first, skip policy | Proposed |

## When to write an ADR
Write one when the decision is hard to reverse, had a real alternative, and someone could ask "why?". Write it before the code when you can. Copy [template.md](template.md) to the next number, keep it under a page, add a row above. Status starts as Proposed; only the owner sets Accepted. Never delete an ADR: supersede it.

## Coming up
| When | Decision |
|---|---|
| Before chunker code | Chunking parameters and the rules vs parent-child vs LlamaIndex comparison method |
| Before embeddings | Local embedding model, what is stored with each vector |
| Before the store | SQLite layout, vector keys, persistence and schema version |
| Before search | Hybrid or not, local reranker, lookup bands |
| Before eval | Question sets, anchors, how independent questions are checked |
