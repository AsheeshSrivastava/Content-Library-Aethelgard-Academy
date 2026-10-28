# PROGRESS

Single source of truth for where rag-core stands. Update at the end of every step.
Last updated: 2026-10-28 (branch created, docs-only first commit).

## Goal

A citation-grounded RAG library for the Aethelgard content factory: approved sources in, clean text, cited passages out, with a test set that proves each change helps. Local embedding model and local reranker, chosen later by the same questions.

## Roadmap

| # | Unit | Status | Notes |
|---|---|---|---|
| 0 | Branch, AGENTS, system design, ADRs, carry-over | Done 2026-10-28 | This commit. Docs only |
| 1 | Contracts: normalize, Source, Passage, IDs, versions, read-back check | Next | ADR 0003 (Accepted). Test first |
| 2 | Source register + licence gate (carried from library3) | Planned | Port, then re-test |
| 3 | Loaders + inspection script | Planned | Order: inspection script, GitHub (merge of both), Markdown/MDX/front matter, rst, HTML, notebooks, txt. See system design |
| 4 | Chunking comparison on cleaned text | Planned | rules vs parent-child vs a LlamaIndex variant branch |
| 5 | Embedding + reranker choice (local) | Planned | Same questions, new embedder means new scores |
| 6 | Persistent store + search + lookup bands | Planned | Vectors keyed by content_id + model |
| 7 | Eval harness + question sets | Planned | Carried from library3, anchors changed |

## Decisions (see docs/adr)

- 0001 record decisions: Accepted (convention).
- 0002 fresh orphan branch and carry-over list: Accepted by the owner 2026-10-28.
- 0003 contracts: canonical text, `passage_id` and `content_id`, versions stored: Accepted 2026-10-28.
- 0004 loaders and inspection script: Accepted 2026-10-28.

## Open debts

- Fix of library3 worktree: git lists `library3` at `D:/01-Project/uncloth/...` as prunable, while the files sit under `unsloth\...\.worktrees\library3` with a `.git` file pointing at the old path. Do not prune before the owner decides; it holds the reference code and uncommitted state may exist.
- Main checkout (`layout1`) has uncommitted changes (handoff, tasks, story-bible draft). This branch does not touch them.
- Not yet read in library3: `fetch.py`, `gate.py`, `register.py`, `search.py`, `lookup.py`, `evaluate.py`. Read them when unit 2 starts.
- 11,475 pieces from 10 new sources are indexed but not embedded in the old index; that old embed is held by the owner and may never be needed.
- Python version for this branch is undecided (library3 venv is 3.12).
- Snapshots (about 250 MB) and old index files are local only and not backed up.

## Session handoff

Resumption prompt: "Read AGENTS.md, PROGRESS.md, docs/carry-over.md. Next: unit 1 (contracts, tests first)."

## Scope note (2026-10-28, owner)
Work happens in this worktree. Relevant data may be read and copied from library3 and production-rag when needed. The front end will also get a fresh start later; the old front end is not carried over.

