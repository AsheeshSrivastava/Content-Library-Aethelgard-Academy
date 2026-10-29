# PROGRESS

Single source of truth for where rag-core stands. Update at the end of every step.
Last updated: 2026-10-28 (standalone repo, Entire remote, specs and ADRs drafted).

## Goal

A citation-grounded RAG library for the Aethelgard content factory: approved sources in, clean text, cited passages out, with a test set that proves each change helps. Local embedding model and local reranker, chosen later by the same questions.

## Roadmap

| # | Unit | Status | Notes |
|---|---|---|---|
| 0 | Branch, AGENTS, system design, ADRs, carry-over | Done 2026-10-28 | This commit. Docs only |
| 1 | Contracts: normalize, Source, Passage, IDs, versions, read-back check | Done 2026-10-28 | `src/ragcore/contracts.py`, 32 tests, 13 hand-made breaks of the code all caught. Python 3.12. Self-review recorded below |
| 2 | Source register, licence gate, pinned fetch (carried from library3) | Register, gate and fetch code done 2026-10-29; network proof and licence detection next | [Spec 02](docs/specs/02-register-and-gate.md), [spec 03](docs/specs/03-pinned-fetch.md) (amended 2026-10-29: pin by exact commit SHA + two-source re-fetch test). `register.py`, `gate.py`, `errors.py`, 71 new tests (103 total); 14 hand-made breaks all caught; all 36 real library3 records load and gate verdicts are sane. `fetch.py` + `webtext.py` ported with pin-by-SHA (53 offline tests; 17 of 18 hand-made breaks caught, the survivor is the symlink test, which is skipped on this PC for lack of symlink rights). Network proof test written (`tests/test_fetch_network.py`, `-m network`, owner runs) but NOT yet run. Not yet done: licence detection (ADR 0006). ADR 0005/0006 and specs 02/03 still marked Proposed/Draft until the owner confirms acceptance in the file |
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
- 0005 port register, gate and fetch with six changes: Proposed.
- 0006 licence detection as a cross-check: Proposed.
- Specs: 01 contracts (written after the code), 02 register and gate, 03 pinned fetch: Draft.

## Open debts

- Fix of library3 worktree: git lists `library3` at `D:/01-Project/uncloth/...` as prunable, while the files sit under `unsloth\...\.worktrees\library3` with a `.git` file pointing at the old path. Do not prune before the owner decides; it holds the reference code and uncommitted state may exist.
- Main checkout (`layout1`) has uncommitted changes (handoff, tasks, story-bible draft). This branch does not touch them.
- Not yet read in library3: `fetch.py`, `gate.py`, `register.py`, `search.py`, `lookup.py`, `evaluate.py`. Read them when unit 2 starts.
- 11,475 pieces from 10 new sources are indexed but not embedded in the old index; that old embed is held by the owner and may never be needed.
- Python version for this branch is undecided (library3 venv is 3.12).
- Snapshots (about 250 MB) and old index files are local only and not backed up.

## Session handoff

Resumption prompt: "Read AGENTS.md, PROGRESS.md, docs/carry-over.md. Units 1 (contracts), 2a (register, gate) and 2b (pinned fetch code) are done. Next: owner backs up the snapshots (zip ~250 MB to another drive), then runs the network proof (see header of tests/test_fetch_network.py; register data is in library3 `content/library` with sources/, licence-rules.yaml, hosts.yaml). If both hashes match, library3 snapshots are the main input. Then licence detection (ADR 0006, spec first), then the inspection script (unit 3)."

## Scope note (2026-10-28, owner)
Work happens in this worktree. Relevant data may be read and copied from library3 and production-rag when needed. The front end will also get a fresh start later; the old front end is not carried over.


## Self-review, unit 1 (2026-10-28)
Restarting review of the work: checked against ADR 0003 and the approved scope. All 32 tests pass on Python 3.12. 14 hand-made breaks of contracts.py (no NFC, no CR fix, no BOM strip, chunker version or text hash left out of passage_id, read-back checks removed, bounds and blank checks removed, forged id accepted, licence-basis check removed) were each caught by a test. One slip found and fixed: a literal BOM character had been written into the source in place of the escape, then mangled by a mutation run; the source is now ASCII-only. Not independent review.

Open debts from unit 1: Source has no title/URI fields yet (add with the first loader); the register will decide whether licence_basis=unknown must pair with licence=unknown; Span.headings is not type-checked beyond tuple use; Source version checks do not limit length.

## Repository (2026-10-28, owner)
This work moved to the standalone repo D:\01-Project\rag-core (branch main), cloned from the orphan branch rag-core with its history. Remote: Entire origin (private, rag-learning-projects/aethelgard-rag-core); Entire enabled for Claude Code, Codex, OpenCode and Antigravity (commit a9aa400). GitHub mirror is on hold until the owner creates an empty private repo. The old worktree .worktrees/rag-core in the app repo is a frozen copy; do not work there. Entire checkpoints contain transcripts: keep every repo that receives them private. Codex hooks need approval in /hooks.
