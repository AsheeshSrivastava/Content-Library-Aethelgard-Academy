# PROGRESS

Single source of truth for where rag-core stands. Update at the end of every step.
Last updated: 2026-10-29 (register, gate, pinned fetch done; specs 02/03 and ADRs 0005/0006 accepted).

## Goal

A citation-grounded RAG library for the Aethelgard content factory: approved sources in, clean text, cited passages out, with a test set that proves each change helps. Local embedding model and local reranker, chosen later by the same questions.

## Roadmap

| # | Unit | Status | Notes |
|---|---|---|---|
| 0 | Branch, AGENTS, system design, ADRs, carry-over | Done 2026-10-28 | This commit. Docs only |
| 1 | Contracts: normalize, Source, Passage, IDs, versions, read-back check | Done 2026-10-28 | `src/ragcore/contracts.py`, 32 tests, 13 hand-made breaks of the code all caught. Python 3.12. Self-review recorded below |
| 2 | Source register, licence gate, pinned fetch (carried from library3) | Register, gate, fetch and licence detection done 2026-10-29 | [Spec 02](docs/specs/02-register-and-gate.md), [spec 03](docs/specs/03-pinned-fetch.md) (amended 2026-10-29: pin by exact commit SHA + two-source re-fetch test). `register.py`, `gate.py`, `errors.py`, 71 new tests (103 total); 14 hand-made breaks all caught; all 36 real library3 records load and gate verdicts are sane. `fetch.py` + `webtext.py` ported with pin-by-SHA (53 offline tests; 17 of 18 hand-made breaks caught, the survivor is the symlink test, which is skipped on this PC for lack of symlink rights). Network proof (`tests/test_fetch_network.py`, `-m network`) RUN 2026-10-29 on pytest-docs (files mode) and mcp-spec: fresh fetch at the recorded SHA gave the same content hash and licence hash as the register, and the same content hash as the stored snapshot folder. Library3 snapshots are accepted as the main input. Backup: F:\backup\aethelgard-snapshots-2026-10-29.zip (68 MB). Licence detection (unit 2c) done 2026-10-29: [spec 04](docs/specs/04-licence-detection.md) accepted; `licence.py`, `licence_warning` on the record, cross-check in `apply_fetch`, warning rule in `set_status`; 62 new tests (218 total); 16 of 18 hand-made breaks caught first time, the 2 survivors were test gaps and are fixed. Run on the stored licence files of the 36 library3 sources: 21 match the declared licence, 2 not detected (python-docs PSF-2.0 is not on the list; python-peps has no licence file), 0 mismatch, 13 have no snapshot. |
| 3 | Loaders + inspection script | 3a inspection script done 2026-10-29 ([spec 05](docs/specs/05-inspection-script.md), `src/ragcore/inspect_sources.py`, `scripts/inspect_sources.py`): look mode and loader mode (no loaders registered yet); 38 tests, 256 in all; mutation check caught all but the symlink case (needs symlink rights, test is skipped here). First run on the 23 stored snapshots failed the gate (25%) for 8 sources only because of code/config files with no reader. Owner decision 2026-10-29: a separate `code-config` bucket outside the gate (listed by extension and known names, named in the report). Re-run: all 23 pass; the only remaining skips are 39 extensionless files and 1 .jl, all `no-reader`. 3b snapshot loader done 2026-10-29 ([spec 06](docs/specs/06-snapshot-loader.md), `src/ragcore/files.py` shared file classifier, `src/ragcore/loaders/snapshot.py`, `gate.licence_for_path`): pinned snapshot + record to `Source` objects with a plain-text stand-in reader (`plain-1`); only approved/limited records load, gate-withheld files are named; 46 new tests (302 in all); mutation check killed all 18 breaks; real-data check: all 23 snapshots load, only 4 notebooks skipped (no reader yet). Next: chunking spec (exact-slice check), then format readers one spec each | Order: Markdown/MDX/front matter, rst, HTML, notebooks, txt. See system design |
| 4 | Chunking comparison on cleaned text | Planned | rules vs parent-child vs a LlamaIndex variant branch |
| 5 | Embedding + reranker choice (local) | Planned | Same questions, new embedder means new scores |
| 6 | Persistent store + search + lookup bands | Planned | Vectors keyed by content_id + model |
| 7 | Eval harness + question sets | Planned | Carried from library3, anchors changed |

## Decisions (see docs/adr)

- 0001 record decisions: Accepted (convention).
- 0002 fresh orphan branch and carry-over list: Accepted by the owner 2026-10-28.
- 0003 contracts: canonical text, `passage_id` and `content_id`, versions stored: Accepted 2026-10-28.
- 0004 loaders and inspection script: Accepted 2026-10-28.
- 0005 port register, gate and fetch with six changes: Accepted 2026-10-29.
- 0006 licence detection as a cross-check: Accepted 2026-10-29.
- Specs: 01 contracts (written after the code): Draft; 02 register and gate, 03 pinned fetch: Accepted 2026-10-29.

## Open debts

- Fix of library3 worktree: git lists `library3` at `D:/01-Project/uncloth/...` as prunable, while the files sit under `unsloth\...\.worktrees\library3` with a `.git` file pointing at the old path. Do not prune before the owner decides; it holds the reference code and uncommitted state may exist.
- Main checkout (`layout1`) has uncommitted changes (handoff, tasks, story-bible draft). This branch does not touch them.
- Python version for this branch is undecided (library3 venv is 3.12).
- Snapshots are backed up (F:\backup, 2026-10-29); old index files (experiments, 780 MB) are local only and not backed up.

## Session handoff

Resumption prompt: "Read AGENTS.md, PROGRESS.md, docs/carry-over.md. Units 1 (contracts) and 2 (register, gate, pinned fetch, licence detection) are done. Snapshots are backed up and proven (see unit 2 row); they live in ~/.aethelgard/library/snapshots, register data in library3 `content/library`. Next: the inspection script (unit 3, spec first)."

## Scope note (2026-10-28, owner)
Work happens in this worktree. Relevant data may be read and copied from library3 and production-rag when needed. The front end will also get a fresh start later; the old front end is not carried over.


## Self-review, unit 1 (2026-10-28)
Restarting review of the work: checked against ADR 0003 and the approved scope. All 32 tests pass on Python 3.12. 14 hand-made breaks of contracts.py (no NFC, no CR fix, no BOM strip, chunker version or text hash left out of passage_id, read-back checks removed, bounds and blank checks removed, forged id accepted, licence-basis check removed) were each caught by a test. One slip found and fixed: a literal BOM character had been written into the source in place of the escape, then mangled by a mutation run; the source is now ASCII-only. Not independent review.

Open debts from unit 1: Source has no title/URI fields yet (add with the first loader); the register will decide whether licence_basis=unknown must pair with licence=unknown; Span.headings is not type-checked beyond tuple use; Source version checks do not limit length.

## Repository (2026-10-28, owner)
This work moved to the standalone repo D:\01-Project\rag-core (branch main), cloned from the orphan branch rag-core with its history. Remote: Entire origin (private, rag-learning-projects/aethelgard-rag-core); Entire enabled for Claude Code, Codex, OpenCode and Antigravity (commit a9aa400). GitHub mirror is on hold until the owner creates an empty private repo. The old worktree .worktrees/rag-core in the app repo is a frozen copy; do not work there. Entire checkpoints contain transcripts: keep every repo that receives them private. Codex hooks need approval in /hooks.
