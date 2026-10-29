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
| 3 | Loaders + inspection script | 3a inspection script done 2026-10-29 ([spec 05](docs/specs/05-inspection-script.md), `src/ragcore/inspect_sources.py`, `scripts/inspect_sources.py`): look mode and loader mode (no loaders registered yet); 38 tests, 256 in all; mutation check caught all but the symlink case (needs symlink rights, test is skipped here). First run on the 23 stored snapshots failed the gate (25%) for 8 sources only because of code/config files with no reader. Owner decision 2026-10-29: a separate `code-config` bucket outside the gate (listed by extension and known names, named in the report). Re-run: all 23 pass; the only remaining skips are 39 extensionless files and 1 .jl, all `no-reader`. 3b snapshot loader done 2026-10-29 ([spec 06](docs/specs/06-snapshot-loader.md), `src/ragcore/files.py` shared file classifier, `src/ragcore/loaders/snapshot.py`, `gate.licence_for_path`): pinned snapshot + record to `Source` objects with a plain-text stand-in reader (`plain-1`); only approved/limited records load, gate-withheld files are named; 46 new tests (302 in all); mutation check killed all 18 breaks; real-data check: all 23 snapshots load, only 4 notebooks skipped (no reader yet). 3c Markdown/MDX reader done 2026-10-29 ([spec 07](docs/specs/07-markdown-reader.md), `src/ragcore/loaders/markdown.py`, `registry.py`, `plain.py`): front matter to a title heading, MDX imports, JSX components, HTML, comments removed; code never touched; 36 reader tests incl. a 300-case random property test; mutation check 23/23 killed after 4 test gaps were closed; real data: 22 of 23 sources pass loader mode, markup matches 66,986 to 60,387 (the rest is rst); fixed a console-encoding crash in the inspection script. 349 tests in all. 3d rst reader done 2026-10-29 ([spec 08](docs/specs/08-rst-reader.md), `src/ragcore/loaders/rst.py`): headings to `#`, literal blocks and code directives to fences, directives and roles cleaned, version notes kept as `3.9: text`; 77 reader tests incl. a 400-case random property test; mutation check 33 of 34 (the survivor is equivalent); real data: leftover markup 66,977 to 3,487 over all sources (python-docs 54,466 to 400); 426 tests in all. Next: chunking spec (exact-slice check) before the HTML and notebook readers, because chunking decides the retrieval comparison | Order: chunking, notebooks, HTML. See system design |
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

## Self-review, unit 4: chunking (2026-10-29)
Restarting review of the work: spec 09 (heading-pack-1, check_chunking), `chunking.py`, `chunk_report.py`, `scripts/chunk_sources.py`. Full suite 476 passed, 2 skipped. Mutation check (34 hand-made breaks): all caught except equivalent ones (whitespace skip after a cut; paragraph-end cut branch); redundant conditions removed instead. Real data: all 23 snapshots chunk with every check passing (32,983 passages). Bugs found on the way: a fence next to text without a blank line could be cut mid-fence; a space-less token was cut at the heading space. Not independent review.
Open: llamaindex-docs has many tiny passages (median 153 chars); try a merge-small-sections variant in the comparison. Still open from earlier: front matter keys as metadata (ADR), `experiments` snapshots not backed up.

## Self-review, unit 5: HTML reader (2026-10-29)
Restarting review of the work: spec 10, `loaders/html.py` (html-1), registered for .html and .htm behind the stub wrapper. Full suite 508 passed, 2 skipped. Mutation check (25 mutants): two equivalent survivors; found and fixed one real bug (empty list item). The reader is not idempotent by design (its output is Markdown-style text), and the spec says so. Real data holds only 9 html files, all theme templates or a generated chart; the reader's job there is to not mistake them for content. Not independent review.
Open: `.overrides`/`overrides` theme folders could be excluded by a register path rule (register change, not made).

## Self-review, unit 6: notebook reader (2026-10-29)
Restarting review of the work: spec 11, `loaders/notebook.py` (ipynb-1), registered for .ipynb behind the stub wrapper. Markdown cells go through the Markdown reader; code cells become fences; outputs, raw cells and metadata are dropped. Two older tests changed (registry sample for .ipynb; the loader test now expects `bad-notebook` for `{}` instead of `no-reader`). Mutation check 19 mutants, 5 survivors closed with tests. Real data: 2 notebooks, both sources pass the gate. Not independent review.

## Self-review, unit 7: eval harness (2026-10-29)
Restarting review of the work: spec 12, `evalset.py` (questions, checks, scoring with a retriever function), `question_check.py` and `scripts/check_questions.py`, 14 question files copied unchanged to `eval/questions/`. Full suite 575 passed, 2 skipped. Mutation check 28 mutants, none surviving after added tests. Real data: 601 questions, 8 problems, all the same unmatched piece (`2026-07-28/schema`, JSON/TS schema files in the code-config bucket). Lookup bands were left out on purpose (they belong to the lookup service). Not independent review.
Next: chunker variants (fixed window, merge small sections, parent-child) behind the Chunker interface, then a run that prints a cost estimate first.

## Plan, units 8 and later: embedder and chunker comparison (2026-10-29)
Owner decided: local embedders bge-m3 and jina-embeddings-v2-base-en; chunkers fixed, merge-small, parent-child, semantic steered by each embedder, LlamaIndex (boundaries only); late chunking as an embedding mode. Written as [ADR 0007](docs/adr/0007-local-embedder-and-chunker-comparison.md) (Proposed) and [spec 13](docs/specs/13-embedder-chunker-comparison.md) (Draft, with the pre-registered decision rule and unit list 13a to 13h). Owner accepted ADR 0007 and spec 13 and agreed to the four recommendations (dependency group for LlamaIndex, torch with CUDA, full corpus, keep fixed and merge-small). Unit 13a is next. Also corrected: the question check ran against 23 snapshots, not 10 as first written in spec 12.

## Self-review, unit 13a: fixed and merge-small chunkers (2026-10-29)
Restarting review of the work: spec 13 section 4.2, `chunking.py` (fixed_spans, merge_small_spans, chunk_fixed, chunk_merge_small, `structure` switch of the check, `version` argument), `chunk_report.py` (`--chunker`, `--min-chars`). Full suite 604 passed, 2 skipped. Mutation check 22 mutants, 5 survivors closed with tests. Real data: all 23 snapshots pass the checks with both chunkers (table in spec 13 section 9a). The refactor of `chunk_source` into `_finish` leaves the rules chunker unchanged (its 79 tests still pass). Not independent review.
Next: 13b, the embedder protocol, FakeEmbedder, HFEncoder, golden and fp16 checks, throughput probe. It needs the `embed` dependency group, so the install lines will be shown to the owner first.

## Self-review, unit 13b: embedder wrapper (2026-10-29)
Restarting review of the work: spec 13 section 4, `embed/base.py` (specs, FakeEmbedder, probe), `embed/hf.py` (HFEncoder), `configs/models.yaml`, `scripts/golden_vectors.py`, new pytest marker `model`. Full suite 633 passed, 2 skipped (model tests are left out by default). Real models on the RTX 5070 Ti: 20 model tests pass for both bge-m3 and jina-v2-base-en (match reference vectors cosine >= 0.999, fp16 vs fp32 >= 0.995, batch independence, truncation count, token offsets). Mutation check on hf.py 5 mutants, 4 caught; the survivor (drop one normalisation) is equivalent because `normalise` runs again after it. bge-m3 pooling is CLS, confirmed by the reference match. Not independent review.
Next: 13c, run folder, ids, atomic write, retriever with file-level collapse, paired report.

## Self-review, unit 13c: run folder, retriever, paired report (2026-10-29)
Restarting review of the work: spec 13 sections 4.4 to 4.6 and 5, `runs.py`, `retrieve.py`, `compare.py`. 56 new tests, full suite 689 passed, 2 skipped. Mutation check about 45 mutants over three rounds. Survivors found and closed: row/vector count checks, tie order, rank just outside top 3, rank beyond depth. One real bug found this way: choosing the best 200 with `argpartition` picked an arbitrary subset when scores tie at the cut, so a run was not repeatable; it now sorts fully with ties by row position. Two equivalent survivors remain (empty-run guard, mean reciprocal rank of nothing). Not run on real data yet: that is stage 0. Not independent review.
Next: 13d, the estimate command and the stage 0 smoke (owner runs it).

## Self-review, unit 13d: estimate, run and report commands (2026-10-29)
Restarting review of the work: spec 13 section 6, `stage.py`, `scripts/run_stage.py`. 23 new tests; full suite passes. Mutation check 26 mutants over 2 rounds: weak spots found and closed were the label missing `hard_limit`, source order, a renamed question file, the `--only` filter, empty-input refusals, and bad-input tests that were passing for the wrong reason (empty question folder). Real data stage 0 smoke on `python-peps` with `fake`, `bge-m3` and `jina-v2-base-en`: all three commands work; run folders are in `~/.ragcore/runs-smoke` (not in the repo). Not independent review.
Next: 13e parent-child, then 13f semantic, 13g LlamaIndex, then the owner runs the stage 1 estimate.

## Self-review, unit 13e: parent-child chunker (2026-10-29)
Restarting review of the work: spec 13 section 4.2 and 4.5, `chunk_parent_child` and `parent_child_spans` in `chunking.py`, the `parent-child` entry in `stage.py`. Children pass `check_chunking` (exact slices, order, no overlap, hard limit, no lost text). Mutation check about 30 mutants over two rounds; two weak spots found and closed (dropped-text blocks, label missing the child size); two equivalent survivors noted in spec 13 section 9d. Not yet run on the full corpus (that is the stage 1 estimate). Not independent review.
Next: 13f semantic chunker (steered by each embedder).
