# AGENTS.md (rag-core)

This branch is a fresh start for the RAG library. It has no history from other branches. The old work stays on `library3` as a reference until this branch beats its results on the same questions. Nothing here imports from `aethelgard.library`.

## Startup and Handoff

- Read this file, then [PROGRESS.md](PROGRESS.md) and [docs/carry-over.md](docs/carry-over.md). Read other references only as needed.
- Check `pwd`, branch, `HEAD`, and `git status`; compare them with PROGRESS.md. Do not assume a clean worktree. An unborn branch is `no commits yet`, not a hash.
- Report missing files, conflicting instructions, or unexplained state before editing. Do not reset or discard changes you do not understand. Observed Git state outranks remembered chat.

## Skill Usage

- Before substantial work, state a skill check line: "Skill: Superpowers → Brainstorming" when one is used, or "Skill check: none fits, proceeding directly".
- For each feature or boundary use the loop in [skills/systems-architect-mentor/SKILL.md](skills/systems-architect-mentor/SKILL.md): inquiry, spec, test first, no-blind-code audit, chaos check, defense.
- Suggest creating a skill when a workflow repeats.

## Research and Scope

- Define the decision or action research is meant to enable. Ask: would more information materially change the next action? If not, act.
- Keep one active task. Park related questions; do not silently change focus.
- Do not reopen a settled decision for a new tool or idea. Reopen only for real failure, changed requirements, materially new evidence, or a benefit well above the switching cost.
- Keep the order: contracts, then loaders with an inspection script, then chunking comparison, then embedder and reranker choice. A later step does not start before the earlier one is accepted.

## Rules of this branch

- Plain simple words with the owner. One step at a time.
- Commit locally. Push only to the Entire remote, and only when the owner asks. Never print keys. No paid run without a printed cost estimate first; the owner runs paid steps himself.
- Every passage is an exact slice of normalized loader output (see ADR 0003). Evaluation anchors are `(source_id, file/heading, quote)`, never passage IDs.
- Nothing is indexed until its source is approved in the register and passes the licence gate.
- Order for every unit: decision, ADR if hard to reverse, spec (docs/specs), tests, code. See docs/specs/README.md. If code would change a decision, amend the ADR or spec first.
- Test first for contracts and loaders. Mutation-check the contract tests where practical.
- Stable or LOCKED documents elsewhere in the project are not edited from this branch; write proposals as drafts.

## Tasks and Review

- A task is a deliverable with acceptance checks. At completion or agent switch update PROGRESS.md.
- After substantive work, state **"Restarting review of the work"** and do a fresh pass against the approved scope, changed files, checks and governing documents. Record findings and fixes. Self-review is not independent review.
- After checks and review, commit only the approved scope and verify the hash. Inspect staged paths first. If a check is missing, report the blocker; do not claim tests, review or a commit that did not happen.

## Mistakes made before (do not repeat)

Each line: what went wrong, then the rule. Add a line when a new mistake is found; keep it to the point.

- Mutation check by editing a source file and restoring it within the same second leaves a stale `.pyc` (same size, same mtime), so later runs fail on code that is correct. Rule: delete `__pycache__` (or run pytest with `-B`) after every mutation round before trusting the suite.

- **Synthetic tests passed, real files failed (parent-child, semantic).** Text was lost on 5 real files. Rule: before handing over any chunker, run it over every loaded source and require 0 failures (spec 13 section 9f). Check semantic chunkers with a random-unit-vector embedder that cuts a lot, at percentiles 90, 50 and 10, never with a stand-in that rarely cuts.
- **Indented code comment mistaken for a heading.** `    # note` inside a code fence in an admonition lost text when it stood alone in a passage. Rule: decide "only headings" on the raw slice before `_trim`, never after (fixed in `parent_child_spans`, `heading_pack_spans`, `fixed_spans`).
- **Loop simplified and behaviour lost.** A gap rule in parent-child became `break` and dropped text. Rule: when simplifying a loop, keep every case; add the case as a test first.
- **Ties gave different results.** Equal scores came back in a different order. Rule: sort with row position as the tie-break (`np.lexsort`), and test it.
- **Case the spec did not foresee.** Late chunking crashed on a passage shorter than a window but cut by every window boundary. Rule: test the edges of every limit: exactly the size, one over, straddling two parts, empty.
- **Equivalent mutants.** Several surviving mutants were harmless guards. Rule: if no input can tell the two versions apart, simplify the code; do not add a test for it. If a survivor can be told apart, the tests had a real gap.
- **Owner's script missed a setting.** `stage1.ps1` did not pass `--exclusions`, so the owner's estimate would have used the wrong corpus. Rule: after adding an option, update the owner's scripts, then run one full command line end to end and read the printed corpus size.
- **Changing things under a running comparison.** The corpus, question files and run-id inputs are frozen while runs exist. Rule: never add sources or edit questions mid-comparison; code edits are fine only if existing run ids stay identical, and a test pins that.
- **Telling the owner to run before decisions were settled.** Rule: send `estimate`/`run` steps only after the open choices (corpus, ADR) are answered.
- **Windows shell traps on the owner's PC.** PowerShell strips `$` and breaks inline quoting: put code in a helper `.py` or `.ps1` file. Pipe nothing into `git commit`: use `-F file`. Desktop Commander `write_file` on an existing file needs `mode: rewrite`. The agent's own Write tool writes to the container, not the PC. `git add -A` would stage the owner's `.obsidian/` folder: add by path. Never kill all `python.exe` (the owner's own processes): stop only the PID you started. A mutant that loops forever hangs pytest: stop only that process.
- **A tool result is not a check of the file.** An `edit_block` that reports "found" or a long echo does not prove the edit; read the changed lines or run the test.
- **Repeated failures.** After two failed attempts, stop, run a small diagnostic, name the likely root cause, and tell the owner which topic to revisit first.

## Documents

- PROGRESS.md: status, next action, open debts. ADRs in `docs/adr/`: consequential decisions, marked Proposed or Accepted (only the owner accepts). `docs/carry-over.md`: what came from where.
- After changing a rule, path or status, search AGENTS.md, PROGRESS.md, the ADR index and carry-over for stale wording. Before ending a session say which core documents were checked.
- `.gitignore` is not access control. Keep secrets, raw transcripts, snapshots and index files out of commits.

## Ownership

The owner has final authority over direction and standing rules. Commit permission does not permit revising settled decisions. The main agent owns the active task, verification and honest reporting.
