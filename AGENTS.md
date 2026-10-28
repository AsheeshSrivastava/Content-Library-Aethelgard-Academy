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
- Local only: commit locally, never push. Never print keys. No paid run without a printed cost estimate first; the owner runs paid steps himself.
- Every passage is an exact slice of normalized loader output (see ADR 0003). Evaluation anchors are `(source_id, file/heading, quote)`, never passage IDs.
- Nothing is indexed until its source is approved in the register and passes the licence gate.
- Order for every unit: decision, ADR if hard to reverse, spec (docs/specs), tests, code. See docs/specs/README.md. If code would change a decision, amend the ADR or spec first.
- Test first for contracts and loaders. Mutation-check the contract tests where practical.
- Stable or LOCKED documents elsewhere in the project are not edited from this branch; write proposals as drafts.

## Tasks and Review

- A task is a deliverable with acceptance checks. At completion or agent switch update PROGRESS.md.
- After substantive work, state **"Restarting review of the work"** and do a fresh pass against the approved scope, changed files, checks and governing documents. Record findings and fixes. Self-review is not independent review.
- After checks and review, commit only the approved scope and verify the hash. Inspect staged paths first. If a check is missing, report the blocker; do not claim tests, review or a commit that did not happen.

## Documents

- PROGRESS.md: status, next action, open debts. ADRs in `docs/adr/`: consequential decisions, marked Proposed or Accepted (only the owner accepts). `docs/carry-over.md`: what came from where.
- After changing a rule, path or status, search AGENTS.md, PROGRESS.md, the ADR index and carry-over for stale wording. Before ending a session say which core documents were checked.
- `.gitignore` is not access control. Keep secrets, raw transcripts, snapshots and index files out of commits.

## Ownership

The owner has final authority over direction and standing rules. Commit permission does not permit revising settled decisions. The main agent owns the active task, verification and honest reporting.
