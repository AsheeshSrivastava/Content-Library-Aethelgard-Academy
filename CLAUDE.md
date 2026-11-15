# CLAUDE.md

Project rules, purpose and workflow live in AGENTS.md (loaded below).
@AGENTS.md

## Checks (run from the repo root)
- Tests:  uv run pytest -q

## Order
Decision, ADR, spec, tests, code. Name the ADR and spec section before
writing a test or code (see Build order in AGENTS.md).

## Done means shown
Run every check that covers your change and paste the output
before saying a change is done. If a check fails, say so.

## Review
- Commit and push rules are in AGENTS.md. There is no CI in this repo yet; run the checks locally.
- Before asking the owner to commit or merge: /code-review, then /security-review. At most 3 review rounds.
- REVIEW.md says what counts as a real problem in this repo.
