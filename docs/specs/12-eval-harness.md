# Spec 12: Eval harness (questions, checks, scoring)

Status: Accepted under the owner's "then your recommended order" (2026-10-29). Builds on specs 06 and 09. Carries library3 `evaluate.py` and the 601 question files as the measuring stick ([carry-over](../carry-over.md), unit 7).

## Evidence (2026-10-29)
library3 holds 14 question files in `content/library/eval/`: 601 questions (each `q` plus `expect`, a list of path pieces) and 162 `outside` questions the library cannot answer. Written by Claude, Codex or Gemini from the pinned docs. A question counts as answered when a result's file path contains any `expect` piece. library3's scoring was tied to its own `search()` and its lookup bands.

## Purpose
`src/ragcore/evalset.py`: load the question files, check them against the loaded library, and score any retriever. No embedding or search code is in this unit; a retriever is a function, so the chunker comparison (and later the embedder and reranker) all use the same ruler.

## Questions and checks
1. **Files** live in `eval/questions/*.yaml`, copied unchanged from library3 (their header comments say who wrote them). Format: `questions: [{q, expect: [pieces]}]`, optional `outside: [question strings]`. `load_question_file` raises `ValueError` naming the file for anything else, including blank question text or blank pieces.
2. **`check_questions(sets, source_ids, broad_share=0.25)`** returns problems, empty when sound:
   - `no-match`: an `expect` piece that appears in no source id of the loaded library (the file was moved, or the piece is mistyped);
   - `broad`: a piece that appears in more than `broad_share` of all source ids (it would count nearly any result as right);
   - `unreachable`: a question none of whose pieces match anything;
   - `duplicate`: the same question text twice among answerable questions;
   - `outside-is-answerable`: an `outside` question with the same text as an answerable one.
3. A source id is `<record id>/<relative path>` (spec 06), so the pieces match against it the way they matched library3's citation file.

## Scoring
`Hit(source_id, score)`; a `Retriever` is `Callable[[str, int], list[Hit]]` (query, how many), best first.
`evaluate(questions, outside, retriever, k=3, depth=10)` returns a `Report`:
- **first**: the first hit is right; **top_k**: one of the first `k` is right (the rule from the plan: "right lesson first" and "top 3");
- **misses**: question and the sources found in the first `k`, for every question that is not in the top `k`;
- **right_scores**: for each question with a right hit within `depth`, the best score of a right hit; **outside_scores**: the best score for each outside question (0.0 when nothing comes back);
- `suggested_threshold`: midway between the weakest right score and the strongest outside score, or `None` when they overlap;
- `summary()` prints the rates with percentages, the misses, and the score lines, as library3 did, without the lookup bands (they belong to the lookup service, not the ruler).
An empty question list gives a report with zero totals and no division by zero. Per-set results (`evaluate_sets`) give one report per question file and one for all of them together.

## Real-data command
`scripts/check_questions.py --questions eval/questions --register DIR --snapshots DIR` loads every approved snapshot in the register that is present on disk, collects the source ids, runs `check_questions`, prints the problems and the counts, and exits 1 if there are any problems, 2 on bad input. Snapshots that are absent or refused are named, not hidden.

## Acceptance tests
Loader errors (each bad shape), each check with one good and one bad case, scoring on a fake retriever (first, top 3 but not first, miss, depth cut-off, outside scores, overlap and clean threshold, empty sets, ties), `evaluate_sets`, CLI exit codes; mutation check; real-data run on the 601 questions and the 23 snapshots.

## Results (2026-10-29)
- Tests: 39 in `test_evalset.py` and 7 in `test_question_check.py`; full suite 575 passed, 2 skipped. Mutation check (28 mutants): all caught after adding tests for a retriever that returns too many results, the merged report fields, and snapshots that are skipped too much.
- Real data: 14 question files copied unchanged to `eval/questions/` (601 questions, 162 outside). The check against the 10 snapshots on disk (2,226 loaded sources; 13 other register records have no snapshot, and no question needs them) found **8 problems, all one**: the piece `2026-07-28/schema` in `mcp-spec-questions-codex.yaml` matches no source, because those files are JSON or TypeScript schema files and sit in the code-config bucket (outside the gate, never loaded). Each of those 8 questions has other pieces that do match, so they can still be answered from the prose pages; the schema piece only adds an expectation that cannot be met. Nothing was edited. No broad pieces, no duplicates, no unreachable questions.

## Open questions for the owner (not blocking)
- The schema files case above: either drop that piece from the 8 questions, or later add a reader for schema files. Dropping it is cheaper; say if you want it.
- Question sets by Codex and Gemini were meant to be re-checked by Claude (files exist, none broad). This unit does that mechanically; whether each question is a good question stays a human or model judgement and is not checked here.
