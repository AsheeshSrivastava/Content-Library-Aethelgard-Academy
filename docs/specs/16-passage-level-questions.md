# Spec 16: Passage-level questions (stage 4)

Status: Accepted 2026-10-30. Written before the code.

## 1. Question
The 598 questions so far only check that the right **file** is found. Stage 3 left the two cells (jina-v2-base-en with llamaindex, bge-m3 with merge-small) almost level. Which cell, and which pipeline (V or HR), finds the right **passage**, the piece of text that actually holds the answer?

## 2. The question set
- About **150 questions**, each tied to one **quote**: a stretch of 40 to 400 characters of a source file that answers the question.
- File: `eval/passage-questions/<name>.yaml`. Each item has `q`, `source` (a source id or its file name, as `expect` today), `quote` (copied exactly from the source text), and `kind`.
- `kind` is one of: `fact` (a single stated fact), `how-to` (steps or advice), `code` (an API name, option or code example), `number` (a value, limit or default), `context` (the answer needs the section heading to make sense).
- Mix (checked by the loader): at least 25 % `code`, 10 % `number`, 15 % `context`, no `kind` above 40 %; no source above 12 questions; at least 15 of the 23 sources covered.

## 3. How the quotes are chosen (fair to every chunker)
- A script picks places in the source text at random with a fixed seed, **not** from any chunker's passages. It snaps each place to the nearest sentence or code line and rejects headings only, link lists, tables of contents and licence text.
- Sources are drawn in proportion to the square root of their size, so large sources do not take everything.
- Claude writes the question for each quote. Each question must be **hard**: it may not repeat any run of 4 words from the quote, and it must not name the file or the exact heading. The loader checks the 4-word rule by script.

## 4. Review, before any run
- The loader checks that every quote is found exactly once in its source, the mix rule, the 4-word rule and that no question is listed twice.
- Reviewers (owner's tools, 2026-10-30): Gemini 3.8 on the Pro plan is the main reviewer and reads all questions. Codex is on the free plan, so it reviews a sample of 50 questions if its limit allows; if it does not, the report says only one outside reviewer was used. A Claude model on Antigravity is not independent of me and is not counted as a reviewer. I make the prompt files. Each reviewer answers: can the question be answered from the quote alone, is it clear, is there another place in the corpus that answers it just as well. A question that fails with either reviewer is **dropped, not reworded**, and the drop is listed in the report. The owner does not review the content.
- The set is frozen (hash written into the report) before the first run.

## 5. Scoring
- A pipeline gives passages (not only files). A passage **holds** a quote if the quote lies fully inside its span in the same source. A question is **right first** if the first passage holds the quote, **top 3** if one of the first three does.
- Also reported: **quote cut**, the share of questions where the quote is split across two passages of a cell's run, which shows chunking damage directly.
- Compared: V and HR on each cell (4 combinations), with the same paired rule as spec 13 section 5 (beats means +5 first, or +3 top 3, and not worse than 2 on the other). With 150 questions a small gap is not proof; the report says so.

## 6. Units, in order (tests first, mutation check, commit each)
1. **16a** question loader and checks (quote found once, mix, 4-word rule, duplicates).
2. **16b** random quote picker (seeded, repeatable) that prints candidates for me to write questions for.
3. **16c** passage-level pipeline output and scorer (quote held, cut rate), reusing `make_pipeline`.
4. **16d** the `stage4` command (`estimate`, `run`, `report`) and `scripts/stage4.ps1`.
5. **16e** I write the questions; reviewer prompt files; reviews; freeze; owner runs; report section 9.2.

## 7. Not tuned
Nothing about the pipelines, chunkers or models changes in this stage. If the set shows a problem, it goes into the report as a follow-up.

## 8. Risks
| Risk | Check |
|---|---|
| Questions favour one chunker | quote places picked at random in the source text, not from passages |
| Questions too easy (copy the quote) | 4-word rule by script |
| Quote not in the source after cleaning | exact match against the loaded source text, found exactly once |
| Wrong or unclear question | one or two outside reviewers; failures dropped, not edited after results |
| 150 questions too few to separate near-equal cells | the report says "tie" when gaps are under the rule |
