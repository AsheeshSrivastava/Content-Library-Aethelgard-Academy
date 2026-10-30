# Retrieval comparison report (provisional)

Status: **PROVISIONAL. Updated after every stage.** Last update: 2026-10-30, after stage 1 and stage 2. Stage 3 (hybrid search and the reranker) and the passage-level questions are not done yet; their sections below are empty on purpose.
Governing documents: spec 13 (comparison plan and the pre-registered rule), ADR 0007, spec 14 and ADR 0008 (corpus exclusions). Raw tables live in `reports/` (git-ignored); the numbers here were pasted by hand.

## 1. Why this comparison exists

The goal is a RAG library that gives the learning content factory reliable, citable facts. One poor document can damage the whole system, so the first question is the plain one: which way of cutting the sources into passages, and which local embedding model, finds the right file most often? Final retrieval will be hybrid search plus the local reranker `BAAI/bge-reranker-v2-m3` (owner, 2026-10-29), so this report also has to say which input to hand that reranker.

Earlier evidence (library3, a different corpus and a paid embedder) is not comparable and is not used here.

## 2. What was tested

**Corpus:** 22 approved sources and 1 limited, from the register, taken from pinned snapshots. 2,226 files were loaded; **342 noisy files were left out** by the exclusions file, leaving **1,884 sources**. The groups left out: most of the Python `library/` pages (30 modules kept), migration and deprecation pages, link-list and contents pages, release-note changelogs, sample test-data READMEs, one site theme template, one exact duplicate page. The snapshots on disk are untouched, so the cut can be undone by removing a rule. The corpus and the question files were frozen for the whole comparison.

**Questions:** 548 primary file-level questions (written by Codex and Gemini, checked against the files) and 50 secondary ones (`rag-questions`). "Right first" means the first result comes from an expected file. "Top 3" means one of the first three does. Question files are **file-level**: a chunker that finds the right paragraph in the right file scores the same as one that finds only the right file. 20 questions that depended on removed files were parked or pruned (`eval/questions-out-of-scope/`).

**Embedders (both local, no cost):** `BAAI/bge-m3` (CLS pooling, 1,024 dimensions) and `jinaai/jina-embeddings-v2-base-en` (mean pooling, 768 dimensions). Settings were chosen before any run and not tuned: `max_chars` 1,500, `hard_limit` 6,000, merge threshold 300, child size 500, semantic percentile 90, LlamaIndex chunk size about 375 tokens.

**Pre-registered rule (spec 13 section 5):** a cell **beats** its baseline (rules, normal mode, same embedder) if it is right-first for at least 5 more questions or in the top 3 for at least 3 more, and not worse than the baseline by more than 2 on the other measure. Otherwise it is **close** or **worse**. The winning cell overall is the beating cell with the most right-first questions.

## 2a. Why these comparisons, and how big the matrix is

Three things can change what a search finds: **where the text is cut** (the chunker), **how a passage becomes a vector** (the embedding mode) and **which model makes the vector** (the embedder). The comparison varies these three, one at a time against a baseline, and keeps the questions, the corpus and the settings fixed.

**Why each option is in the grid:**
- **rules** is the baseline: our own heading-following chunker (spec 09), the simplest thing that respects document structure. Every other option has to beat it.
- **fixed** is the control with no structure at all. If plain windows do as well, structure-aware cutting is not worth its complexity.
- **merge-small** answers a problem seen in the data: `llamaindex-docs` produced thousands of tiny passages (median 153 characters), which match questions by accident.
- **parent-child** tests precision against context: search small pieces, hand back the whole section.
- **semantic** (steered by bge-m3 and by jina) tests whether cutting where the topic changes beats cutting at headings. The owner asked for each embedder to steer, with bge-m3 as the main one and jina as the control.
- **llamaindex** is a popular framework splitter used for the cut points only (ADR 0002 and 0007). It tells us whether a ready-made splitter beats ours, without adopting the framework as the backbone.
- **late chunking** is an embedding mode the owner asked to try: embed the whole file once so each passage carries file context. It only makes sense on the chunkers that already do well, so it was run on the best three per embedder.
- **bge-m3 and jina-v2-base-en** are two local models, so no paid calls and no money cost. bge-m3 is the larger model (1,024 dimensions, CLS pooling) and jina-v2 is smaller and faster (768 dimensions, mean pooling, built for long text). Paid embedders from the earlier library3 work are not repeated.

**The matrix:**

| Stage | What varies | Cells | Status |
|---|---|---|---|
| 1 | 7 chunkers (rules, fixed, merge-small, parent-child, semantic by bge-m3, semantic by jina, llamaindex) x 2 embedders, normal mode | 14 | done |
| 2 | late mode on the best 3 chunkers per embedder | 6 | done |
| 3 | hybrid search (lexical plus vectors), then `BAAI/bge-reranker-v2-m3`, on the best two cells | 2 | not started |

The full grid would be 7 chunkers x 2 modes x 2 embedders = 28 cells. Only 20 were built (14 + 6) because late mode was tried where it had the best chance, not on weak chunkers. That is a saving in time, with the cost that some late cells were never measured. Stage 3 reuses existing runs; it adds a search method on top, not new embeddings.

## 3. The chunkers, in plain words

Every passage is an exact slice of the source text, so any result can be quoted and cited. The chunkers differ only in where they cut.

| Chunker | How it cuts | Passages | Good at | Weak at |
|---|---|---|---|---|
| **rules** (`heading-pack-1`, the baseline) | Follows the document's own headings. A heading always starts a new passage. Inside a section it packs whole paragraphs and code blocks together up to 1,500 characters. A code block is never split unless it is over the 6,000 hard limit. | 22,125 | Keeps a section's text and its heading path together; code stays whole. | Some passages are tiny (a one-line section); a very long section is cut only at paragraph edges. |
| **fixed** | Ignores structure. Cuts windows of at most 1,500 characters, preferring a paragraph end, then a line end, then a space. | 15,241 | Even sizes; almost no tiny passages. | Can cut through code blocks and sentences, and a passage may start in the middle of a section. |
| **merge-small** | Runs `rules`, then joins a passage under 300 characters with its right neighbour inside the same top-level section, while the join stays within 1,500 characters. | 19,751 | Removes the tiny passages that match by accident, keeps the section structure. | A joined passage can mix two sub-topics. |
| **llamaindex** | LlamaIndex's `MarkdownNodeParser` and then `SentenceSplitter` (about 375 tokens) decide **only where the cuts go**. Our own text is sliced, never LlamaIndex's. | 22,675 | Splits at sentence edges, so passages have a steady size (median about 640 characters). | Follows its own rules for headings and code, not ours. |
| *parent-child* | Searches small child passages (about 500 characters, one paragraph or code block) and returns the larger section (parent) the child sits in. | 92,904 children | More precise matches. | Python docs crowd the results; clearly worse for bge-m3 here. |
| *semantic* | Embeds every paragraph with one model and cuts where neighbouring paragraphs stop being alike (90th percentile of distances), always at headings. Steered by bge-m3 or by jina. | 27,764 (jina steered) | Cuts follow topic changes. | One extra embedding pass; did not beat plain `rules`. |
| *late chunking* | Not a chunker but an embedding mode: the whole file goes through the model once (windows of 8,176 tokens, overlap 512) and each passage's vector is the mean of its token vectors. | same as the chunker it is applied to | Passages carry document context. | See section 5. |

## 4. Results, stage 1 (normal mode, 14 runs)

Primary set, 548 questions. Net = difference to the baseline `rules` of the same embedder.

**bge-m3**

| Run | Right first | Top 3 | MRR | Recall@10 | Net first | Net top 3 | Verdict |
|---|---|---|---|---|---|---|---|
| rules | 310 | 417 | 0.675 | 488 | - | - | baseline |
| fixed | 314 | 416 | 0.679 | 480 | +4 | -1 | close |
| **merge-small** | **316** | **421** | 0.681 | 488 | +6 | +4 | **beats** |
| parent-child | 271 | 390 | 0.618 | 471 | -39 | -27 | worse |
| llamaindex | 301 | 411 | 0.664 | 488 | -9 | -6 | worse |
| semantic-bge-m3 | 304 | 412 | 0.669 | 488 | -6 | -5 | worse |
| semantic-jina | 299 | 407 | 0.662 | 480 | -11 | -10 | worse |

**jina-v2-base-en**

| Run | Right first | Top 3 | MRR | Recall@10 | Net first | Net top 3 | Verdict |
|---|---|---|---|---|---|---|---|
| rules | 328 | 419 | 0.691 | 475 | - | - | baseline |
| fixed | 314 | 403 | 0.668 | 469 | -14 | -16 | worse |
| merge-small | 323 | 416 | 0.686 | 477 | -5 | -3 | worse |
| parent-child | 315 | 422 | 0.680 | 483 | -13 | +3 | worse |
| **llamaindex** | **334** | 420 | 0.700 | 479 | +6 | +1 | **beats** |
| semantic-bge-m3 | 323 | 415 | 0.685 | 481 | -5 | -4 | worse |
| semantic-jina | 327 | 414 | 0.688 | 480 | -1 | -5 | worse |

Secondary set (50 questions) agrees in direction but is too small to decide anything: bge-m3 rules 18/33, merge-small 20/33, llamaindex 21/33, fixed 15/30, parent-child 19/32; jina rules 23/36, merge-small 25/36, llamaindex 22/34, fixed 17/35, parent-child 15/30.

## 5. Results, stage 2 (late chunking on the 3 best chunkers per embedder)

Picked by right-first on the primary set. Primary set, 548 questions:

| Run | Normal (first / top 3) | Late (first / top 3) | Late net first | Verdict |
|---|---|---|---|---|
| bge-m3 rules | 310 / 417 | 62 / 117 | -248 | worse |
| bge-m3 merge-small | 316 / 421 | 57 / 112 | -253 | worse |
| bge-m3 fixed | 314 / 416 | 64 / 133 | -246 | worse |
| jina rules | 328 / 419 | 305 / 402 | -23 | worse |
| jina llamaindex | 334 / 420 | 306 / 406 | -22 | worse |
| jina semantic-jina | 327 / 414 | 309 / 405 | -19 | worse |

**Diagnosis** (554 real passages from 60 files, script `late_diag`): for bge-m3, mean-pooled late vectors have cosine 0.49 to the normal vector and collapse towards one direction (mean pairwise cosine 0.86 against 0.51), so queries embedded the normal way stop matching. This is the risk named in ADR 0007 point 30, not a code fault. For jina, late vectors stay close to the normal ones (cosine 0.94) and find their own passage 97.5% of the time, so the code works; the small loss is real: our passages are already self-contained and file-wide context blurs them slightly. **Late chunking is not adopted.**

## 6. What we know and what we do not

**Settled by evidence:**
- Late chunking loses everywhere.
- Parent-child is clearly worse for bge-m3 and does not help jina on right-first.
- Semantic chunking never beat plain `rules` with either steering model.

**Not settled:**
- **The chunker is a near tie.** `rules`, `merge-small`, `llamaindex` and `fixed` differ by at most 6 questions of 548 inside one embedder. The two "beats" verdicts (bge-m3 merge-small, jina llamaindex) are each +6 first, right at the margin.
- **The embedder leans to jina on this question set** (328 to 334 against 310 to 316 right first), but cosine scales differ by model and the questions are file-level.
- **Pre-registered winner overall:** jina-v2-base-en with llamaindex (334 first), then bge-m3 with merge-small (316). The owner has not accepted a final pair.

## 7. Limits

- Questions check the file, not the passage, so chunk quality is barely measured. A quote-anchored passage-level set (about 150 questions, reviewed by Codex or Gemini) is a later stage.
- One setting each: no semantic percentile sweep, no chunk-size sweep, one late window and overlap. These were fixed before the runs and not tuned.
- Differences of 5 or 6 questions can change with a different question set.
- Eight mcp-spec questions that expected a `2026-07-28/schema` file lost that expectation when the exclusions were applied; they still have other valid files.
- Hybrid search and the reranker reordered the picture: see section 9.1. The chunker and embedder differences above shrink once the reranker is used.

## 8. Decisions on record

| Date | Decision | By |
|---|---|---|
| 2026-10-29 | Retrieve with hybrid search plus local `BAAI/bge-reranker-v2-m3` | owner |
| 2026-10-29 | bge-m3 is the main steering model for semantic chunking, jina the control | owner |
| 2026-10-30 | Corpus exclusions file accepted (ADR 0008, spec 14); the 3 optional restores declined | owner |
| 2026-10-30 | Late chunking not adopted (rule in spec 13 section 5) | pre-registered rule |
| 2026-10-30 | Stage 3 cells: jina + llamaindex and bge-m3 + merge-small | owner |
| 2026-10-30 | Stage 3 result: HR (hybrid + reranker) beats V on both cells (section 9.1) | pre-registered rule |
| open | Final chunker and embedder | owner |

## 9. To be filled in after stage 3 and the passage-level questions

### 9.1 Stage 3: hybrid search (lexical plus vectors) and bge-reranker-v2-m3
Spec 15, ADR 0009 (both accepted). Run on 2026-10-30 on the two cells chosen by the owner, all 598 questions (548 primary, 50 secondary), saved in `~/.ragcore/stage3/`. The baseline for each cell is its own V (vector only), which matches the stage 1 numbers exactly (a check that the same runs and corpus were used). Two power cuts interrupted the run; results are saved one pipeline at a time, so only the interrupted pipeline was scored again.

**Pipelines:** V vector only; L keyword only (own BM25 over heading path and passage text); H hybrid (reciprocal rank fusion, k 60, 200 candidates from each list); VR and HR = V or H, then `BAAI/bge-reranker-v2-m3` on the first 30 passages. Settings were fixed before the run and not tuned.

**Primary questions (548), right first / top 3 / MRR:**

| Pipeline | jina-v2-base-en + llamaindex | bge-m3 + merge-small |
|---|---|---|
| V | 334 / 420 / 0.700 (baseline) | 316 / 421 / 0.681 (baseline) |
| L | 287 / 385 / 0.629 (worse) | 294 / 387 / 0.636 (worse) |
| H | 332 / 425 / 0.710 (beats: +5 top 3) | 318 / 431 / 0.696 (beats: +10 top 3) |
| VR | 362 / 453 / 0.748 (beats: +28 / +33) | 376 / 457 / 0.765 (beats: +60 / +36) |
| **HR** | **373 / 468 / 0.775 (beats: +39 / +48)** | **379 / 466 / 0.779 (beats: +63 / +45)** |

**Secondary questions (50, rag-questions), right first / top 3:**

| Pipeline | jina + llamaindex | bge-m3 + merge-small |
|---|---|---|
| V | 22 / 34 | 20 / 33 |
| L | 17 / 29 (worse) | 18 / 32 (close) |
| H | 30 / 39 (beats) | 26 / 38 (beats) |
| VR | 24 / 37 (beats) | 23 / 35 (close) |
| HR | 24 / 37 (beats) | 25 / 32 (beats by the first-gain rule; top 3 is -1) |

**Cost per question:** V 0.017 to 0.019 s, L 0.002 s, H 0.013 to 0.018 s, VR 0.14 to 0.20 s, HR 0.15 to 0.20 s. Pairs cut to the 1,024-token limit: jina 0; bge-m3/merge-small 483 (VR) and 343 (HR) of about 16,000 each.

**What this says:**
- The pre-registered rule picks **HR on both cells** (most right-first among the beating variants).
- Almost all of the gain comes from the reranker. Hybrid alone adds mostly in the top 3 (+5 and +10), little in right-first. Keyword-only is clearly worse than vectors, so it is a helper, not a replacement.
- With HR the two cells end nearly level on the primary set (373 vs 379 right first; 468 vs 466 top 3). The earlier lead of jina over bge-m3 is gone. bge-m3 gains more from the reranker (+63 against +39).
- The secondary set is small (50 questions), so differences there are noisy; it agrees in direction.
- HR on bge-m3/merge-small cut 343 pairs at the token limit. Merge-small passages are longer. This is a follow-up (a longer pair limit), not a change made after seeing the results.
- The questions still check files, not passages. The ranking of the two cells under HR is a near tie and is not settled by this set.

### 9.2 Passage-level (quote-anchored) questions
_Not written yet._

### 9.3 Final choice and reasons
_Open._

## 10. Change log

- 2026-10-30: first version after stage 1 and stage 2.
- 2026-10-30: added section 2a (why each comparison, and the size of the matrix).
- 2026-10-30: stage 3 results added (section 9.1); sections 7 and 8 updated.
