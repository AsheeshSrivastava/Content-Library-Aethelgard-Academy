# Spec 15: Hybrid search and reranking (stage 3)

Status: Accepted 2026-10-30, follows ADR 0009 (Accepted). Written before the code.

## 1. Question
On the two cells chosen from stages 1 and 2 (**jina-v2-base-en with llamaindex** and **bge-m3 with merge-small**), does a keyword index, rank fusion and the local reranker `BAAI/bge-reranker-v2-m3` find the right file first more often than vector search alone, and which step is responsible?

## 2. The grid
| Variant | What it does |
|---|---|
| **V** (baseline) | Vector search only, as in spec 13 section 4.5 |
| **L** | Keyword search (BM25) only |
| **H** | Reciprocal rank fusion of V and L |
| **VR** | V, then the reranker on its first 30 passages |
| **HR** | H, then the reranker on its first 30 passages (the owner's chosen pipeline) |

5 variants x 2 cells = 10 evaluations. Vector runs already exist; no new embeddings are made.

## 3. Components
### 3.1 Keyword index (`lexical.py`)
- Text of a passage for the index: its heading path joined with " > ", a line break, then the passage text. Rebuilt from the corpus and the run's rows (`start`, `end`); the evaluation stops if the corpus hash of the loaded sources differs from the one in the run manifest.
- Tokens: `[A-Za-z0-9]+` lower-cased. A word that has `_` or `.` inside, or a camelCase change, also yields its parts (`load_snapshot` gives `load_snapshot`, `load`, `snapshot`; `HFEncoder` gives `hfencoder`, `hf`, `encoder`). No stemming, no stop words.
- BM25 with k1 1.2 and b 0.75, document length in tokens. Query terms are used once each. A query with no tokens returns an empty list.
- Results: the first 200 passages by score, ties by row position (repeatable).
### 3.2 Fusion (`fusion.py`)
Reciprocal rank fusion: score(passage) = sum over lists of 1 / (60 + rank), ranks from 1. Lists: vector top 200 and keyword top 200 passages. Equal scores can only come from the same ranks in the same lists, so they go to the lower row position. If one list is empty, the other passes through unchanged.
### 3.3 Reranker (`rerank.py`)
- Pinned in `configs/rerankers.yaml` (repo `BAAI/bge-reranker-v2-m3`, full commit hash, licence, no remote code). `AutoModelForSequenceClassification`, fp16, pair input (query, passage), at most 1,024 tokens per pair with truncation of the passage and a count of truncated pairs. Score = the raw logit; order is descending score, ties by the fused rank.
- Candidates: the first 30 passages of the list it follows (V or H). Passages beyond 30 are dropped, since the result is collapsed to files anyway.
- `FakeReranker` (word overlap) for tests.
- Golden check (`-m model`, needs the `embed` group): scores for 5 fixed pairs match the reference implementation (`sentence-transformers` `CrossEncoder` with an identity activation) to 1e-2, fp16 against fp32 keeps the same order on 50 pairs, batch size does not change order.
### 3.4 Pipeline and the command (`retrieve.py`, `stage.py`)
Each variant is a retriever `(query, n) -> [Hit]` for the spec 12 harness: candidates as above, then collapse to one hit per source file keeping the best (or best reranked) passage, first `n` returned. A new command `stage3` (`scripts/stage3.ps1`: `estimate`, `run`, `report`) loads each cell's existing run, builds the keyword index once, evaluates the variants and prints the paired tables with the baseline V, per question set, plus the average seconds per question for each variant. `estimate` prints the number of pairs to rerank (questions x 30 x 2 reranked variants x 2 cells) and a timed sample.

## 4. Decision rule (fixed before any run)
- Same rule as spec 13 section 5, with **V of the same cell as baseline**: a variant **beats** V if it is right-first for at least 5 more questions or in the top 3 for at least 3 more and not worse than V by more than 2 on the other measure; otherwise **close** or **worse**.
- The best pipeline per cell is the beating variant with the most right-first questions; ties go to the simpler pipeline (V, L, H, VR, HR in that order), then the faster one.
- If HR does not beat V on a cell, that is reported as it is. The owner decided on HR as the target; the report then says what it costs and what it gives, and the owner decides.
- Not tuned after seeing results: BM25 parameters, tokeniser, fusion k, candidate depth 30, reranker length. Ideas for changes go into the report as follow-ups, not into a second try on the same questions.

## 5. Failure modes and checks
| Risk | Check |
|---|---|
| Corpus differs from the run | stop with a message naming the hash mismatch |
| Empty query, or a query of symbols only | keyword list empty, fusion passes the vector list; no crash |
| Fewer than 30 candidates | rerank what there is |
| Candidate longer than the pair limit | truncated and counted in the report |
| Non-repeatable order | all sorts end in row position; a test runs each variant twice |
| Reranker out of memory | batch size drops (documented value 16 pairs), same scores |
| Reranker scores change the file of an answer to one the question does not expect | visible in the miss lists of the report, not hidden |

## 6. Units, in order (tests first, mutation check, commit each)
1. **15a** `lexical.py`: tokeniser and BM25 (with a hand-computed example).
2. **15b** `fusion.py`.
3. **15c** `rerank.py` and the model pin (network needed once to read the commit hash and download the model; the golden check is run by the owner or on the GPU).
4. **15d** pipelines in `retrieve.py`, `stage3` command, `scripts/stage3.ps1`.
5. **15e** the run on the two cells and the update of the report in `docs/reports/retrieval-comparison-report.md` section 9.1.

## 7. Not in this spec
A persistent keyword store, the "not found" threshold, the quote-anchored question set, query rewriting, anything paid.
