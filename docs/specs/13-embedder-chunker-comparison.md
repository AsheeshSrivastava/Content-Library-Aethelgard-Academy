# Spec 13: Embedder and chunker comparison (plan, pre-registration, unit list)

Status: Accepted (2026-10-29, owner, with ADR 0007). Pre-registration in section 5 is now frozen. Builds on specs 09 (chunking) and 12 (eval harness). Nothing here is built yet; this spec is what gets built, in the order of the last section.

## 1. Question
Which combination of chunker, embedding mode and local embedder puts the right file first, and in the top 3, most often on our questions, and is the gain worth the extra code? The two embedders are `BAAI/bge-m3` (MIT, 1024 dimensions, 8192 tokens, multilingual, also gives sparse and multi-vector output) and `jinaai/jina-embeddings-v2-base-en` (Apache-2.0, 137M parameters, 8192 tokens through ALiBi, English only, mean pooling, needs `trust_remote_code`). Hardware: RTX 5070 Ti 16 GB, 20 cores, 64 GB RAM, 200+ GB free.

## 2. The grid
| Axis | Values |
|---|---|
| Chunker | **rules** (`heading-pack-1`, done); **fixed** (window of `max_chars`, cut at whitespace, no structure); **merge-small** (rules, then neighbours under 300 characters joined up to `max_chars`); **parent-child** (children = one paragraph or code block, up to 500 characters; parent = the rules passage or section); **semantic-jina** and **semantic-bge** (cuts where adjacent blocks stop being similar, steered by that embedder); **llamaindex** (LlamaIndex splitter for boundaries only) |
| Embedding mode | **normal** (each passage alone) or **late** (whole document through the model once, each passage = mean of its token vectors) |
| Embedder | bge-m3, jina-v2-base-en |

Stage 1 runs every chunker with the **normal** mode and both embedders: 7 chunkers x 2 = 14 runs. Stage 2 runs the **late** mode on the 3 best chunkers per embedder (6 runs). Stage 3 (separate spec, not planned here): hybrid with a lexical index, then the local reranker `BAAI/bge-reranker-v2-m3` on the best two cells. In total at most 20 runs, all local, no money cost.

"semantic-bge" and "semantic-jina" differ only in who decides the cuts. Each is also embedded with both embedders, so the table shows the cross cells (jina cuts, bge embeds, and so on).

## 3. Corpus and questions
- **Corpus**: every approved source with a snapshot on disk (23 snapshots, 2,226 loaded sources at the time of writing), loaded by the spec 06 loader and the readers of specs 07, 08, 10 and 11. The corpus is fixed once: the run manifest stores the source ids with their snapshot versions, and every run uses the same list. Full corpus, not only AI sources: distractors from the Python docs are part of real use, and local embedding costs no money.
- **Primary question set**: the 13 files in `eval/questions/` except `rag-questions.yaml`. **Secondary**: `rag-questions.yaml` alone. The 8 mcp-spec questions with the unmatched `2026-07-28/schema` piece stay in (their other pieces match); the result is also printed without them.
- Question files are frozen by their SHA-256, stored in the manifest. Editing a question starts a new set.

## 4. Components and interfaces (what gets built)

### 4.1 Embedder wrapper (`embed/encoder.py`)
```
class Embedder(Protocol):
    id: str            # "bge-m3"
    fingerprint: str   # model id + revision + pooling + dtype + max_tokens
    dim: int
    def embed(self, texts: list[str], *, kind: "passage" | "query") -> np.ndarray   # (n, dim), L2-normalised, float32
    def token_states(self, text: str) -> TokenStates   # for late chunking: (n_tokens, dim) and char offsets
```
- `HFEncoder(model_id, revision, pooling, trust_remote_code, max_tokens, dtype)` implements it with `AutoTokenizer` (fast, offsets) and `AutoModel`. bge-m3: pooling `cls`, no instruction text. jina-v2: pooling `mean` with the attention mask, no instruction text.
- `FakeEmbedder` (hashed bag of words, deterministic, no model) for every unit test. `MOCK_EMBED=1` selects it in the command line tools.
- Padding: batches are sorted by length and every text is truncated at `max_tokens`; the count of truncated passages is stored in the manifest. Output order always equals input order (tested).
- **Golden check** (optional test, needs the `embed` group): on 5 fixed texts the wrapper's vectors must match the reference library (`sentence-transformers` for both, which applies each model's own pooling) at cosine >= 0.999. This catches a wrong pooling choice before 20 runs are wasted on it.
- **fp16 check**: on 200 passages the fp16 and fp32 vectors must agree at cosine >= 0.995 (otherwise the run uses fp32). The chosen dtype is part of `fingerprint`.

### 4.2 Chunkers (`chunking.py` and `chunkers/`)
All return spans that are exact slices (ADR 0003); `check_chunking` stays the gate. Two switches are added to `check_chunking`: `structure=False` skips the "heading line inside the passage" check for chunkers that do not follow headings (fixed, merge-small, semantic across headings if ever allowed), and everything else (exact slice, order, no overlap, hard limit, coverage, ids) stays mandatory for all.
- **fixed**: windows of at most `max_chars`, cut at the last paragraph end, else line end, else space in the window (the cut rules of spec 09 without sections). Heading path = the heading in force at the passage start.
- **merge-small**: runs rules, then joins neighbours inside the same top-level section when the left one is under `min_chars` (300) and the join stays within `max_chars`. Heading path = the first passage's.
- **parent-child**: children from the existing block scanner (paragraph or fenced code block, cut again at 500 characters by the spec 09 hard-cut rules); the parent of a child is the rules passage that contains it. The parent map is stored in the run folder, not in `Passage`.
- **semantic**: inside each rules section, take the blocks (code blocks never split), embed each block, take the cosine distance between neighbours, cut where it exceeds the **90th percentile of that source's distances** (chosen now, not tuned), always cut at headings, then apply `max_chars` and `hard_limit` as in spec 09. The chunker version holds the embedder fingerprint and the percentile. Block vectors are cached by block hash.
- **llamaindex**: run `MarkdownNodeParser` then `SentenceSplitter(chunk_size=~375 tokens)` (setting chosen now); map each node to `[start, end)` in our text by its character indexes, falling back to a forward search of the node text from the previous end; unmappable nodes are counted and, over 1% of nodes, the variant fails with a clear message. The mapping is tested on Markdown, rst-derived and code-heavy text.

### 4.3 Late chunking (`embed/late.py`)
For each source text: tokenise with offsets; if it fits `max_tokens`, one forward pass; if not, windows of `W` tokens (default 8192, 4096 if memory is short) with a stride of `W - 512` so neighbouring windows share 512 tokens. A passage is pooled from the window in which it lies fully and has the most context on both sides (largest minimum distance to a window edge); a passage longer than a window falls back to normal mode and is counted. The pooled vector is the mean of last-layer token vectors whose character offsets lie inside the span (special tokens excluded), then L2-normalised. Queries are always embedded normally.
Checks: with a window large enough for the whole text, a one-passage source gives the same vector as normal mode to 1e-4 for mean-pooling models; pooling is independent of batch order; passages in different windows get vectors from the right window (tested with a fake model that returns the window index).

### 4.4 Run folder and ids (`runs.py`)
`~/.ragcore/runs/<run_id>/` holds `manifest.json` (all inputs, library versions, timings, counts of truncated and fallback passages), `passages.jsonl` (passage id, source id, start, end, headings, parent id if any), `vectors.npy` (float32), `scores.json`. `run_id` is a SHA-256 over the canonical JSON of the inputs, shortened. A run is written to a temp folder and renamed when complete, so an interrupted run leaves nothing half-readable. Loading checks that the vector count equals the passage count and the manifest hash matches.

### 4.5 Retriever (`retrieve.py`)
`retriever = load_run(run_id)` returns a function `(query, n) -> [Hit]` for the spec 12 harness: embed the query (kind `query`), cosine against all vectors, take the top `n_raw` passages (200), map child to parent where the run has parents (keep the best child score per parent), then **collapse to one hit per source file** keeping the best score and the order of first appearance, and return the first `n`. The raw passage list is kept for the secondary, passage-level report.

### 4.6 Reports (`compare.py`)
For each run: counts of first, top 3, per question file and for the primary and secondary sets; mean reciprocal rank and recall@10 at file level; misses; scores. For each pair of runs: **paired** counts of questions where only A or only B was right (first and top 3), and the net difference. Output: a Markdown table and a JSON file per stage in `reports/` (git-ignored; the table is pasted into spec 13 results by hand).

## 5. Pre-registered decision rule (fixed before any run)
- **Baseline for an embedder** = rules, normal mode, that embedder.
- A cell **beats the baseline** if, on the primary set, it is right-first for at least **5 more** questions or in the top 3 for at least **3 more**, and it is not worse than the baseline by more than 2 on the other measure. (The same margins as library3 plan rev 2.)
- A cell that does not beat the baseline but is within the margins is reported as **close**, never tuned afterwards.
- The **winning cell overall** is the beating cell with the most right-first questions; ties go to the simpler chunker, then the smaller embedder, then the faster run.
- Outside-question scores are printed but not compared across embedders (cosine scales differ by model). The "not found" threshold is chosen later, per model.
- Settings are not tuned per cell. One value each: `max_chars` 1500, `hard_limit` 6000, merge threshold 300, child size 500, semantic percentile 90, llamaindex chunk size about 375 tokens, late window 8192 (4096 only if memory forces it, noted).

## 6. Staging and the printed estimate
Each stage prints, before it runs anything: number of runs, passages, tokens to embed, an estimated time from a measured **throughput probe** (the first 256 passages), disk needed, and the run ids that already exist and will be skipped. The owner runs the stage. Rough scale, to be replaced by the probe: about 35,000 passages per chunker; tens of thousands of kilobytes of vectors per run; the 568M bge-m3 is the slow one.

- **Stage 0, smoke**: both embedders load; golden and fp16 checks pass; one small source (for example `python-peps`) goes through every chunker with the fake and then the real embedder; the harness runs on its questions.
- **Stage 1**: 14 runs. Report. Owner reads the table before stage 2.
- **Stage 2**: late mode on the best three chunkers per embedder.
- **Stage 3**: a separate spec (lexical hybrid, reranker, band for "not found").

## 7. Failure modes and chaos checks
| Failure | Check or defence |
|---|---|
| Wrong pooling or missing normalisation | golden check against `sentence-transformers` |
| Remote code of jina breaks or changes | pinned commit, files read once, offline load, smoke test first |
| GPU out of memory on 8192-token windows | window falls to 4096, recorded; batch size halves on error |
| fp16 differs from fp32 | agreement test; fp32 fallback |
| Run interrupted | temp folder then rename; resumable per stage |
| Same passage text in several sources | counted in the manifest; file-level collapse keeps them from filling the top 3 |
| Truncation hiding the end of long passages | truncated count in the manifest; passages over 512 tokens counted |
| Questions favour one variant | primary set excludes `rag-questions.yaml`; paired counts; ties reported |
| File-level questions hide chunk quality | stated limit; passage-level report is secondary; a quote-anchored set is a later stage |
| LlamaIndex nodes not exact | mapping check, 1% limit |
| Results not repeatable | rerun of one cell must give the same top 1 file for >= 99.5% of questions (GPU rounding) |
| Query and passage embedded differently | `kind` argument tested; neither model uses instruction prefixes (checked in the model cards before stage 0) |

## 8. Units, in order (each: tests first, mutation check where practical, commit)
1. **13a** Chunker switches in `check_chunking`; `fixed` and `merge-small`.
2. **13b** Embedder protocol, `FakeEmbedder`, `HFEncoder`, golden and fp16 checks, throughput probe.
3. **13c** Run folder, ids, atomic write; retriever with file-level collapse; paired report.
4. **13d** Estimate command and stage 0 smoke (owner runs).
5. **13e** Parent-child.
6. **13f** Semantic chunker.
7. **13g** LlamaIndex mapper.
8. **13h** Late chunking.
Stage 1 can start after 13a to 13g. Late chunking (13h) only needs to exist by stage 2.

## 9. Owner decisions (2026-10-29)
Asked with a recommendation each; the owner agreed to all four:
1. LlamaIndex goes in an **optional dependency group** (`variants`), not a variant branch. This amends the wording of ADR 0002 for this one case: runs are folders, so a branch is not needed to compare them.
2. **Torch with CUDA** goes into the `embed` group. The NVIDIA driver on the PC (610.60) is enough; CUDA comes inside the PyTorch wheel. Install lines are shown to the owner before they run.
3. Stage 1 uses the **full corpus** (23 snapshots).
4. The **fixed** and **merge-small** controls stay.
ADR 0007 was accepted by the owner the same day.

## 9a. Results, unit 13a (2026-10-29)
- Built: `check_chunking(..., structure=False)`, `chunk_source(..., version=, structure=)`, `chunk_fixed` (`fixed-1/<max>/<hard>`), `chunk_merge_small` (`merge-small-1/<max>/<hard>/<min>`), and `--chunker rules|fixed|merge-small` with `--min-chars` on `scripts/chunk_sources.py`.
- Tests: 32 new in `test_chunkers_baseline.py` plus 3 in `test_chunk_report.py`; full suite 604 passed, 2 skipped. Mutation check (22 mutants): 5 survivors at first, each closed with a test (window uses `max_chars` not `hard_limit`; line-end and paragraph-end preference; exact threshold and limit boundaries); none left.
- Real data, all 23 snapshots, every check passes (exit 0) for both:

| Chunker | Passages | Dropped empty | Under 100 chars | Over `max_chars` |
|---|---|---|---|---|
| rules (log predates the HTML and notebook readers) | 32,983 | 1,765 | 2,249 | 1,140 |
| fixed | 19,919 | 2 | 54 | 0 |
| merge-small (300) | 26,818 | 1,765 | 202 | 1,141 |

llamaindex-docs, the case that started this: rules 7,638 passages (median 153 chars), merge-small 3,907 (median 408), fixed 1,658 (median 1,411). The rules numbers will be re-run in stage 1 so all runs use the same readers.
- Note for the stage 1 report: `fixed` has no passage over `max_chars` by construction and almost no tiny ones, but it cuts through code blocks and sentences; this is exactly what the comparison should price.

## 9b. Results, unit 13b and 13c (2026-10-29)
- 13b: `embed/base.py`, `embed/hf.py`, `configs/models.yaml`. On the RTX 5070 Ti both models pass 20 `-m model` tests: vectors match the reference libraries (cosine >= 0.999), fp16 vs fp32 >= 0.995, batch size does not change vectors, truncation is counted, token offsets point into the text. bge-m3 pooling is CLS (confirmed by the match). Section 4.3 is kept as written; late chunking is checked in 13h.
- 13c: `runs.py`, `retrieve.py`, `compare.py`. Choices made while building, all inside section 4:
  - The run folder stores no passage text, only ids, source id, span, headings and the parent id; `vectors.npy` has a SHA-256 in the manifest and `load_run` checks it, the row count and that the manifest inputs hash back to the folder name. A complete run with the same inputs is not rewritten.
  - Search is exact cosine over all vectors. The best 200 passages are taken with ties broken by row position, so the same run always returns the same list. Children of one parent count once; then the list collapses to one hit per source file.
  - Depth: reciprocal rank and recall@10 only count ranks within `depth` (10); top 3 uses `k`.
  - Verdict words: **beats** (rule in section 5), **close** (not beating, neither measure down by more than 2), **worse** (anything else).
- Tests: 56 new (`test_runs.py`, `test_retrieve.py`, `test_compare.py`); full suite 689 passed, 2 skipped, 21 deselected.

## 9c. Results, unit 13d (2026-10-29)
- Built: `stage.py` and `scripts/run_stage.py` with three commands, all taking the same arguments: `estimate` (prints runs, passages, tokens, a speed sample, minutes and disk; writes no run), `run` (builds each run, skips ones that exist) and `report` (paired tables per embedder, primary and secondary question sets, verdict words from section 5). `--only` limits the corpus to chosen source ids (stage 0 uses `python-peps`). Embedders: `fake`, `bge-m3`, `jina-v2-base-en`. Chunkers so far: `rules`, `fixed`, `merge-small`; 13e to 13g add the rest to the same list.
- The run id covers chunker label (with sizes), embedding mode, embedder fingerprint, a hash of the corpus (source ids, versions, text hashes) and a hash of the question files, so a changed corpus or question file never reuses a run.
- Tests: 23 new in `test_stage.py`; mutation check 26 mutants, 2 rounds, none surviving (one mutant could not apply after a rewrite). Full suite below in PROGRESS.
- Stage 0 smoke on real data (`python-peps`, 3 sources, 72 / 42 / 69 passages): estimate, run and report work for `fake`, `bge-m3` and `jina-v2-base-en`. Both real models find the right file first for all 25 questions that python-peps can answer (the other 526 questions need other sources, so these tables only prove the pipeline, not a ranking). Speed sample on 72 passages: bge-m3 about 23,000 tokens per second, jina about 64,000; the sample includes model warm-up, so full runs should be faster than estimated.
- Seen: transformers prints "torch_dtype is deprecated, use dtype"; harmless with the pinned 4.x, to be fixed if the pin moves.

## 9d. Results, unit 13e (2026-10-29)
- Built: `chunk_parent_child` in `chunking.py` (version `parent-child-1/<max>/<hard>/<child>`, child size 500). Children are the paragraphs and code blocks of each section, cut at 500 characters by the spec 09 cut rules; a block is first clipped at the parent boundaries, so a child never straddles two parents. Parents are exactly the `rules` passages (`parent_of` maps child id to parent id). Blocks the rules chunker dropped (only headings or comments) give no children.
- Wired into the stage runner as chunker `parent-child`: only children are embedded; the run folder stores each child's parent id; retrieval counts the children of one parent once (best child score), then collapses to one hit per file.
- Tests: 16 new in `test_parent_child.py` plus 2 in `test_stage.py`. Mutation check: the remaining survivors are equivalent (the heading check on children can never fire because blocks are cut per section; taking blocks over the whole text gives the same children because of the parent clipping). Mutants that loop forever were stopped by hand and count as caught.

## 10. Not in this spec
Hybrid/lexical search, the reranker, the "not found" threshold, a quote-anchored question set, the content page, the paid embedders from library3.
