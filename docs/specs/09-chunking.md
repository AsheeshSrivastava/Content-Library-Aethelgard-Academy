# Spec 09: Chunking (baseline) and the exact-slice check

Status: Accepted under the owner's "Ok Proceed" (2026-10-29); sizes are provisional until the retrieval comparison. Builds on specs 01 (contracts), 06 to 08 (loaders) and [ADR 0003](../adr/0003-contracts-canonical-text-and-ids.md).

## Purpose
Cut each loaded `Source` into `Passage` objects: the units that get embedded, searched and cited. This spec ships one baseline chunker, `heading-pack`, and the checking that every chunker must pass. Later chunkers (fixed window, parent-child, the LlamaIndex variant) plug in through the same interface and are compared on the 601 eval questions; none of that comparison is in this unit.

Because all readers now produce Markdown-form text (spec 07, spec 08), one chunker covers every source; library3's separate rst chunking is not needed.

## Interface (`src/ragcore/chunking.py`)
- `ChunkConfig(max_chars=1500, hard_limit=6000)`; its `version` is `heading-pack-1/<max_chars>/<hard_limit>`, so changing a size changes every passage ID (ADR 0003).
- `Chunker = Callable[[Source], list[Span]]`: a chunker only decides spans. `contracts.make_passages` turns spans into passages (exact slices, IDs).
- `chunk_source(source, config=ChunkConfig()) -> ChunkResult(passages, dropped_empty)`; it always runs `check_chunking` and raises `ChunkingError` if anything fails.
- `check_chunking(source, passages, config) -> list[str]`: the problems found, empty when the chunking is sound. Written against `Source` and `Passage` only, so any chunker can use it.

## Rules for `heading-pack`
1. **Sections**: a Markdown heading line (`#` to `######`) outside a fenced code block starts a section; text before the first heading is its own section. A section's heading path is the titles of its ancestors and itself (`("Guide", "Install")`); a skipped level contributes nothing. The path goes into the span's `headings`.
2. **Paragraphs**: inside a section, blocks are separated by blank lines outside fenced code; a fenced block is never split at its inner blank lines, and a fence is its own block even when no blank line separates it from the text next to it.
3. **Packing**: consecutive paragraphs of a section are packed into one passage while the passage stays within `max_chars`; a passage never crosses a section. A heading paragraph always goes with the paragraph after it, even when that makes the passage longer than `max_chars`, so a heading is never alone.
4. **Hard limit**: a passage longer than `hard_limit` is cut at the latest paragraph end inside the window, else the latest line end, else (one line longer than the limit) at a space in the second half of the window, else at the limit. A cut never leaves a piece that is only headings. Whitespace after a cut is skipped.
5. **Trim and drop**: passages carry no leading or trailing whitespace. A passage that is blank, or only headings and HTML comments, is dropped and counted in `dropped_empty`; nothing else is ever dropped.
6. **Deterministic**: same source and config give the same passages in the same order.

## The checks (`check_chunking`)
1. **Exact slice**: each passage's text equals `source.text[start:end]` and belongs to this source version (`verify_passage`).
2. **Order**: spans are in order and do not overlap.
3. **Not over the hard limit.** (A blank passage cannot be built: `make_passages` refuses blank spans.)
4. **No heading inside**: a heading line outside code may only be the first line of a passage.
5. **Coverage**: everything not inside a passage is whitespace, or headings and HTML comments only; no other text was lost.
6. **IDs**: every `passage_id` and `content_id` matches `contracts` (re-derived), and IDs are unique inside the source.

## Reporting
`scripts/chunk_sources.py --source ID --register DIR --snapshots DIR` loads a snapshot (spec 06), chunks every source, runs the checks and prints per source: sources, passages, dropped, median and 90th percentile length, passages under 100 characters, passages over `max_chars`, and any check failures. Exit code 1 if any check fails. Nothing is embedded or indexed.

## Acceptance tests
A before/after test for each rule (sections and paths with skipped levels, headings inside fences, fenced blocks with blank lines, packing limit, sticky heading, hard-limit cuts in all three forms, trimming, heading-only drop); a test for each of the six checks that builds a bad passage list and sees the problem named; property tests over random Markdown documents (checks pass, deterministic, sizes within limits, text outside fences unchanged); mutation check; real-data check on all 23 snapshots.

## Results (2026-10-29)
- Tests: 52 in `test_chunking.py` and `test_chunk_report.py`; full suite 476 passed, 2 skipped. A hand-made mutation check (34 mutants) left only equivalent ones: the whitespace skip after a cut (pieces are trimmed anyway) and the paragraph-end cut branch (a piece only reaches the hard cut when it is one block, or a heading plus one block, so a usable paragraph end rarely exists). Two redundant conditions in the heading-inside check and one in the drop test were removed instead.
- Real data, all 23 snapshots: every check passes. 32,983 passages, 1,765 dropped as empty, 2,249 under 100 characters, 1,140 over `max_chars`.
- Finding for the comparison: llamaindex-docs has 7,638 passages with a median of 153 characters and 1,483 under 100, because adjacent small sections are never merged. A "merge small sections" variant is a candidate chunker.

## Open questions for the owner (not blocking)
- The sizes (1500 and 6000) come from library3 and stay provisional until the comparison.
- At embedding time the heading path will probably be put in front of the passage text; the passage itself stays an exact slice. That is the embedder's unit, not this one.
