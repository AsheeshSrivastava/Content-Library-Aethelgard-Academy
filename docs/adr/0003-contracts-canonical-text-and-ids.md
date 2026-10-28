# ADR 0003: Canonical text, exact-slice passages, two IDs

## Status: Accepted (2026-10-28, owner)

## Context
Citations are only checkable if a passage can be found again byte for byte. library3 slices raw file text; production-rag normalizes once and derives IDs from content. Vectors are the costly part, so a re-chunk should not re-embed unchanged text.

## Decision
- **Canonical text.** The loader output is the canonical text. Normalize once when a Source is made: BOM removed, Unicode NFC, `\n` line endings. Spans and quotes are slices of this text, never of the raw file.
- **Source fields.** `source_id`, `version` (commit or content hash), `raw_sha256`, `text_sha256`, `loader_version`, `normalizer_version`, `schema_version`, licence and its basis. Changing the normalizer changes IDs, so its version is stored.
- **Exact slice.** `passage.text == source_text[start:end]`, checked on read-back in tests and in the inspection script.
- **Two IDs.** `passage_id` = hash(source_id, version, start, end, text_hash, chunker_version): positional, used in citations. `content_id` = text hash after normalization: identifies the text. Vectors are keyed by (`content_id`, model).
- **Chunker and load report.** `chunker_version` is stored per build. The load report (files seen, indexed, skipped by reason) is persisted.
- **Evaluation anchors** are `(source_id, file/heading, quote)`, never passage IDs.

## Why
Makes quote checking mechanical, makes re-index cheap, and lets a later reader tell which loader, normalizer and chunker produced a result.

## Trade-offs accepted
Every ID changes if the normalizer changes (hence the stored version). Two IDs are more to explain. For notebooks and HTML the citation points into derived text, so the Source also keeps the raw file hash and, for notebooks, the cell index.
