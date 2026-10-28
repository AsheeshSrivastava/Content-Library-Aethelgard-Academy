# Spec 01: Contracts

Status: Draft. **Written after the code** (unit 1, commit 2405fae); it records what `src/ragcore/contracts.py` and `tests/test_contracts.py` do. Decision: [ADR 0003](../adr/0003-contracts-canonical-text-and-ids.md).

## Purpose
One agreed shape for loaded text and cited passages, so every later stage can check a quote mechanically.

## Objects
- `normalize_text(text)`: removes a leading BOM, applies Unicode NFC, turns CRLF and lone CR into LF. Idempotent. Other whitespace untouched.
- `Source`: `source_id`, `version`, `text` (canonical), `raw_sha256` (64 hex), `loader_version`, `licence`, `licence_basis` (detected / declared / unknown), `normalizer_version`, `schema_version`, and `text_sha256` (derived). Frozen.
- `Span(start, end, headings)`: `0 <= start < end`.
- `Passage`: slice of a Source with `headings`, `chunker_version`, `passage_id` (checked against its inputs), `content_id` (hash of the text).

## Rules
1. Source text must equal `normalize_text(text)`; otherwise `ContractError`.
2. Required text fields are non-blank; `raw_sha256` is lowercase hex of length 64; `licence_basis` is one of the three values.
3. `make_passages` raises on a span past the end of the text or a blank slice, and on an empty `chunker_version`.
4. `passage_id = make_passage_id(source_id, version, start, end, content_id, chunker_version)`. It changes with any input. `content_id` does not depend on position, source or chunker.
5. `verify_passage(source, passage)` is true only for the same source id and version and when `source.text[start:end] == passage.text`.

## Not covered yet
Title, URI and metadata fields on Source (added with the first loader). Whether `licence_basis=unknown` must pair with `licence=unknown` (decided in spec 02). Serialization of `schema_version`.

## Acceptance tests
The 32 tests in `tests/test_contracts.py`; 14 hand-made breaks of the code were each caught.
