# Spec 06: Snapshot loader (GitHub at a pinned commit)

Status: Draft, waiting for owner review. Decisions: [ADR 0003](../adr/0003-contracts-canonical-text-and-ids.md), [ADR 0004](../adr/0004-loaders-and-inspection-script.md). Builds on specs 01 (contracts), 03 (pinned fetch), 05 (inspection script).

## Purpose
Turn one pinned snapshot folder plus its register record into `contracts.Source` objects, one per readable file. This is the shared front door: later format loaders (Markdown/MDX, rst, HTML, notebooks, plain text) are readers plugged into it by extension, each with its own spec. This spec ships the front door and a plain-text reader only, so the pipeline runs end to end before the format readers exist.

## Use
`load_snapshot(folder, record, rules, *, readers=None) -> LoadResult` in `src/ragcore/loaders/snapshot.py`. No printing, no network, no index.
`LoadResult` holds `sources` (sorted by path), `skipped` (path -> reason), `withheld` (path -> reason from the gate), `code_config` (paths), `counts` (the same five buckets as spec 05, plus `withheld`).

## Rules
1. **One file classifier.** The bucket decision (licence, ignored, code-config, skipped reasons, symlink, never `.git`) moves from the inspection script into one function, `classify_file`, used by both. The inspection script's behaviour and tests do not change. A file the inspector calls `read` is exactly a file the loader tries to read.
2. **Only gated files load.** The record's status must be `approved` or `limited`; any other status raises `LoadRefused`. Per file, `gate.check_path` decides: `reference-only` files are `withheld` with the gate's reason and never become a Source. Nothing is withheld silently.
3. **Source fields.** `source_id` = `<record.id>/<relative posix path>`; `version` = the pinned commit (`record.fetch.commit`, else `record.version_ref`; a record with neither raises `LoadRefused`); `text` = `normalize_text(reader output)`; `raw_sha256` = hash of the file bytes as read; `loader_version` = the reader's version string; `licence` and `licence_basis` from `register.to_contract_fields`, with the longest-prefix `licence_overrides` licence id for that path when one applies.
4. **Same input, same output.** Running twice on the same folder gives equal Sources in the same order. Output does not depend on folder location or file times.
5. **Skip, name, never crash.** A file that cannot be read becomes a `skipped` row with a spec 05 reason code. Reader exceptions become `unreadable`. A text that normalises to blank becomes `empty`.
6. **Stop on a bad source.** If the share of skipped files among read-or-skipped files is above `max_skip_share` (default 0.25, same as spec 05), `load_snapshot` raises `SkipShareExceeded` carrying the result, so a half-loaded source cannot reach the index.
7. **Plain-text reader (this spec only).** `.txt`, `.md`, `.mdx`, `.rst`, `.html`, `.htm` are read as UTF-8 and normalised, with no markup removal; it is labelled `plain-1` so nobody mistakes it for the real loaders. Real readers replace it per extension and bump the version.

## Loader checks this spec promises
- every folder file lands in exactly one bucket and the buckets add up (shared with spec 05);
- every Source passes `normalize_text(text) == text` and has a 64-hex `raw_sha256`;
- identical runs give identical Sources.
Format-specific checks (markup removed, front matter to metadata, outputs dropped and so on) belong to each format loader's spec.

## Acceptance tests
- `approved` and `limited` records load; `quarantine`, `reference-only` and `rejected` raise `LoadRefused`.
- A `licence_overrides` prefix that is `reference-only` withholds those files with a reason; others still load.
- Source ids, version and licence fields are exactly as rule 3 says, including an override licence id.
- CRLF, BOM and decomposed accents come out canonical; `raw_sha256` is of the original bytes.
- Reader that raises, binary, not-UTF-8, empty, too large: each is a named skip, the rest still loads.
- Over the skip share raises `SkipShareExceeded`; an empty folder cannot pass.
- Two runs are equal; a copy of the folder elsewhere gives equal Sources.
- `classify_file` unit tests, and the spec 05 tests still pass unchanged.
- Real-data check (not in the normal test run): load all 23 stored snapshots that have approved or limited records, and report Sources, skips and withheld per source.

## Out of scope
Chunking (next spec, with the exact-slice check), format readers, quarantined sources, embedding and indexing.
