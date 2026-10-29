# Spec 05: Inspection script

Status: Draft, awaiting the owner's review. Decision: [ADR 0004](../adr/0004-loaders-and-inspection-script.md) (Accepted). Ideas taken from production-rag `scripts/inspect_sources.py` and `inspection.py`; this version works on pinned snapshot folders and adds the checks ADR 0004 lists.

## Purpose
Show, per source, what is really in the snapshot before any loader or chunker is written, so loaders are designed on evidence. Nothing is changed and nothing is indexed. It runs before the loaders exist, so it reads files in a plain "look" mode: bytes decoded as UTF-8 and normalised with `contracts.normalize_text`. Loader-specific cleaning (front matter to metadata, JSX stripping, rst directives, notebook cells) is measured here and done later by the loaders.

## Use
`uv run python scripts/inspect_sources.py <snapshot-folder> [--id ID] [--out DIR] [--max-skip-share 0.25] [--section-cap 6000]`
Also `--source ID --register DIR --snapshots DIR`, which finds the folder from the record's `version_ref`. Exit code 0 = report written and gate passed, 1 = gate failed (report still written), 2 = bad input. The logic lives in `src/ragcore/inspect_sources.py` (testable, no printing); the script only parses arguments and prints.

## What each file becomes
Every file under the folder (never `.git`) lands in exactly one bucket, and the buckets add up to the files seen:
1. `read`: extension in `.md .mdx .rst .txt .html .htm .ipynb` (notebooks are parsed as JSON; markdown and code cells counted, outputs measured and left out of the text).
2. `licence`: files named like a licence file (spec 03 names), counted but not analysed.
3. `ignored`: by design, not content: images, fonts, audio/video, archives, `.css .js .map .lock .svg`. Not counted in the skip share.
4. `skipped`, each with one reason code: `no-reader` (any other extension), `binary` (NUL bytes), `not-utf8`, `too-large` (over 2 MB), `empty`, `unreadable` (OS error). These count in the skip share.

## Checks on `read` files
Counted per file and summed per source, with up to 5 example paths each (examples never longer than 120 characters of text):
- Raw text problems before normalising: BOM, CRLF or lone CR, text that changes under NFC, NUL.
- Leftover markup in the normalised text: HTML tags, JSX/MDX tags and `import`/`export` lines, rst directive lines (`.. name::`) and roles (`:ref:`), Markdown tables, setext underlines left as text.
- Front matter present (`---` block at the very start, or MyST/rst field lists).
- Tiny files (under 100 characters) and directive-only stubs (under 100 characters of prose once directive lines, headings and toctree entries are removed).
- Sections (split at Markdown or rst headings, and never at lines inside code fences) over `--section-cap` characters (default 6000, provisional) or under 100 characters.
- Notebooks: cells by kind, share of the file that is outputs.
- Idempotence: `normalize_text(normalize_text(t)) == normalize_text(t)` for every file; any failure is listed as a bug.
- Duplicates: exact (same hash of normalised text) and near (same hash after lowercasing and removing markup characters and all whitespace) groups, listed by path. Flagged, never removed.

## Report
Plain text, sorted so the same snapshot gives a byte-identical report. Header: source id, snapshot hash (`fetch.snapshot_hash`), normaliser version, thresholds used. Then the bucket table, skipped files grouped by reason (every one named, up to 50 per reason, then a count), the checks above, and a final line `GATE PASSED` or `GATE FAILED: <share> of candidate files skipped (limit <limit>)`. Saved as `<out>/<id>-<first 12 hex of snapshot hash>.txt` (default `reports/`, which is git-ignored because reports quote document text). A JSON copy with the same numbers sits beside it for later comparison.

## Gate
Skip share = `skipped / (read + skipped)`, files not bytes. Default limit 0.25, provisional: it is tuned once real reports exist and the chosen values are recorded in PROGRESS.md and this spec. A folder with no candidate files fails.

## Failure behaviour
One bad file is a `skipped` row, never a crash. Missing folder or bad arguments exit 2 with a plain message. The folder is only read: a test compares its hash before and after.

## Not in this spec
The exact-slice check of passages (needs the chunker, spec to follow) and any cleaning. Loader specs follow, using what this report shows.

## Acceptance tests (written first, small fixture folders in a temp dir)
- Buckets add up to files seen; each skip reason produced by a fixture; ignored files do not count in the share.
- Each raw problem and each markup kind counted correctly; examples capped at 5 and 120 characters.
- Front matter, tiny file, rst stub, section over and under the cap (code-fence lines not treated as headings).
- Notebook: cell counts, output share, outputs absent from the text, bad JSON becomes `skipped: unreadable`.
- Exact duplicates and near duplicates grouped; unique files not grouped.
- Gate passes below and fails above the limit; empty folder fails; limit is configurable.
- Same folder gives an identical report twice; snapshot hash unchanged afterwards; file name carries the hash prefix.
- Script: exit codes 0, 1, 2; `--source` finds the folder from a record.
- Once, not committed: run on all 23 stored snapshots; the owner reads the reports; thresholds are set from them.
