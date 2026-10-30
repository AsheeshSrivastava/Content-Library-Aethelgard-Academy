# ADR 0008: Corpus exclusions live in a rag-core file, not in the library3 register

## Status: Proposed (2026-10-30; the owner asked for the exclusions themselves, the mechanism is open for the owner to accept)

## Context
The noise audit (reports/noise-audit.md, 2026-10-30) found content that is approved and licensed but should not be searched: most of Python `library/` (tkinter, audio, curses and removed modules), migration and deprecation pages, link-list pages, release notes, one site-theme file, test fixtures. The owner said noise makes a bad RAG and asked to curate first, before any chunking. The register can only withhold files by licence (`licence_overrides`), which is the wrong lever: a file would have to be given a false licence id, and the register records live in the library3 app repository, which this branch treats as a frozen reference (ADR 0002).

## Decision
1. Exclusions live in one reviewed file in this repository, `configs/corpus-exclusions.yaml`: per source id, an ordered list of rules (`exclude` or `include`, a path glob, a written reason). The last matching rule wins; a file with no matching rule stays.
2. They are applied when the corpus is loaded (stage runner and question check), after the licence gate and before chunking. The register and the snapshots on disk are not changed, so the exclusion can be reversed by editing one file.
3. Every excluded file is counted with its reason, and a rule that matches no file is reported (a typo must not silently keep noise or drop content).
4. The corpus fingerprint is computed from the sources that remain, so every run records which corpus it used and an old run is never reused after the corpus changes.

## Why
It keeps licensing and relevance apart, keeps the app repo untouched, shows the whole curation in one file under review, and is reversible. The exclusion reasons also become the first draft of the "why is this not in the library" notes.

## Alternatives
- **Fake licence ids in `licence_overrides`:** works today with no code, but the licence field would lie, and the record changes live in the other repository.
- **Delete the files from the snapshots:** irreversible and breaks the pinned hash.
- **A new `exclude` field in the register records:** the right long-term home once the register is ported fully, but it needs a record schema change in two places; this file can be moved into the records later without changing the rules.

## Trade-offs accepted
- Two places now decide what is searchable (register status and this file). The loader reports both, so nothing is hidden.
- Glob rules are blunt; reviewing the list of excluded files in the report is the check.
