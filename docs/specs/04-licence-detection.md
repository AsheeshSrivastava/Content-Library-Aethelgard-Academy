# Spec 04: Licence detection (cross-check)

Status: Accepted by the owner 2026-10-29. Decision: [ADR 0006](../adr/0006-licence-detection-cross-check.md) (Accepted). Detector ported in idea and phrase list from production-rag `licenses.py` (12 licences); two Creative Commons titles added because the register uses them.

## Purpose
After a fetch, read the licence file we hashed and check that it says what the register record declares. A person still declares and approves; the tool only warns. It never guesses.

## Detector `detect_licence(text) -> str | None`
Returns an SPDX id or `None` (no match). It never raises: empty, binary-looking or odd input gives `None`.
1. Prepare: if the text is an HTML page (the CC legal code is fetched as one), strip tags and entities first; drop copyright lines, lowercase, collapse spaces, keep only the first 4000 characters. A README that quotes a licence far below its start is not that licence.
2. Licences recognised (14): MIT, ISC, BSD-2-Clause, BSD-3-Clause, Apache-2.0, MPL-2.0, GPL-2.0, GPL-3.0, LGPL-3.0, AGPL-3.0, CC0-1.0, Unlicense, CC-BY-4.0, CC-BY-SA-4.0.
3. GNU and Mozilla licences are judged by the first licence name in the title and its version number (the GPL text mentions the Lesser and Affero licences further down). A version we do not list (GPL v1, LGPL 2.1, MPL 1.1) gives `None`.
4. BSD: the standard opening plus both redistribution clauses; "neither the name of ... endorse or promote" makes it 3-Clause, otherwise 2-Clause; an advertising clause gives `None`.
5. CC: the legal-code title "Creative Commons Attribution 4.0 International Public License" or "...Attribution-ShareAlike 4.0 International Public License"; the NonCommercial variants give `None` (they are not on our list, and the gate treats an unknown id as reference-only anyway).
6. Anything custom, modified, or a file holding two licence texts gives `None`.

## Reading the file
`read_licence_file(folder, name) -> str | None`: the file the fetch already chose (`licence_name`), read as UTF-8, at most 200 KB; too large or not UTF-8 gives `None`. For `per-document` (licence stated inside each document) there is no file, so detection is skipped and the result is "undetected".

## Cross-check `cross_check(declared, detected) -> Outcome`
Compare after trimming and case folding; a trailing `-only` or `-or-later` on the declared id is ignored (the text cannot tell them apart). A declared expression `A OR B` matches when the detected id is A or B.
| declared | detected | outcome | basis kept | warning |
|---|---|---|---|---|
| X | X (or listed alternative) | `match` | `detected` | none |
| X | none | `undetected` | `declared` | none (a note in the fetch result) |
| X | Y, Y not X | `mismatch` | `declared` | "the licence file says Y, the record says X" |
| `unknown` | Y | `mismatch` | `declared` | same text |

## Record changes
- `SourceRecord.licence_warning: str` (default empty), saved and loaded with the record.
- `FetchResult.detected_licence: str` (empty when none).
- `apply_fetch` runs the cross-check on every fetch: sets `licence.basis` and `licence_warning` as the table says (a later fetch that matches clears an old warning). A mismatch on an approved or limited source sends it to `quarantine`, as a changed licence hash already does.
- `set_status`: while `licence_warning` is not empty, a move to `approved` or `limited` needs the reviewer's written `override_reason`, or the record to be corrected (fix the declared id, then re-run the cross-check, which clears the warning). Other moves are unaffected.
- `recheck_licence(record, detected) -> record` does the cross-check alone, for use after a correction.

## Failure behaviour
Detection and file reading never raise. A bad outcome is a warning in the record, never an exception, except that `set_status` refuses as above.

## Acceptance tests (written first)
- Each of the 14 licences is detected from its real opening text (short texts in the test file); copyright lines and years do not matter.
- Not detected: empty, whitespace, binary, README quoting MIT after 5000 characters, custom text, BSD with advertising clause, GPL v1, LGPL 2.1, MPL 1.1, CC BY-NC.
- GPL text that mentions LGPL and AGPL below its title is GPL, not LGPL.
- Cross-check table: every row, `-only` and `-or-later` suffixes, `MIT OR Apache-2.0`, case folding, declared `unknown`.
- `read_licence_file`: normal, too large, not UTF-8, missing file, `per-document`.
- `apply_fetch`: match sets basis `detected`; undetected keeps `declared`; mismatch sets the warning and sends approved or limited to quarantine; a later matching fetch clears the warning.
- `set_status` refuses approved or limited with a warning and no reason; accepts with a reason (reason stored); accepts after `recheck_licence` cleared the warning.
- Record with `licence_warning` saves and loads equal.
- Once, not committed as data: run the detector on the licence file of every library3 snapshot and report match, undetected and mismatch per source for the owner to read.
