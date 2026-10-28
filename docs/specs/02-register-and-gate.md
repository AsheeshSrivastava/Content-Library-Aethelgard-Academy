# Spec 02: Source register and licence gate

Status: Draft. Decisions: [ADR 0005](../adr/0005-port-register-gate-fetch.md), [ADR 0006](../adr/0006-licence-detection-cross-check.md). Ported from library3 `register.py` and `gate.py`, with the changes below.

## Purpose
Nothing is indexed until a person has approved its source, and no approval can exceed what the licence allows without a written reason.

## Data: `SourceRecord` (one YAML file per source)
Fields as in library3: `id`, `title`, `owner`, `url`, `kind` (docs, repo, paper, book, video, guide, course), `area` (python, ai-engineering, backend), `licence` {`id`, `file_url`, `covers` (code, text, both, unknown), `note`, `file_hash`}, `version` {`ref`, `fetched`}, `content_hash`, `status`, `reviewer`, `reviewed_on`, `override_reason`, `notes`, `fetch`, `max_mb`.
New: `licence.basis` (detected / declared, default declared), `licence_overrides` (map of path prefix to licence id, default empty) for mixed-licence repositories.
`fetch` is exactly one of `git` (with `ref`, `paths`, `files`) or `web` (with `licence_url`). Files must be plain relative paths with no `..`.
Renamed from `Source` so it does not clash with `contracts.Source`. Errors use `RegisterError`, not the app's `ContentError`.

## Statuses and transitions
`quarantine` (start), `approved`, `limited`, `reference-only`, `rejected`.
1. `rejected` is final.
2. Every change needs a reviewer name; the record stores reviewer and date.
3. Moving to `approved` or `limited` needs: licence file URL, pinned version, licence file hash, content hash. Otherwise `TransitionError` naming what is missing.
4. The gate gives a ceiling (below). A move above it is refused unless `override_reason` is written, which is then stored.
5. A fetch that changes the licence file hash sends an approved or limited source back to `quarantine`.

## Gate
Licence rules are a YAML file mapping classes `allowed`, `share-alike`, `non-commercial` to licence ids; an id in two classes is an error; an id in none is `unknown`.
Ceiling: unknown or non-commercial licence gives `reference-only`; `covers` unknown gives `reference-only`; `covers` not `both` gives `limited`; share-alike gives `limited` (and `share_alike=True`); otherwise `approved`. The gate never returns `rejected`.
Ranking: reference-only < limited < approved.

## Failure behaviour
Bad YAML, a missing field, a wrong choice, a duplicate id, a bad `max_mb` or `fetch.files` entry stop the load with `RegisterError` naming the file and field. A bad record never loads silently.

## Contract link
`to_contract_fields(record)` returns the licence id and basis for a loaded Source. Basis `unknown` is allowed only with licence `unknown`.

## Acceptance tests (written first)
- Load valid record; each missing field, bad choice, bad YAML, duplicate id raises with file and field.
- Save then load returns an equal record.
- Each transition rule 1 to 5, including override with and without reason.
- Gate matrix: each class x covers value, share-alike, unknown licence, duplicate class entry.
- `licence_overrides` prefix match: longest prefix wins; an override cannot raise the ceiling above the record's own.
- A record from the real library3 register loads unchanged (checked once, not committed as data).
