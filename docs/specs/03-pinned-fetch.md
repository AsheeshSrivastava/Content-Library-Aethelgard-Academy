# Spec 03: Pinned fetch

Status: Draft. Decisions: [ADR 0005](../adr/0005-port-register-gate-fetch.md), [ADR 0006](../adr/0006-licence-detection-cross-check.md). Ported from library3 `fetch.py`; merged with ideas from the production-rag GitHub loader and web fetcher.

## Purpose
Get an exact, hashed copy of an approved source and of its licence, without running anything fetched.

## Git fetch `fetch_git(record, snapshots, hosts)`
1. The clone URL must be https on an allowed host. **The check is inside this function.**
2. Shallow clone, no checkout, optional branch or tag `ref`, blob filter for https; sparse checkout for `paths` (folders) and `files`; always include root licence files.
3. Record the commit SHA, remove `.git`, drop symlinks (listed in `dropped`), refuse if larger than `max_mb`.
4. Licence file: first of LICENSE, LICENSE.md, LICENSE.txt, COPYING, COPYING.md, COPYING.txt, LICENSE-MIT, LICENSE-APACHE; its hash is stored. With `files` and no licence file, the hash covers the named documents and the name is `per-document`.
5. Result: `ref` (SHA), `content_hash` (hash over sorted relative paths and file bytes), `licence_hash`, `licence_name`, `dropped`.
6. The copy is built in a temp folder and moved into `snapshots/<id>/<sha>/` only when complete; an existing folder for the same SHA is replaced; the temp folder is always removed (read-only git files handled on Windows).
7. Timeout 900 s per git call; git prompts disabled; `core.autocrlf=false`.

## Pin by exact commit SHA (amendment 2026-10-29)
A record may carry `commit`, a full 40-hex SHA (short forms and branch names are refused here; `ref` stays for first-time fetches). When `commit` is set:
1. Init an empty repo, add the https remote, `git fetch --depth 1 --filter=blob:none origin <sha>`, then check out that SHA. No branch is cloned.
2. After checkout, `git rev-parse HEAD` must equal `commit`. If not, or if the host refuses to serve that SHA, raise `FetchError` naming the SHA and leave nothing behind.
3. Everything else (sparse paths, licence file, `.git` removal, symlinks, hashes, temp-then-move) is the same as above, so the same SHA and file selection must give the same `content_hash`.
4. Snapshots are stored under `snapshots/<id>/<sha>/`, so an existing local snapshot for a SHA can be reused without a fetch. Reuse still runs `verify`.

Why: the library3 snapshots were fetched by branch or tag and recorded a SHA afterward. Fetching by SHA lets us prove, later, that the stored bytes are exactly what that commit contains.

## Web fetch `fetch_web(record, snapshots, hosts)` (single paper page)
Both URLs must be https on allowed hosts, redirects included. The abstract page must declare the licence in the record. The page text is converted to Markdown; under 500 words is refused. Page size capped at 20 MB, timeout 60 s. The version is `web-` plus 12 hex characters of the text hash, so a changed page is a new pin. Files: `paper.md` and `LICENSE`.

## After a fetch
`apply_fetch(record, result, on)`: store ref, date, content hash, licence hash; if the licence hash differs from the stored one and status is approved or limited, status becomes `quarantine`. Licence detection (ADR 0006) runs on the fetched licence file and warns on a mismatch with the declared licence.

## Drift check `verify(record, snapshots)`
Returns problems: no stored snapshot, or the stored copy's hash differs from the pinned hash. Empty list means no drift.

## Failure behaviour
Any failure raises `FetchError` and leaves nothing behind. Messages name the URL or limit.

## Acceptance tests (offline, written first)
- Host check: rejects http, wrong host, host with credentials, and a redirect to another host.
- Local test repository built with `git init` in a temp folder: clone by file path under an explicit test hook; pinned SHA recorded; `.git` removed; symlink dropped and reported; sparse paths and files honoured; size cap refuses; licence file found for each name; dual-licence repo uses LICENSE-MIT; per-document licence hash.
- Same input gives the same content hash; changing one byte changes it.
- Interrupted copy leaves no folder; re-fetch of the same SHA replaces cleanly.
- `apply_fetch` quarantines on a changed licence hash and keeps status on an unchanged one.
- `verify` reports missing snapshot and changed bytes; returns empty otherwise.
- Web fetch with a stubbed downloader: licence not declared, too little text, oversize, redirect off-host.
- Pin by SHA, offline: in the local test repository with two commits, `commit` = the first SHA gives the first commit's files; a short SHA, a branch name and an unknown SHA are refused; a resulting HEAD that differs from `commit` is refused; nothing is left behind on failure.
- **Network acceptance test (marked `network`, run by the owner, not in the default suite):** pick two small approved sources whose snapshots already exist in the library3 snapshot folder. Re-fetch each at its recorded SHA with the new fetcher, then compare `content_hash` and `licence_hash` with the stored snapshot's. Both must match exactly; any difference is reported with the first differing path and the source is not migrated until explained. Failure of this test blocks reuse of the old snapshots as the main input.
