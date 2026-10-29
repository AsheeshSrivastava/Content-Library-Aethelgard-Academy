# ADR 0005: Port the source register, licence gate and pinned fetch from library3, with six changes

## Status: Accepted (owner, 2026-10-29)

## Context
The register, gate and fetch in library3 work: nothing is indexed until a person approves, approvals cannot exceed the licence ceiling without a written reason, and copies are pinned by hash. They also depend on the old app (`aethelgard.catalog`, the arXiv converter), share the name `Source` with the new contract, check the git host in the caller, have no licence detection, hash only whole snapshots, and cannot express mixed licences inside one repository. Reviewed on 2026-10-28 (files read: `register.py`, `gate.py`, `fetch.py`).

## Options considered
1. Leave them in library3 and import from there.
2. Rewrite from scratch using production-rag quarantine (in memory, no hashes).
3. Port library3's logic, test first, with targeted changes.

## Decision
Option 3. Changes: (1) no imports from the old app; own `RegisterError` and a copied arXiv converter; (2) the record is `SourceRecord`; (3) the host allow-list is enforced inside `fetch_git` and `fetch_web`; (4) licence detection as a cross-check (ADR 0006); (5) per-file raw hashes are computed at load time for the contract; (6) optional `licence_overrides` by path prefix. Behaviour of statuses, ceilings, evidence rules and drift check stays as in library3. Details: [spec 02](../specs/02-register-and-gate.md), [spec 03](../specs/03-pinned-fetch.md).

Existing register records and the local snapshots stay where they are; rag-core reads them by configured path and copies nothing until a record is migrated and its hash re-verified.

## Why
The rules were already tested against real sources and match the owner's "one poor source must not damage the library" requirement. Rewriting would discard that for no gain; importing would keep rag-core tied to the old app.

## Trade-offs accepted
Two copies of the logic exist for a while. A record's pinned hash stays valid only while the snapshot bytes are unchanged. Pinning is by exact commit SHA (amended 2026-10-29, see spec 03): fetch by SHA is only possible where the host serves it, and failure is an error, not a fallback to a branch. The library3 snapshots are reused after a two-source re-fetch proves equal hashes.
