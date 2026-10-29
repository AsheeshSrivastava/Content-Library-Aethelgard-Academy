# ADR 0006: Detect licences from the licence file and cross-check the declared one

## Status: Accepted (owner, 2026-10-29)

## Context
In library3 a person types the licence id and URL; nothing checks it against the fetched licence file. production-rag detects 12 common licences from distinctive phrases and records the basis (`detected`, `declared`, `unknown`). The contract (ADR 0003) carries `licence_basis`.

## Options considered
1. Human-declared only (library3).
2. Detection decides the licence.
3. Detection as a cross-check: the human still declares and approves; the tool warns when the file says something else.

## Decision
Option 3. After a fetch, the licence file is matched against a short list of common licences by distinctive phrases (copyright lines ignored). Outcomes: match equals declared (basis `detected`); no match (basis `declared`, noted); match differs from declared (warning in the fetch result; the source cannot move to approved or limited until the record is corrected or the reviewer writes an override reason). Dual-licence repositories and custom licences come back as no match, never as a guess.

## Why
A wrong typed licence is the failure the gate cannot see. Detection catches typos and copied records cheaply and makes `licence_basis` mean something, while a person still makes the decision.

## Trade-offs accepted
Phrase matching is crude and covers few licences. A detected match is evidence, not legal advice. The phrase list needs upkeep.
