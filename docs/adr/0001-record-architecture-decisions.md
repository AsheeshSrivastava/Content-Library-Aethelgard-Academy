# ADR 0001: Record architecture decisions

## Status: Accepted (2026-10-28, owner)

## Context
Decisions here are hard to reverse once data is indexed (IDs, formats, embedding model). Chat history does not survive between sessions.

## Decision
Write an ADR when a decision is hard to reverse, had a real alternative, and someone could ask why. Write it before the code when possible. Status starts as Proposed; only the owner sets Accepted. Never delete an ADR; supersede it with a new one. Template: [template.md](template.md).

## Why
A short written record lets a fresh session or reviewer see what was decided and why.

## Trade-offs accepted
Some writing effort. Choices that can be undone in minutes get no ADR.
