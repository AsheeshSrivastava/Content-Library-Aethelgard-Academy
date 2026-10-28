# ADR 0004: Loader set, inspection script first, skip policy

## Status: Proposed

## Context
Ten new sources arrived as Markdown, MDX, rst, HTML, JSX and mostly notebooks. library3 indexes only .md/.mdx/.rst and drops the rest silently. production-rag has a skip report and guard rail but no notebook, MDX or duplicate handling.

## Decision
- Build the **inspection script** first. Per source it prints files seen, indexed and skipped by reason; pieces over the cap or under 100 characters; leftover markup; directive-only stubs; front matter; notebook outputs; BOM, CRLF, non-NFC text; exact and near duplicates. It saves a text report beside the snapshot hash and fails when too large a share is skipped (threshold tuned on our sources).
- Loaders, in this order: GitHub at a pinned commit (best of both), Markdown/MDX with front matter to metadata and JSX tags stripped, rst, HTML, Jupyter notebook, plain text. Notebooks: markdown cells as is, code cells fenced, outputs dropped by default, cell index kept.
- Skip policy from production-rag ADR 0008: skip a bad file, name it with a reason code, stop when most files are skipped. Nothing is skipped without appearing in the report.
- Duplicates are flagged, not deleted. Exact: hash of canonical text. Near: hash after lowercasing and removing markup and whitespace. Query-time collapse is tested on the eval before adoption.

## Why
Cleaning changes results as much as chunking does; comparing chunkers on dirty text mixes two effects.

## Trade-offs accepted
Near-duplicate detection by normalized hash misses paraphrased copies; MinHash is a later step if needed. The skip threshold is untuned until real reports exist.
