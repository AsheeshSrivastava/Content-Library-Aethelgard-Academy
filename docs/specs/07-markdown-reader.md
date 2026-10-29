# Spec 07: Markdown and MDX reader

Status: Accepted under the owner's "proceed in the order as you see fit" (2026-10-29); the cleaning rules stay open for the owner's comments. Builds on specs 05 and 06 and [ADR 0004](../adr/0004-loaders-and-inspection-script.md).

## Evidence (inspection reports, 2026-10-29)
23 sources hold 1,413 `.md` and 353 `.mdx` files. Leftover markup counted: 9,733 HTML tag matches, 1,838 JSX tag matches, 313 `import`/`export` matches, 1,046 table rows. Many of the counts are false alarms the reader must not touch: `<token>` or `<REGION>` written inside inline code, `import os` inside a code sample, and prose lines that begin with the word "import". Real noise: `import X from "..."` lines in MDX, components such as `<Callout type="info">`, `<Tabs>`, `<Tip warning={true}>`, `<Youtube id="..."/>`, multi-line `<ClickableImage ... />` and `<PageHero ... />`, HTML blocks such as `<iframe>`, `<img>`, `<div class=...>`, `<details>`, and YAML front matter (135 files in haystack-docs alone).

## Purpose
`read_markdown(raw: bytes) -> str` in `src/ragcore/loaders/markdown.py`, version `md-1`, registered for `.md` and `.mdx`. It makes the text a person would read: markup syntax goes, words stay. Tables are kept as they are (the chunker decides what to do with them).

## Rules (outside fenced code blocks and inline code spans, which are never changed)
1. **Decode** as UTF-8 (a leading BOM is fine); `LoadError("not-utf8")` otherwise. The result is passed through `normalize_text` by the snapshot loader as before.
2. **Front matter**: a `---` block at the very start, closed by `---`, is removed. If it has a `title` key and the body does not start with a `# ` heading, the text starts with `# <title>`, a blank line, then the body. Other keys are dropped for now: the contracts have no metadata field (open question below). An unclosed or non-YAML block is left alone.
3. **MDX only** (`.mdx`): top-level `import ... from "..."` and `export ...` lines are removed (a line must look like real ESM: `import` followed by a name, brace or quote, and `from` or a quote). In `.md` an `import` line is prose and stays.
4. **JSX components** (`.md` and `.mdx`): tags whose name starts with a capital letter are removed, including multi-line opening tags up to their closing `>` or `/>`. Text inside a wrapper (`<Callout>`, `<Tab>`, `<Tip>`) stays; a self-closing component (`<Youtube id="x"/>`, `<PageHero .../>`) disappears with its props (its attributes are configuration, not prose). Lower-case JSX such as `<hfoptions id="x">` is treated as HTML.
5. **HTML**: for a fixed list of tags (`div span details summary iframe img br hr p center picture source video a abbr sup sub kbd table thead tbody tr td th ul ol li`) the tag syntax is removed and the inner text kept; `img` leaves its `alt` text if it has one, `iframe` and `video` leave nothing. Unknown tags are left alone, because `<token>` in prose is more likely a placeholder than markup.
6. **Comments**: `<!-- ... -->` and MDX `{/* ... */}` are removed.
7. **Whitespace**: trailing spaces are trimmed outside code, and runs of 3 or more blank lines become 2.
8. **Idempotent**: reading the reader's own output gives the same text.
9. If nothing readable is left the reader returns an empty string. The registry wraps `.md` and `.mdx` so that nothing left is a named skip `empty`, and under 100 characters left (a page that was only front matter, a heading and a promo component) is a named skip `stub`. Neither counts as a regression in the inspection script. The readers themselves stay pure and are tested without this wrapper.
10. The plain-text stand-in reader (spec 06) also returns canonical text now (loader check: not normalised).

## Loader checks this reader promises
In addition to the spec 06 checks and the loader checks of spec 05 (empty output, too short, not normalised, more markup than before, non-deterministic, skipped by a loader that used to read):
- **Code survives**: every fenced block and inline code span of the input appears unchanged in the output.
- **No prose lost**: the words of the output (minus the front-matter title line) are a subsequence, in order, of the words of the input. Only tag syntax, attributes and removed lines may disappear.
- **Idempotent** (rule 8).
- **No leftover**: on the 23 snapshots, JSX component tags and real MDX `import` lines outside code drop to zero in the report.

## Acceptance tests
Each rule has a before/after test; each false alarm above has a "left alone" test (`<token>` in backticks, `import os` in a fence, an `import` prose line in `.md`, unknown tag in prose); the property checks run over every fixture and over a random mix of constructs; mutation check as usual. The snapshot loader's default readers become plain for the other text types and `md-1` for `.md` and `.mdx`, and the inspection script's loader mode uses the same registry. Real-data check: loader report over the 23 snapshots, with the before/after markup counts.

## Open question for the owner
Front matter keys other than `title` (for example `description`, `tags`, `sidebar_position`) are dropped. If you want them kept as searchable metadata, the contracts need a metadata field (a small ADR). Not needed for the first retrieval comparison.

## Result of the real-data check (2026-10-29)
Inspection script in loader mode over the 23 snapshots, before and after: leftover-markup matches 66,986 to 60,387 in total. Almost all of the rest is rst (python-docs, pytest-docs, python-peps: 0 change, as expected until the rst reader exists). Markdown-heavy sources dropped sharply, for example dair-ai-prompt-engineering-guide 797 to 17, mcp-spec 981 to 123, haystack-docs 464 to 118, fastapi-docs 795 to 248. Text shrank by 0.1% to 7%. 22 of 23 sources pass; 22 pages are named `stub` skips, 1 is `empty`. The one failing source, dair-ai-prompt-engineering-guide, fails only the "too short" suspicion check on `pages/techniques/graph.en.mdx` (1,251 characters of input, about 250 left): reviewed by hand, the rest of that page is `CoursesSection`/`CourseCard` promo components, so the cleaning is right and the check is a false alarm. The check stays as it is.
Found on the way and fixed: the inspection script crashed after saving its report when the console could not print a character (Windows cp1252); it now prints with replacement characters.