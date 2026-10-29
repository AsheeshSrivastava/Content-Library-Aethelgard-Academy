# Spec 10: HTML reader

Status: Accepted under the owner's "then the HTML reader, then the notebook reader, then your recommended order" (2026-10-29). Builds on specs 05 to 09 and [ADR 0004](../adr/0004-loaders-and-inspection-script.md).

## Evidence (2026-10-29)
The 23 snapshots hold only 9 `.html` files, and none is a documentation page. Eight are site-theme templates under `docs/.overrides` and `docs/mkdocs/overrides` (Jinja: `{% extends "base.html" %}`, `{{ page.meta.title }}`, `{% block %}`), 87 to 2,766 bytes each. One is a 7.3 MB generated benchmark timeline (`vllm_bench_serve_timeline.html`), which the 5 MB file limit already skips. So today the reader mostly proves that templates are not mistaken for content; real HTML pages will appear when a source with built HTML is approved (for example a docs site fetched as pages). The reader is kept small for that reason.

## Purpose
`read_html(raw: bytes) -> str` in `src/ragcore/loaders/html.py`, version `html-1`, registered for `.html` and `.htm` (wrapped by the same stub check as `.md`). It makes the text a person would read, as Markdown-style text the chunker already understands. It uses only the standard library `html.parser`.

## Rules
1. **Decode** as UTF-8 (a BOM is fine); the loader already names a decode failure.
2. **Dropped with everything inside**: `script style head nav footer template noscript svg iframe`. The `<title>` text is kept aside; if the page has no `<h1>`, the text starts with `# <title>`.
3. **Headings** `h1` to `h6` become `#` to `######` lines.
4. **Blocks** (`p div section article main header ul ol table tr li dl dt dd blockquote figure details summary form`) and `br` start a new paragraph or line. `li` becomes `- item`. Table cells in a row are joined with ` | `.
5. **Code**: `pre` becomes a fenced block with its text unchanged (whitespace kept); `code` inline becomes backticks. A fence is made longer when the text holds backticks.
6. **Links and images**: the link text stays and the address goes; `img` leaves its `alt` text.
7. **Entities** are decoded. Other whitespace outside `pre` collapses to single spaces; 3 or more blank lines become 2.
8. **Template syntax** (`{% ... %}`, `{# ... #}`, `{{ ... }}`) is removed, outside `pre`.
9. Nothing left returns an empty string (the registry wrapper names it `empty` or `stub`).
10. **Deterministic and clean**: same input, same output; no triple blank lines, one final newline. (Not idempotent by design: the output is Markdown-style text, not HTML, so reading it again as HTML would join its lines.)

## Checks promised
Code survives (every `pre` text appears unchanged), no prose lost (the words of the output are a subsequence of the words of the input text nodes, outside dropped elements), deterministic, and no tag syntax left (`<tag ...>` for the dropped and handled tags).

## Acceptance tests
A before/after test per rule; "left alone" tests (`a < b` in text, `<` in `pre`, `&lt;div&gt;` decoded not parsed); fixtures from the real templates; property checks; mutation check; real-data check on the 9 files (8 become `stub`/`empty` skips, the 7 MB file stays skipped by size).

## Results (2026-10-29)
- Tests: 32 in `test_html_reader.py`; full suite 508 passed, 2 skipped. Mutation check (25 mutants): all caught except two equivalent ones (a self-closing `img` or `script` also calls the end handler, which does nothing). It found one real bug, now fixed: an empty list item left a lone `-` because the line trim removed the space the cleanup looked for.
- Real data (loader mode, ruff-docs, uv-docs, vllm-docs): `fathom.html` is empty and `main.html` is a stub, both named skips. `rule-filters.html` in ruff-docs (2,760 characters in, 243 out, markup 87 to 0) trips the "too short" suspicion: it is a theme partial with filter labels, reviewed by hand, not documentation, so the check stays as it is. uv-docs passes; ruff-docs shows that one failure. vllm-docs shows no html in its report (its html files are theme partials under `docs/mkdocs/overrides` and a 7 MB generated chart that sits outside the read paths).
- Finding: files under `.overrides` or `overrides` folders are site theme, not documentation. A register path rule can leave them out; that is a register change, not a reader change, and is not made here.
