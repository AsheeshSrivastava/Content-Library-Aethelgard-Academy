# Spec 08: reStructuredText reader

Status: Accepted under the owner's "ok, proceed" (2026-10-29); the cleaning rules stay open for comments. Builds on specs 05, 06 and 07, and [ADR 0004](../adr/0004-loaders-and-inspection-script.md).

## Evidence (inspection reports, stored snapshots)
480 `.rst` files: python-docs 397, pytest-docs 65, sentence-transformers 15, python-peps 3. They hold 34,239 role matches and 19,817 directive matches, almost all of the remaining leftover markup. Directive counts: `method` 3,561, `versionchanged` 2,415, `function` 2,121, `versionadded` 1,982, `code-block` 1,324, `attribute` 1,309, `data` 1,234, `class` 1,024, `note` 782, `index` 688, `doctest` 487, `availability` 412, `module` 320, `seealso` 306. Roles: `class` 7,578, `meth` 5,764, `func` 4,580, `mod` 2,703, `ref` 2,568, `exc` 2,153, `attr` 2,013, `term` 1,517. Heading adornments: `-` 2,005, `=` 864, `^` 780, `~` 354, `*` 170.

## Purpose
`read_rst(raw: bytes) -> str` in `src/ragcore/loaders/rst.py`, version `rst-1`, registered for `.rst`. It makes text a person would read, in Markdown form so the chunker sees one shape: headings become `#` lines, code becomes fenced blocks, roles and directives lose their syntax. Words stay.

## Rules
1. **Decode** UTF-8 and normalise, as spec 07.
2. **Headings**: a text line with an underline of one repeated punctuation character (at least as long as the text), or with matching over- and underlines, is a heading. Levels follow the order the styles first appear in the file (overline-plus-underline is a different style from underline only); the heading becomes `#` repeated by its level. A bare line of 4 or more repeated characters between blank lines is a transition and is removed. Table borders (which contain spaces) are not touched.
3. **Literal blocks**: a paragraph ending in `::` loses one colon (`::` alone on a line, or after a space, disappears) and the indented block after it becomes a fenced block, de-indented. Anything already inside a fence is left alone.
4. **Directives** (`.. name::` with optional argument, option lines, indented body; the body is cleaned by the same rules, recursively):
   - **Code**: `code-block`, `sourcecode`, `code`, `doctest`, `testcode`, `testsetup`, `testoutput`, `testcleanup`, `productionlist`, `grammar-snippet`, `parsed-literal` become a fenced block (the argument of `code-block`/`sourcecode`/`code` is the language); options are dropped.
   - **Dropped with their body**: `toctree`, `index`, `include`, `literalinclude`, `raw`, `contents`, `only`, `module`, `currentmodule`, `sectionauthor`, `moduleauthor`, `highlight`, `default-domain`, `rst-class`, `tabularcolumns`, `audit-event`, `audit-event-table`, `availability`, and anything in the `image` family without a caption.
   - **Version notes**: `versionadded`, `versionchanged`, `deprecated`, `deprecated-removed`, `versionremoved`: the version argument is kept as `3.9: ` in front of the first body line; with no body the directive disappears. (The "added in 3.8" fact of a bare `versionadded` is lost; see the open question.)
   - **Everything else keeps its argument as a line and its body de-indented**: object descriptions (`method`, `function`, `attribute`, `data`, `class`, `exception`, `option`, `opcode`, `decorator`, `describe`, `confval`, `fixture`, and the `py:`/`c:` forms), admonitions (`note`, `warning`, `tip`, `seealso`, `caution`, `important`, `impl-detail`, `admonition`, `sidebar`, `topic`, `rubric`), `tab`, tables, and any unknown directive. Option lines directly under a directive are dropped.
   - **Comments** (`.. text`), link targets (`.. _label:`), substitution definitions (`.. |x| replace::`) and footnote definitions are removed with their indented continuation.
5. **Inline** (outside fenced blocks): ` ``x`` ` becomes `` `x` ``; `` :role:`target` `` becomes the target (for `title <target>` the title); code-like roles (`class`, `meth`, `func`, `mod`, `exc`, `attr`, `const`, `data`, `keyword`, `deco`, `envvar`, `option`, `program`, `file`, `samp`, `kbd`, and the like, judged by the part after any `py:`/`c:` prefix) keep it as `` `x` ``, with a leading `~` meaning only the last dotted part (`~a.b.C` gives `C`), a leading `!` or `.` dropped; `:pep:` and `:rfc:` give `PEP 8` and `RFC 2616`; text roles (`ref`, `doc`, `term`, `dfn`, `abbr`, `issue`, `manpage`, and the rest) give plain text. `` `text <url>`_ `` and `` `text`_ `` give `text`. Footnote and citation references (`[1]_`, `[#]_`) are removed. Emphasis, field lists and tables are left alone.
6. **Whitespace**: trailing spaces trimmed, runs of 3 or more blank lines become 2.
7. **Idempotent**: reading the reader's own output gives the same text.
8. Nothing left gives an empty string; the registry turns that into `empty` or `stub` as for Markdown.

## Loader checks this reader promises
- **Code survives**: the text of every literal block and code directive appears unchanged (de-indented) inside a fence in the output.
- **No prose lost**: the words of the output (compared without case) are a subsequence, in order, of the words of the input. `PEP`/`RFC` come from the role name, which is in the input.
- **Idempotent** (rule 7), **canonical** and **deterministic** (spec 05 checks).
- **No leftover**: on the 23 snapshots, rst directive lines and role matches outside code drop to zero in the report (the report counts these outside fences only).

## Acceptance tests
A before/after test per rule and per directive family; "left alone" tests (table borders, a `::` inside a fence, backticked text with no role, an unknown role); the property checks over fixtures taken from the real files and over random mixes of constructs; mutation check as usual. Real-data check: loader report over the 23 snapshots with before/after markup counts.

## Open question for the owner
The version information of `versionadded`/`availability` with no body is dropped. Keeping it needs a metadata field (the same small ADR as the Markdown front matter keys). Not needed for the first retrieval comparison.

## Result of the real-data check (2026-10-29)
Inspection script in loader mode over the 23 snapshots: leftover-markup matches fell from 66,977 (before both readers) to 3,487 in total. The rst sources: python-docs 54,466 to 400, pytest-docs 2,019 to 9, python-peps 104 to 2. The 400 that remain in python-docs are false alarms of the counter: `<string>` and `<BLANKLINE>` written inside code spans. Text shrank by 16% in python-docs (directive and role syntax, index entries, `availability` lines). 22 of 23 sources pass; the one source that fails the "too short" suspicion check for rst is pytest-docs, on `doc/en/contents.rst`, which holds only a `toctree`, so cleaning it to almost nothing is right (it becomes a named `stub`). The other failing source is the Markdown one reviewed under spec 07. 28 pages are `stub` skips and 1 is `empty` in all.
Reading a real page (`json.rst`) by eye: headings, the `RFC 7159` form, code in fences and inline code spans all came out as intended.
Mutation check: 33 of 34 hand-made breaks caught after 8 test gaps were closed; the survivor (an overline heading directly after text without a blank line) is equivalent for valid rst.
Implementation note: removing a construct can bring two lines together that now form a heading, so the reader cleans until the text no longer changes (at most 3 rounds). This makes rule 7 (idempotent) true by construction.