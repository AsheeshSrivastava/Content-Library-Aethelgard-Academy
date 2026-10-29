# Carry-over (what came from where)

Reference code: branch `library3`, commit 2d61cc0, files at `D:\01-Project\unsloth\aethelgard-academy\.worktrees\library3\app\src\aethelgard\library\`. Reference project: `D:\01-Project\production-rag-with-LangChain-and-vector-databases`, PROGRESS.md dated 2026-10-28. Nothing is copied yet except the skill and the ADR template. Each row says what to port and what to re-test.

| Item | From | How | Status |
|---|---|---|---|
| Source register (YAML, quarantine first, statuses) | library3 `register.py` | Port; not yet read in detail | Planned, unit 2 |
| Licence gate and ceilings | library3 `gate.py` | Port; not yet read in detail | Planned, unit 2 |
| Pinned fetch, hashes, licence-file hash, host allow-list, symlinks dropped | library3 `fetch.py` | Merge with production-rag GitHub loader (owner approved: best of both) | Planned, unit 3 |
| Markdown and rst splitters | library3 `chunks.py` | Reuse as the rules baseline, behind the new chunker interface; add chunker_version | Planned, unit 4 |
| arXiv HTML to Markdown | library3 `webtext.py` | Keep as an HTML profile | Planned, unit 3 |
| Separate chunk and vector tables | library3 `index.py` | Keep the idea; change vector key to (content_id, model) | Planned, unit 6 |
| Eval harness, question sets (601, Codex/Gemini sets), result files | library3 `evaluate.py`, `eval-out/` | Carry as the measuring stick; change anchors | Questions and scoring done 2026-10-29 ([spec 12](specs/12-eval-harness.md)); library3 `search()`-bound parts not carried |
| Rerank and hybrid code | library3 `rerank.py`, `search.py` | Port later; embedder and reranker are chosen again | Planned, unit 5/6 |
| Contract ideas (Source/Span/Passage, skip report, guard rail) | production-rag `contracts.py`, ADRs 0005, 0008, 0011 | Rewrite here, test first | Planned, unit 1 |
| HTML extraction | production-rag `loaders/html_extract.py` | Reuse as a starting point | Planned, unit 3 |
| systems-architect-mentor skill, ADR template | production-rag `skills/`, `docs/adr/template.md` | Copied 2026-10-28 | Done |

Not carried: the old app front end, AXIS work, the 2026-10-26 embed of 11,475 pieces, parent-child experiment code (re-built only if the comparison needs it).
