# Spec 17: Persistent index, automatic sync and retriever adapter

Status: Draft, follows ADR 0011 (Proposed). Written before the code.

## 1. Purpose
Turn the chosen setup (ADR 0010: jina-v2-base-en, `llamaindex` chunker, hybrid search, local reranker) into a working library: an index that survives restarts, updates only what changed, and answers `search(query)` with citable passages.

## 2. The store (`index_store.py`)
File `index.db` (SQLite, standard library). Tables:
- `meta(key, value)`: `schema_version`, `embedder_id`, `embedder_revision`, `chunker`, `chunker_version`, `chunk_config`, `reranker_id`, `reranker_revision`, `created_at`.
- `sources(source_id PK, version, content_sha256, url, licence, loaded_at)`.
- `passages(id INTEGER PK, passage_id UNIQUE, source_id, start, end, heading_path, text, text_hash)`; `text` equals the source slice `[start:end]`.
- `vectors(passage_id PK, embedder_id, dim, vector BLOB)` float32, unit length.
Rules:
1. `open_index(path, setup)` creates the file with the setup on first use. On later use it compares the setup with `meta` and raises `IndexError` ("rebuild the index") on any difference.
2. A `schema_version` newer than the code is refused. An older one is copied to `index.db.bak-<version>` before it is upgraded.
3. All writes for one source happen in one transaction (`replace_source`), foreign keys on. A crash between sources leaves a valid index.
4. A lock file stops two syncs at once with a clear message.

## 3. Sync (`index_sync.py`)
`sync(register, snapshots, exclusions, setup, embedder, store) -> SyncReport`.
1. Load approved, gate-passed, non-excluded sources (same loading as the comparison).
2. For each source: unchanged if stored `content_sha256` and `chunker_version` match, else cut passages with the chunker, drop the old rows of that source, insert the new ones.
3. A passage whose `text_hash` already has a vector under this embedder reuses it; only the rest are embedded, in batches.
4. Sources in the store but not in the loaded set are deleted with their passages and vectors.
5. `SyncReport`: sources added, replaced, unchanged, deleted; passages added, reused, deleted; vectors embedded; seconds. A dry run (`--dry-run`) reports what would change and writes nothing.
6. Running sync twice in a row changes nothing the second time.

## 4. Retriever (`index_retriever.py`)
- `Retriever.open(path, setup, embedder, reranker)` loads all vectors into one array and builds the keyword index from stored texts (spec 15 section 3.1).
- `search(query, k=5, max_per_file=None) -> list[Result]`: vector top 200, keyword top 200, rank fusion, reranker on the first 30, then the first `k` (limited to `max_per_file` per source when set). Empty or whitespace query returns an empty list. An empty index returns an empty list.
- `Result` (frozen): `passage_id, source_id, url, licence, heading_path, text, start, end, rank (from 1), score`. Scores are the reranker's, so they compare only within one search.
- Same query and same index give the same list (ties by row position).

## 5. Commands (`scripts/index.ps1`, owner's PC)
- `index.ps1 sync [-DryRun]`, `index.ps1 status` (setup, counts, last sync, size), `index.ps1 search "question" [-K 5]` (prints rank, heading path, source URL, first 200 characters).
- Paths come from the same options as `stage4.ps1`; first run prints the number of passages to embed and the time estimate, and costs no money.

## 6. Failure behaviour
Setup mismatch, newer schema, a source whose text no longer matches its stored hash after load (corrupt snapshot), a locked index: each stops with a plain message and changes nothing. A power cut during sync: next `sync` completes it.

## 7. Acceptance tests
1. Fresh sync of a small fake corpus fills all tables; `text` equals the source slice for every passage.
2. Second sync: all sources unchanged, zero embedded.
3. Edit one source: only that source's passages change; other vectors are not re-embedded; unchanged passages inside the edited source reuse their vectors.
4. Remove a source from the register: its passages and vectors disappear.
5. Setup mismatch (other embedder, chunker, reranker) is refused and the file is unchanged.
6. Newer schema refused; older schema backed up then upgraded.
7. Interrupted sync (fake failure after N sources): store valid, rerun completes and equals an uninterrupted sync.
8. `search` order equals the stage 3 `HR` pipeline on a fake corpus with the fake embedder and fake reranker; ties repeatable; `max_per_file` respected; empty query and empty index return `[]`.
9. Second sync lock refused.
10. **End-to-end:** with the real setup, sync the real corpus, then score the 137 frozen passage-level questions through `Retriever.search`: right first and top 3 must equal the stage 4 HR result for jina/llamaindex (100 and 121), and must be repeatable. A difference stops the unit.
11. Mutation check over the store, sync and retriever logic.

## 8. Units
17a store, 17b sync, 17c retriever, 17d commands and owner's script, 17e end-to-end check and report section 10.
