# Spec 17: Persistent index, automatic sync and retriever adapter

Status: Accepted by the owner 2026-10-30, follows ADR 0011 (Accepted). Written before the code.

## 1. Purpose
Turn the chosen setup (ADR 0010: jina-v2-base-en, `llamaindex` chunker, hybrid search, local reranker) into a working library: an index that survives restarts, updates only what changed, and answers `search(query)` with citable passages.

## 2. The store (`index_store.py`)
File `index.db` (SQLite, standard library). Tables:
- `meta(key, value)`: `schema_version`, `embedder_id`, `embedder_revision`, `chunker`, `chunker_version`, `chunk_config`, `reranker_id`, `reranker_revision`, `created_at`.
- `sources(source_id PK, version, content_sha256, chunker_version, url, licence, loaded_at)`.
- `passages(id INTEGER PK, passage_id UNIQUE, source_id, start_pos, end_pos, heading_path, text, text_hash)`; `text` equals the source slice `[start_pos:end_pos]`; `heading_path` is a JSON list; `text_hash` is the SHA-256 of `text` (checked on every write).
- `vectors(passage_id PK, embedder_id, dim, vector BLOB)` float32, unit length (within 0.001), one dimension for the whole file (kept in `meta` as `dim` after the first vector).
- Rows come back in the order they were added (`id`), which is the row position used for ties.
Rules:
1. `open_index(path, setup, read_only=False, create=True)` creates the file with the setup on first use. On later use it compares the setup with `meta` and raises `IndexStoreError` ("rebuild the index") on any difference. (The name avoids Python's built-in `IndexError`.)
2. A `schema_version` newer than the code is refused. An older one is copied to `index.db.bak-<version>` before it is upgraded.
3. All writes for one source happen in one transaction (`replace_source`), foreign keys on. A crash between sources leaves a valid index.
4. An operating-system lock on `index.db.lock` stops two writers at once with a clear message; the system releases it if the process dies, so a power cut leaves no stale lock. Read-only opens never take it.
5. The file uses SQLite's write-ahead mode (`journal_mode=WAL`), so searches are never blocked while a sync writes.
6. The retriever opens the file **read-only** (`mode=ro`); only sync opens it for writing. A test proves that a write through the retriever's connection fails.

## 3. Sync (`index_sync.py`)
`sync(sources, store, chunk, embedder, *, urls=None, batch=32, dry_run=False, allow_mass_delete=False, progress=None) -> SyncReport`.
The caller (the `index.ps1` command, unit 17d) loads the approved, gate-passed, non-excluded sources exactly as the comparison did and passes them in as `Source` objects, together with `urls` (register entry, the first part of the source id, to its url). `sync` itself never reads files, so it is tested without a corpus.
1. Sources are handled in sorted order of source id; two sources with the same id are refused. `progress(done, total, source_id)` is called after each.
2. A source is unchanged if its stored version, `content_sha256`, `chunker_version`, url and licence all match; otherwise it is chunked and replaced in one transaction (`replace_source`).
3. A passage whose text hash already has a vector in the index (also from another source, or from the source being replaced) reuses it; the same text inside one run is embedded once. Only new texts are embedded, in batches of `batch`.
4. Sources in the index but not in `sources` are deleted with their passages and vectors, after the other sources are done (so their vectors can still be reused). **Mass-delete guard:** if that would remove half or more of the stored sources, `sync` stops before changing anything with a plain message, unless `allow_mass_delete` is set. This protects against a wrong folder or an empty load.
5. `SyncReport`: sources added, replaced, unchanged, deleted; passages added, reused (passages stored without a new embedding), deleted (including the old passages of replaced sources); vectors embedded; seconds; dry run. `dry_run` reports what would change and writes and embeds nothing (limit: if a replaced source gives up a text that a later source needs, the real run embeds it again and the dry run does not count it).
6. Running sync twice in a row changes nothing the second time. A crash or an embedder failure leaves every finished source stored; the next sync completes the rest.
7. **Same inputs as the comparison.** A passage is embedded as its exact text with `kind="passage"` (the heading path is NOT put in front, as in the stage 1 to 4 runs); a question is embedded with `kind="query"`. Changing this would break the end-to-end check in section 7 test 10.
8. **Pinned chunker dependency.** `llama-index-core` is a pinned dependency of the index (exact version in `pyproject.toml` and recorded in `meta` as part of the setup), because another version can move the cuts and change passage IDs (spec 09 update of 2026-10-30). The command in 17d refuses to run when the installed version differs from the one in `meta`.
9. **Chunker check.** Every passage the chunker gives must carry the chunker version recorded in `meta`; otherwise `sync` stops with a `SyncError` and stores nothing for that source. The embedder must return exactly one vector per text; the store refuses vectors that are not unit length.

## 4. Retriever (`index_retriever.py`)
- `Retriever.open(path, setup, embedder, reranker, *, n_raw=200, depth=30)` opens the file **read-only** (never creating it; a different setup is refused), loads all vectors into one array and builds the keyword index from the stored texts (spec 15 section 3.1). It uses the stage 3 `HR` ranking code itself (`retrieve.make_ranker`), so the measured pipeline is the one that runs. A new `Retriever` is needed to see a later sync. `close()` or a `with` block closes the file.
- `search(query, k=5, max_per_file=None) -> list[Result]`: vector top `n_raw`, keyword top `n_raw`, rank fusion, reranker on the first `depth`, then the first `k` (limited to `max_per_file` per source when set; only the reranked passages can be returned, so a limit can leave fewer than `k`). Empty or whitespace query returns an empty list without embedding. An empty index returns an empty list. `k` or `max_per_file` below 1 raises `ValueError`; a closed retriever raises `IndexStoreError`.
- `Result` (frozen): `passage_id, source_id, url, licence, heading_path, text, start, end, rank (from 1), score`. Scores are the reranker's, so they compare only within one search.
- Same query and same index give the same list (ties by row position).

## 5. Commands (`index_cli.py`, `scripts/run_index.py`, `scripts/index.ps1` on the owner's PC)
- `index.ps1 sync [-DryRun] [-AllowMassDelete]`: loads the corpus exactly as the comparison did (register, snapshots, exclusions file; a source's url comes from its register record), then calls `sync` (section 3) with the setup read from `configs/models.yaml` and `configs/rerankers.yaml` (embedder and reranker revisions, the embedder's code pin, the chunker version, the installed `llama-index-core` version). Models and the chunker are built only when needed, so a sync with nothing new loads no model. Prints progress every 100 sources and the report. A dry run on a missing index creates no file. Loading no sources at all is refused (exit 2). After a sync the chunker's own check runs (over 1% of LlamaIndex nodes not found in the text); if it fails the command exits 1 and says not to trust the index.
- `index.ps1 status`: opens the index read-only (works while a sync runs) and prints counts, last sync, schema version, vector size and the setup the file was built with. Loads no model.
- `index.ps1 search "question" [-K 5] [-MaxPerFile 2]`: prints rank, score, heading path, source link, source id, licence and the first 200 characters of each passage.
- Exit codes: 0 done, 1 could not continue (index unchanged or still valid), 2 bad input. The index file defaults to `~\.ragcore\index\index.db` (`-Index` changes it). The cost is no money: everything runs on the PC. `index.ps1 sync -DryRun` first shows the number of passages and texts that would be embedded.
- Real-run check of 17d (2026-10-30): `sync -DryRun` on the real corpus loaded 1,884 sources and cut 22,675 passages, of which 22,362 texts would be embedded (313 passages share a text with another).

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
9a. The store is in write-ahead mode; a search runs and returns while a sync transaction is open; a write through the retriever's read-only connection fails.
10. **End-to-end:** with the real setup, sync the real corpus, then score the 137 frozen passage-level questions through `Retriever.search`: right first and top 3 must equal the stage 4 HR result for jina/llamaindex (100 and 121), and must be repeatable. A difference stops the unit.
11. Mutation check over the store, sync and retriever logic.

## 8. Units
17a store, 17b sync, 17c retriever, 17d commands and owner's script, 17e end-to-end check and report section 10.
