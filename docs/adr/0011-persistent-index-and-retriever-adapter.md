# ADR 0011: One SQLite index file, synced automatically, with a small retriever adapter

## Status: Accepted

## Context
The comparison ended (ADR 0010): jina-v2-base-en, the `llamaindex` chunker, hybrid search and the local reranker. So far every result came from a "run folder" built for an experiment (ADR 0007): vectors without text, rebuilt from the corpus each time, never updated in place. The real library needs three things the run folders do not give: the index must **survive restarts**, it must **update by itself** when a source changes (only what changed), and callers (the Aethelgard app later) need **one small, stable way to ask for passages** without knowing how search works inside. This is the "before the store" decision in `docs/adr/README.md`.

## Decision
1. **One SQLite file** (Python standard library, no new dependency) at `~/.ragcore/index/index.db` (path can be set). It holds: `meta` (schema version and the exact embedder, chunker, reranker and their pins), `sources`, `passages` (with their text, heading path and position), and `vectors` (float32 blobs, in their own table keyed by passage and embedder, so a model change touches only vectors).
2. **Sync is one repeatable command** (`ragcore index sync`). It reads the approved register, the pinned snapshots and the exclusions file, and compares each source's content hash and chunker version with what is stored. Unchanged sources are left alone; changed sources are replaced; removed, unapproved or excluded sources are deleted. Only new passages are embedded. Each source is replaced in **one transaction**, so a power cut leaves a valid index and the next sync finishes the job.
3. **One pinned setup per index file.** The file records the embedder, chunker and reranker it was built with. Opening it with a different setup is refused with a message ("rebuild the index"); two models are never mixed in one file. The schema has a version number; a file with a newer version is refused, and an older one is copied to a backup file before it is upgraded.
4. **Retrieval loads the index into memory** at start: the vectors as one array and the keyword index built from the stored texts (seconds at 22,000 passages, about 70 MB of vectors). Search is exactly ADR 0009 (vector 200 + keywords 200, rank fusion, reranker on the first 30).
5. **The adapter** is a small class `Retriever.search(query, k=5)` that returns a list of results: passage id, source id, source URL, licence, heading path, passage text, position in the source, rank and score. It does not import from Aethelgard; the app calls it. By default it returns passages (not collapsed to files); an option limits results per file.
6. **"Automatic" means one idempotent command plus a status command**, not a background service. The owner can run it by hand or schedule it with Windows Task Scheduler later.

## Why
At about 22,000 passages, memory holds everything the search needs; a vector database or search server would add a dependency, a process and a failure mode for no gain at this size. SQLite gives atomic updates, one file to back up, and nothing to install. Keeping vectors in their own table and recording the setup in `meta` is what makes a later model change safe and cheap to reason about.

## Alternatives
- **Chroma, Qdrant, pgvector, sqlite-vec, FAISS:** useful past a few hundred thousand passages or for many users; extra dependency now. The adapter hides the store, so we can swap later.
- **Keep run folders as the index:** no incremental update, no text stored, rebuilt from the corpus every time.
- **A background watcher or service:** more moving parts than one repeatable command; revisit when the library lives in the cloud app.
- **Store text only in the source snapshots:** saves space but makes every search depend on matching snapshots; storing the passage text keeps results self-contained and citable.

## Trade-offs accepted
- Everything is loaded into memory, so this does not scale to millions of passages; revisit then.
- Search uses one process on one machine; no concurrent writers (sync takes a lock and a second sync stops with a clear message).
- Passage text is stored twice (snapshot and index); the snapshot stays the source of truth and sync can always rebuild the index from it.
