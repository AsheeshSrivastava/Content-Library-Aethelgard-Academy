"""Filling and updating the index from loaded sources (spec 17 section 3).

`sync` makes the index match a list of loaded sources. It is safe to run again at any time: a source whose version,
text, chunker, url and licence are unchanged is skipped, a changed one is replaced in one transaction, and a
source that is no longer in the list is removed. Text that is already in the index is never embedded again."""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

from ragcore.chunking import ChunkResult
from ragcore.contracts import Source
from ragcore.index_store import IndexStore, IndexStoreError, PassageRow, SourceRow

DEFAULT_BATCH = 32


class SyncError(Exception):
    """The sync cannot go ahead as asked; what was already stored is still a valid index."""


@dataclass(frozen=True)
class SyncReport:
    sources_added: int
    sources_replaced: int
    sources_unchanged: int
    sources_deleted: int
    passages_added: int
    passages_reused: int
    passages_deleted: int
    vectors_embedded: int
    seconds: float
    dry_run: bool

    def render(self) -> str:
        head = "Dry run, nothing was changed. Would have done:" if self.dry_run else "Sync finished."
        return "\n".join([
            head,
            f"  {self.sources_added} sources added, {self.sources_replaced} replaced, "
            f"{self.sources_unchanged} sources unchanged, {self.sources_deleted} removed",
            f"  {self.passages_added} passages stored ({self.passages_reused} reused an existing vector), "
            f"{self.passages_deleted} old passages removed",
            (f"  {self.vectors_embedded} texts would need embedding (cutting the passages took {self.seconds:.1f} seconds)"
             if self.dry_run else f"  {self.vectors_embedded} texts embedded in {self.seconds:.1f} seconds"),
        ])


def sync(sources: list[Source], store: IndexStore, chunk: Callable[[Source], ChunkResult], embedder, *,
         urls: dict[str, str] | None = None, batch: int = DEFAULT_BATCH, dry_run: bool = False,
         allow_mass_delete: bool = False, progress: Callable[[int, int, str], None] | None = None,
         loaded_at: str | None = None, clock: Callable[[], float] = time.monotonic) -> SyncReport:
    """Bring the index in line with `sources`. `urls` maps the first part of a source id (its register entry) to a url."""
    if store.read_only and not dry_run:
        raise IndexStoreError("this index was opened read-only; open it for writing to sync")
    started = clock()
    urls = urls or {}
    loaded_at = loaded_at or datetime.now(timezone.utc).isoformat()
    ordered = sorted(sources, key=lambda s: s.source_id)
    ids = [s.source_id for s in ordered]
    if len(set(ids)) != len(ids):
        raise SyncError("two sources have the same source id: " + ", ".join(sorted({i for i in ids if ids.count(i) > 1})))
    chunker_version = store.meta()["chunker_version"]
    stored = store.sources()
    removed = [source_id for source_id in stored if source_id not in set(ids)]
    if removed and len(removed) * 2 >= len(stored) and not allow_mass_delete:
        raise SyncError(f"this sync would remove {len(removed)} of {len(stored)} sources; if that is right, run it with "
                        "allow_mass_delete, otherwise check that the right folder was loaded")

    counts = dict(added=0, replaced=0, unchanged=0, passages=0, deleted=0, embedded=0)
    planned: set[str] = set()  # text hashes a dry run would have stored
    for done, source in enumerate(ordered, start=1):
        row = SourceRow(source.source_id, source.version, source.text_sha256, chunker_version,
                        urls.get(source.source_id.split("/")[0], ""), source.licence, loaded_at)
        before = stored.get(source.source_id)
        if before is not None and _same(before, row):
            counts["unchanged"] += 1
        else:
            _store_source(source, row, before, store, chunk, embedder, counts, planned, batch, dry_run)
            counts["replaced" if before is not None else "added"] += 1
        if progress:
            progress(done, len(ordered), source.source_id)

    for source_id in removed:
        counts["deleted"] += store.count_passages(source_id)
        if not dry_run:
            store.delete_source(source_id)
    return SyncReport(counts["added"], counts["replaced"], counts["unchanged"], len(removed), counts["passages"],
                      counts["passages"] - counts["embedded"], counts["deleted"], counts["embedded"],
                      clock() - started, dry_run)


def _same(before: SourceRow, now: SourceRow) -> bool:
    return (before.version, before.content_sha256, before.chunker_version, before.url, before.licence) == \
           (now.version, now.content_sha256, now.chunker_version, now.url, now.licence)


def _store_source(source: Source, row: SourceRow, before: SourceRow | None, store: IndexStore, chunk, embedder,
                  counts: dict[str, int], planned: set[str], batch: int, dry_run: bool) -> None:
    passages = chunk(source).passages
    if before is not None:
        counts["deleted"] += store.count_passages(source.source_id)
    for passage in passages:
        if passage.chunker_version != row.chunker_version:
            raise SyncError(f"the chunker gives version {passage.chunker_version!r} but this index was built with "
                            f"{row.chunker_version!r}; use the same chunker or rebuild the index")
    hashes = [p.content_id for p in passages]
    known = store.find_vectors(hashes)
    fresh: dict[str, str] = {}  # text hash -> text, in first-seen order
    for passage in passages:
        if passage.content_id not in known and passage.content_id not in planned:
            fresh.setdefault(passage.content_id, passage.text)
    vectors = dict(known)
    if not dry_run:
        for i in range(0, len(fresh), batch):
            keys = list(fresh)[i:i + batch]
            out = np.asarray(embedder.embed([fresh[k] for k in keys], kind="passage"), dtype=np.float32)
            if len(out) != len(keys):
                raise SyncError(f"the embedder returned {len(out)} vectors for {len(keys)} texts")
            vectors.update(zip(keys, out))
        items = [(PassageRow(p.passage_id, p.source_id, p.start, p.end, p.headings, p.text, p.content_id),
                  vectors[p.content_id]) for p in passages]
        store.replace_source(row, items)
    else:
        planned.update(hashes)
    counts["passages"] += len(passages)
    counts["embedded"] += len(fresh)
