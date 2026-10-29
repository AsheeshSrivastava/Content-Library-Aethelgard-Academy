"""Chunk every loaded source of a snapshot, run the checks and report sizes (spec 09). Nothing is embedded."""
from __future__ import annotations

import argparse
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path

from ragcore.chunking import (ChunkConfig, ChunkingError, ChunkResult, chunk_fixed, chunk_merge_small,
                              chunk_source)
from ragcore.errors import LoadRefused, RegisterError, SkipShareExceeded
from ragcore.files import MIN_CHARS
from ragcore.gate import LicenceRules
from ragcore.loaders.snapshot import load_snapshot
from ragcore.register import SourceRecord, load_record


@dataclass
class ChunkStats:
    source_id: str
    chunker_version: str
    sources: int = 0
    passages: int = 0
    dropped_empty: int = 0
    median_chars: float = 0.0
    p90_chars: int = 0
    under_min: int = 0
    over_max: int = 0
    skipped: dict = field(default_factory=dict)
    failures: list = field(default_factory=list)

    def render(self) -> str:
        lines = [f"{self.source_id}: sources {self.sources}, passages {self.passages}, dropped {self.dropped_empty}, "
                 f"median {self.median_chars:.0f} chars, p90 {self.p90_chars}, under {MIN_CHARS} chars {self.under_min}, "
                 f"over max {self.over_max}, skipped files {len(self.skipped)} [{self.chunker_version}]"]
        lines += [f"  CHECK FAILED {sid}: {msg}" for sid, msg in self.failures]
        return "\n".join(lines)


CHUNKERS = ("rules", "fixed", "merge-small")


def _chunk(source, config: ChunkConfig, chunker: str, min_chars: int) -> ChunkResult:
    if chunker == "fixed":
        return chunk_fixed(source, config)
    if chunker == "merge-small":
        return chunk_merge_small(source, config, min_chars)
    return chunk_source(source, config)


def chunk_snapshot(folder: Path, record: SourceRecord, rules: LicenceRules, config: ChunkConfig,
                   max_skip_share: float = 0.25, chunker: str = "rules", min_chars: int = 300) -> ChunkStats:
    if chunker not in CHUNKERS:
        raise ValueError(f"unknown chunker {chunker!r}; choose one of {', '.join(CHUNKERS)}")
    loaded = load_snapshot(folder, record, rules, max_skip_share=max_skip_share)
    stats = ChunkStats(record.id, config.version, sources=len(loaded.sources), skipped=loaded.skipped)
    sizes: list[int] = []
    for source in loaded.sources:
        try:
            result = _chunk(source, config, chunker, min_chars)
        except ChunkingError as exc:
            stats.failures.append((source.source_id, str(exc)))
            continue
        stats.dropped_empty += result.dropped_empty
        if result.passages:
            stats.chunker_version = result.passages[0].chunker_version
        sizes += [len(p.text) for p in result.passages]
    stats.passages = len(sizes)
    if sizes:
        ordered = sorted(sizes)
        stats.median_chars = statistics.median(ordered)
        stats.p90_chars = ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))]
    stats.under_min = sum(1 for n in sizes if n < MIN_CHARS)
    stats.over_max = sum(1 for n in sizes if n > config.max_chars)
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Chunk a loaded snapshot and run the checks (spec 09).")
    parser.add_argument("--source", help="source id")
    parser.add_argument("--register", help="folder of register records (yaml)")
    parser.add_argument("--snapshots", help="folder that holds <id>/<version>/ snapshots")
    parser.add_argument("--rules", help="licence rules file (default: licence-rules.yaml beside the register folder)")
    parser.add_argument("--max-chars", type=int, default=ChunkConfig.max_chars)
    parser.add_argument("--hard-limit", type=int, default=ChunkConfig.hard_limit)
    parser.add_argument("--max-skip-share", type=float, default=0.25)
    parser.add_argument("--chunker", choices=CHUNKERS, default="rules")
    parser.add_argument("--min-chars", type=int, default=300, help="merge-small only")
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code or 0)
    if not (args.source and args.register and args.snapshots):
        print("Give --source, --register and --snapshots.", file=sys.stderr)
        return 2
    try:
        config = ChunkConfig(args.max_chars, args.hard_limit)
        record = load_record(Path(args.register) / f"{args.source}.yaml")
        rules = LicenceRules.load(Path(args.rules) if args.rules else Path(args.register).parent / "licence-rules.yaml")
        folder = Path(args.snapshots) / record.id / record.version_ref
        if not folder.is_dir():
            print(f"Not a folder: {folder}", file=sys.stderr)
            return 2
        stats = chunk_snapshot(folder, record, rules, config, args.max_skip_share, args.chunker, args.min_chars)
    except (RegisterError, LoadRefused, OSError, ValueError) as exc:
        print(f"Cannot chunk {args.source}: {exc}", file=sys.stderr)
        return 2
    except SkipShareExceeded as exc:
        print(f"Cannot chunk {args.source}: {exc}", file=sys.stderr)
        return 1
    if getattr(sys.stdout, "encoding", None) and sys.stdout.encoding.lower().replace("-", "") != "utf8":
        sys.stdout.reconfigure(errors="replace")  # A narrow console must not stop the run.
    print(stats.render())
    return 1 if stats.failures else 0
