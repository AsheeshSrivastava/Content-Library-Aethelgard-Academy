"""Write the random quote candidates for the passage-level questions (spec 16 section 3). Usage:

    uv run python scripts/run_pick_quotes.py --register DIR --snapshots DIR --count 400 --seed 20261030 \
        --out eval/passage-candidates/candidates.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from ragcore.errors import RegisterError
from ragcore.quote_picker import group_of, pick_candidates
from ragcore.stage import _csv, _exclusions, _safe_print, load_corpus

DEFAULT_COUNT, DEFAULT_CAP, DEFAULT_SEED = 400, 25, 20261030


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pick random quote places for the passage-level questions.")
    parser.add_argument("--register")
    parser.add_argument("--snapshots")
    parser.add_argument("--rules")
    parser.add_argument("--exclusions", default="", help="corpus exclusions file (default: configs/corpus-exclusions.yaml)")
    parser.add_argument("--only", default="")
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT, help="how many candidates to draw (a pool larger than the 150 needed)")
    parser.add_argument("--cap", type=int, default=DEFAULT_CAP, help="most candidates from one source")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out", default="")
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code or 0)
    if not (args.register and args.snapshots and args.out):
        print("Give --register, --snapshots and --out.", file=sys.stderr)
        return 2
    try:
        if args.count < 1 or args.cap < 1:
            raise ValueError("--count and --cap must be at least 1")
        sources, notes = load_corpus(Path(args.register), Path(args.snapshots), Path(args.rules) if args.rules else None,
                                     _csv(args.only) or None, _exclusions(args.exclusions))
        if not sources:
            raise ValueError("no sources loaded: " + "; ".join(notes))
    except (ValueError, RegisterError, OSError) as exc:
        print(f"Cannot start: {exc}", file=sys.stderr)
        return 2
    picks = pick_candidates({s.source_id: s.text for s in sources}, args.count, args.seed, args.cap)
    data = {"seed": args.seed, "count": len(picks),
            "candidates": [{"source": c.source_id, "start": c.start, "end": c.end, "hint": c.hint, "heading": c.heading, "quote": c.quote}
                           for c in picks]}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=10_000), encoding="utf-8")
    _safe_print(f"{len(picks)} candidates from {len({group_of(c.source_id) for c in picks})} register sources written to {out}")
    return 0
