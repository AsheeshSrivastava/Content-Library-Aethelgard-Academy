"""Check the question files against the loaded library (spec 12). Nothing is embedded or searched."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ragcore.errors import LoadRefused, RegisterError, SkipShareExceeded
from ragcore.evalset import check_questions, load_question_dir
from ragcore.gate import LicenceRules
from ragcore.loaders.snapshot import load_snapshot
from ragcore.register import load_record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check question files against the loaded library (spec 12).")
    parser.add_argument("--questions", help="folder of question files (yaml)")
    parser.add_argument("--register", help="folder of register records (yaml)")
    parser.add_argument("--snapshots", help="folder that holds <id>/<version>/ snapshots")
    parser.add_argument("--rules", help="licence rules file (default: licence-rules.yaml beside the register folder)")
    parser.add_argument("--broad-share", type=float, default=0.25)
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code or 0)
    if not (args.questions and args.register and args.snapshots):
        print("Give --questions, --register and --snapshots.", file=sys.stderr)
        return 2
    register = Path(args.register)
    notes: list[str] = []
    source_ids: list[str] = []
    try:
        sets = load_question_dir(Path(args.questions))
        rules = LicenceRules.load(Path(args.rules) if args.rules else register.parent / "licence-rules.yaml")
        for path in sorted(register.glob("*.yaml")):
            record = load_record(path)
            folder = Path(args.snapshots) / record.id / record.version_ref
            if not folder.is_dir():
                notes.append(f"{record.id}: not on disk")
                continue
            try:
                source_ids += [s.source_id for s in load_snapshot(folder, record, rules).sources]
            except (LoadRefused, SkipShareExceeded) as exc:
                notes.append(f"{record.id}: not loaded ({exc})")
    except (RegisterError, OSError, ValueError) as exc:
        print(f"Cannot check the questions: {exc}", file=sys.stderr)
        return 2
    if getattr(sys.stdout, "encoding", None) and sys.stdout.encoding.lower().replace("-", "") != "utf8":
        sys.stdout.reconfigure(errors="replace")  # A narrow console must not stop the run.
    problems = check_questions(sets, source_ids, args.broad_share)
    print(f"{len(sets)} question files, {sum(len(s.questions) for s in sets)} questions, "
          f"{sum(len(s.outside) for s in sets)} outside; {len(source_ids)} sources loaded.")
    for note in notes:
        print(f"  {note}")
    for p in problems:
        print(f"  {p.kind} [{p.set}] {p.message}")
    print(f"{len(problems)} problems." if problems else "no problems.")
    return 1 if problems else 0
