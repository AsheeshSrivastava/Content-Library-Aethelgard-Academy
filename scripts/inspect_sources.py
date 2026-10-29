"""Inspect a pinned snapshot folder (spec 05). Usage:

    uv run python scripts/inspect_sources.py <snapshot-folder> [--id ID] [--out reports]
    uv run python scripts/inspect_sources.py --source ID --register DIR --snapshots DIR
    uv run python scripts/inspect_sources.py <folder> --loaders real --compare

Exit code 0 = gate passed, 1 = gate failed (report still written), 2 = bad input.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ragcore.inspect_sources import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
