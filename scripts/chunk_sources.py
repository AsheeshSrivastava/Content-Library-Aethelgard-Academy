"""Chunk a loaded snapshot and run the chunking checks (spec 09). Usage:

    uv run python scripts/chunk_sources.py --source ID --register DIR --snapshots DIR [--max-chars 1500]

Exit code 0 = all checks passed, 1 = a check failed or too many files were skipped, 2 = bad input.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ragcore.chunk_report import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
