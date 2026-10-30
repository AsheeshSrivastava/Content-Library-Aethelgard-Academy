"""Sync, inspect and search the index (spec 17). Usage:

    uv run --group embed --group variants python scripts/run_index.py sync --register DIR --snapshots DIR
    uv run --group embed --group variants python scripts/run_index.py status
    uv run --group embed --group variants python scripts/run_index.py search "your question" -k 5

Exit code 0 = done, 1 = could not continue (the index is unchanged or still valid), 2 = bad input.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ragcore.index_cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
