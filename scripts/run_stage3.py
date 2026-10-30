"""Estimate, run and report stage 3 (hybrid search and reranking, spec 15). Usage:

    uv run --group embed python scripts/run_stage3.py estimate --register DIR --snapshots DIR --questions eval/questions
    uv run --group embed python scripts/run_stage3.py run ...      (same arguments; saves per-question results)
    uv run --group embed python scripts/run_stage3.py report ...   (same arguments; prints the paired tables)

Always run `estimate` first and read it. Exit code 0 = done, 1 = could not continue, 2 = bad input.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ragcore.stage3 import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
