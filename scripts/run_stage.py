"""Estimate, build and report a stage of the chunker and embedder comparison (spec 13 section 6). Usage:

    uv run --group embed python scripts/run_stage.py estimate --register DIR --snapshots DIR --questions eval/questions \
        --embedders bge-m3 --only python-peps
    uv run --group embed python scripts/run_stage.py run ...      (same arguments; builds the runs)
    uv run --group embed python scripts/run_stage.py report ...   (same arguments; prints the paired tables)

Always run `estimate` first and read it. Exit code 0 = done, 1 = report could not be made, 2 = bad input.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ragcore.stage import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
