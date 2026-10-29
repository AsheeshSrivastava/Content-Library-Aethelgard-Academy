"""Check question files against the loaded library (spec 12). Usage:

    uv run python scripts/check_questions.py --questions eval/questions --register DIR --snapshots DIR

Exit code 0 = no problems, 1 = problems found, 2 = bad input.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ragcore.question_check import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
