"""Write the random quote candidates for the passage-level questions (spec 16). See ragcore/pick_quotes.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ragcore.pick_quotes import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
