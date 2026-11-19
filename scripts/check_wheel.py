"""Spec 19 test 11: build the wheel, check the pins are inside, install it in a clean venv with only the core
dependencies, and check the public surface there. Run: `uv run python scripts/check_wheel.py`."""
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHECK = """
import sys, ragcore
assert set(ragcore.__all__) == {'Retriever','Result','IndexSetup','open_index','IndexStoreError','load_models','__version__'}
assert not [m for m in ('torch','transformers','llama_index') if m in sys.modules]
try:
    ragcore.load_models('fake')
except ValueError as exc:
    assert 'jina-v2-base-en' in str(exc)
else:
    raise SystemExit('unknown id was accepted')
try:
    ragcore.load_models()
except ImportError as exc:
    assert 'ragcore[embed]' in str(exc)
else:
    raise SystemExit('expected ImportError without torch')
print('wheel ok', ragcore.__version__)
"""


def run(*cmd):
    subprocess.run([str(c) for c in cmd], check=True)


with tempfile.TemporaryDirectory() as tmp:
    out = Path(tmp) / "dist"
    run("uv", "build", "--wheel", "--out-dir", out, ROOT)
    wheel = next(out.glob("ragcore-*.whl"))
    names = zipfile.ZipFile(wheel).namelist()
    for needed in ("ragcore/pins/models.yaml", "ragcore/pins/rerankers.yaml"):
        assert needed in names, f"{needed} is missing from {wheel.name}"
    venv = Path(tmp) / "venv"
    run("uv", "venv", "--python", f"{sys.version_info.major}.{sys.version_info.minor}", venv)
    py = venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    run("uv", "pip", "install", "--python", py, wheel)
    run(py, "-I", "-c", CHECK)
