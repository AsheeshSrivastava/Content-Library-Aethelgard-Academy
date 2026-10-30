"""The pick-quotes command: loads the corpus as the stages do and writes the candidate file (spec 16 section 3)."""
import yaml

from ragcore.pick_quotes import main
from test_chunk_report import PARA as DISK_PARA, setup


def corpus(tmp_path):
    files = {f"{n}.md": "\n\n".join(f"# {n} {i}\n\nPart {i} of {n}. {DISK_PARA}" for i in range(12)) for n in ("a", "b", "c")}
    library, snaps, _ = setup(tmp_path, files)
    return ["--register", str(library / "sources"), "--snapshots", str(snaps), "--out", str(tmp_path / "cand.yaml")]


def test_it_writes_candidates_that_are_slices_of_the_sources(tmp_path, capsys):
    args = corpus(tmp_path)
    assert main([*args, "--count", "9", "--seed", "5", "--cap", "9"]) == 0
    data = yaml.safe_load((tmp_path / "cand.yaml").read_text(encoding="utf-8"))
    assert data["seed"] == 5 and data["count"] == len(data["candidates"]) > 0
    first = data["candidates"][0]
    assert set(first) == {"source", "start", "end", "hint", "heading", "quote"} and first["heading"].startswith("# ") is False
    assert first["end"] - first["start"] == len(first["quote"])
    assert f"{data['count']} candidates from" in capsys.readouterr().out


def test_the_same_seed_writes_the_same_file(tmp_path):
    args = corpus(tmp_path)
    assert main([*args, "--count", "9", "--seed", "5"]) == 0
    one = (tmp_path / "cand.yaml").read_text(encoding="utf-8")
    assert main([*args, "--count", "9", "--seed", "5"]) == 0
    assert one == (tmp_path / "cand.yaml").read_text(encoding="utf-8")
    assert main([*args, "--count", "9", "--seed", "6"]) == 0
    assert one != (tmp_path / "cand.yaml").read_text(encoding="utf-8")


def test_bad_input(tmp_path, capsys):
    assert main([]) == 2
    args = corpus(tmp_path)
    assert main([*args, "--count", "0"]) == 2
    assert main([*args, "--only", "nope"]) == 2
    assert "Cannot start" in capsys.readouterr().err
