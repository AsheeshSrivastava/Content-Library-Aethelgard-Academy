"""Run folders (spec 13 section 4.4): ids from inputs, atomic write, checked load."""
import json

import numpy as np
import pytest

from ragcore.contracts import Source, Span, make_passages
from ragcore.runs import RunError, compute_run_id, load_run, write_run

INPUTS = {"chunker": "rules", "embedder": "fake|dim=4", "mode": "normal", "corpus": ["a@1"], "questions": "abc"}


def passages(n=3, source_id="docs/a"):
    text = "\n\n".join(f"paragraph number {i}" for i in range(n))
    source = Source(source_id, "v1" * 4, text, "0" * 64, "plain-1", "MIT", "declared")
    spans, pos = [], 0
    for part in text.split("\n\n"):
        start = text.index(part, pos)
        spans.append(Span(start, start + len(part), ("Top",)))
        pos = start + len(part)
    return make_passages(source, spans, chunker_version="rules-1")


def vectors(n=3, dim=4):
    return np.arange(n * dim, dtype=np.float32).reshape(n, dim) + 1.0


def test_run_id_is_stable_and_ignores_key_order():
    a = compute_run_id({"x": 1, "y": [1, 2]})
    assert a == compute_run_id({"y": [1, 2], "x": 1})
    assert len(a) == 16 and all(c in "0123456789abcdef" for c in a)


def test_run_id_changes_with_any_input():
    base = compute_run_id(INPUTS)
    assert compute_run_id({**INPUTS, "chunker": "fixed"}) != base
    assert compute_run_id({**INPUTS, "corpus": ["a@2"]}) != base


def test_write_then_load_round_trip(tmp_path):
    ps, vs = passages(), vectors()
    run_id = write_run(tmp_path, INPUTS, ps, vs, extra={"truncated": 2})
    run = load_run(tmp_path, run_id)
    assert run.run_id == run_id == compute_run_id(INPUTS)
    assert np.array_equal(run.vectors, vs) and run.vectors.dtype == np.float32
    assert [r["passage_id"] for r in run.rows] == [p.passage_id for p in ps]
    assert run.rows[1]["source_id"] == "docs/a" and run.rows[1]["headings"] == ["Top"]
    assert (run.rows[1]["start"], run.rows[1]["end"]) == (ps[1].start, ps[1].end)
    assert run.manifest["extra"] == {"truncated": 2} and run.manifest["count"] == 3 and run.manifest["dim"] == 4
    assert run.manifest["inputs"] == INPUTS


def test_parents_are_stored(tmp_path):
    ps = passages()
    run_id = write_run(tmp_path, INPUTS, ps, vectors(), parents={ps[0].passage_id: "P1", ps[1].passage_id: "P1"})
    rows = load_run(tmp_path, run_id).rows
    assert [r["parent"] for r in rows] == ["P1", "P1", None]


def test_unknown_parent_key_refused(tmp_path):
    with pytest.raises(ValueError, match="parent"):
        write_run(tmp_path, INPUTS, passages(), vectors(), parents={"nope": "P"})


def test_count_mismatch_refused_on_write(tmp_path):
    with pytest.raises(ValueError, match="count"):
        write_run(tmp_path, INPUTS, passages(3), vectors(2))
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("bad", [np.zeros((3, 4), dtype=np.float64), np.zeros(12, dtype=np.float32),
                                 np.full((3, 4), np.nan, dtype=np.float32)])
def test_bad_vectors_refused(tmp_path, bad):
    with pytest.raises(ValueError):
        write_run(tmp_path, INPUTS, passages(3), bad)
    assert list(tmp_path.iterdir()) == []


def test_existing_complete_run_is_not_rewritten(tmp_path):
    ps = passages()
    run_id = write_run(tmp_path, INPUTS, ps, vectors())
    before = (tmp_path / run_id / "vectors.npy").stat().st_mtime_ns
    assert write_run(tmp_path, INPUTS, ps, vectors() * 2) == run_id
    assert (tmp_path / run_id / "vectors.npy").stat().st_mtime_ns == before


def test_stale_temp_folder_does_not_block_and_is_cleaned(tmp_path):
    run_id = compute_run_id(INPUTS)
    stale = tmp_path / f".tmp-{run_id}"
    stale.mkdir()
    (stale / "junk").write_text("x")
    assert write_run(tmp_path, INPUTS, passages(), vectors()) == run_id
    assert sorted(p.name for p in tmp_path.iterdir()) == [run_id]


def test_failed_write_leaves_no_half_run(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise OSError("disk full")
    monkeypatch.setattr(np, "save", boom)
    with pytest.raises(OSError):
        write_run(tmp_path, INPUTS, passages(), vectors())
    assert list(tmp_path.iterdir()) == []


def test_missing_run(tmp_path):
    with pytest.raises(RunError, match="no run"):
        load_run(tmp_path, "0" * 16)


def test_incomplete_folder_refused(tmp_path):
    run_id = write_run(tmp_path, INPUTS, passages(), vectors())
    (tmp_path / run_id / "scores.json").write_text("{}")  # extra files are fine
    (tmp_path / run_id / "vectors.npy").unlink()
    with pytest.raises(RunError, match="vectors.npy"):
        load_run(tmp_path, run_id)


def test_changed_vectors_detected(tmp_path):
    run_id = write_run(tmp_path, INPUTS, passages(), vectors())
    np.save(tmp_path / run_id / "vectors.npy", vectors() + 1)
    with pytest.raises(RunError, match="checksum"):
        load_run(tmp_path, run_id)


def test_row_count_mismatch_detected(tmp_path):
    run_id = write_run(tmp_path, INPUTS, passages(), vectors())
    path = tmp_path / run_id / "passages.jsonl"
    path.write_text("\n".join(path.read_text(encoding="utf-8").splitlines()[:2]) + "\n", encoding="utf-8")
    with pytest.raises(RunError, match="rows"):
        load_run(tmp_path, run_id)


def test_manifest_that_does_not_match_its_folder_name(tmp_path):
    run_id = write_run(tmp_path, INPUTS, passages(), vectors())
    path = tmp_path / run_id / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["inputs"]["chunker"] = "fixed"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RunError, match="inputs"):
        load_run(tmp_path, run_id)


def _edit_count(tmp_path, run_id, count, keep_rows):
    folder = tmp_path / run_id
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    manifest["count"] = count
    (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    lines = (folder / "passages.jsonl").read_text(encoding="utf-8").splitlines()[:keep_rows]
    (folder / "passages.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_rows_and_vectors_must_agree_even_if_manifest_agrees_with_rows(tmp_path):
    run_id = write_run(tmp_path, INPUTS, passages(3), vectors(3))
    _edit_count(tmp_path, run_id, 2, 2)
    with pytest.raises(RunError, match="rows"):
        load_run(tmp_path, run_id)


def test_manifest_count_must_agree_with_rows_and_vectors(tmp_path):
    run_id = write_run(tmp_path, INPUTS, passages(3), vectors(3))
    _edit_count(tmp_path, run_id, 2, 3)
    with pytest.raises(RunError, match="rows"):
        load_run(tmp_path, run_id)
