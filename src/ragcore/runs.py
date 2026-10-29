"""Run folders (spec 13 section 4.4): one folder per run, named by a hash of its inputs, written all at once.

A run holds `manifest.json`, `passages.jsonl` (one row per vector, no text: the text is the source slice) and
`vectors.npy` (float32). It is built in a temporary folder and renamed when complete."""
from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ragcore.contracts import Passage

RUN_VERSION = 1
_FILES = ("manifest.json", "passages.jsonl", "vectors.npy")


class RunError(Exception):
    """A run folder is missing, incomplete or does not match its own manifest."""


@dataclass
class Run:
    run_id: str
    manifest: dict
    rows: list
    vectors: np.ndarray


def _canonical(data: object) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def compute_run_id(inputs: dict) -> str:
    return hashlib.sha256(_canonical(inputs).encode("utf-8")).hexdigest()[:16]


def _vectors_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _check_vectors(vectors: np.ndarray, count: int) -> None:
    if vectors.ndim != 2 or vectors.dtype != np.float32:
        raise ValueError("vectors must be a 2-D float32 array")
    if vectors.shape[0] != count:
        raise ValueError(f"count mismatch: {count} passages but {vectors.shape[0]} vectors")
    if not np.isfinite(vectors).all():
        raise ValueError("vectors contain NaN or infinity")


def write_run(root: Path, inputs: dict, passages: list[Passage], vectors: np.ndarray, *,
              parents: dict | None = None, extra: dict | None = None) -> str:
    """Write a complete run and return its id. A complete run with the same inputs is left as it is."""
    root = Path(root)
    _check_vectors(vectors, len(passages))
    parents = parents or {}
    unknown = set(parents) - {p.passage_id for p in passages}
    if unknown:
        raise ValueError(f"parent map names passages that are not in the run: {sorted(unknown)[:3]}")
    run_id = compute_run_id(inputs)
    final, temp = root / run_id, root / f".tmp-{run_id}"
    try:
        load_run(root, run_id)
        return run_id
    except RunError:
        pass
    root.mkdir(parents=True, exist_ok=True)
    for leftover in (temp, final):
        if leftover.exists():
            shutil.rmtree(leftover)
    temp.mkdir()
    try:
        np.save(temp / "vectors.npy", vectors)
        rows = [{"passage_id": p.passage_id, "source_id": p.source_id, "start": p.start, "end": p.end,
                 "headings": list(p.headings), "parent": parents.get(p.passage_id)} for p in passages]
        (temp / "passages.jsonl").write_text("".join(_canonical(r) + "\n" for r in rows), encoding="utf-8")
        manifest = {"run_version": RUN_VERSION, "run_id": run_id, "inputs": inputs, "count": len(passages),
                    "dim": int(vectors.shape[1]), "vectors_sha256": _vectors_sha256(temp / "vectors.npy"),
                    "extra": extra or {}}
        (temp / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        temp.rename(final)
    except BaseException:
        shutil.rmtree(temp, ignore_errors=True)
        raise
    return run_id


def load_run(root: Path, run_id: str) -> Run:
    folder = Path(root) / run_id
    if not folder.is_dir():
        raise RunError(f"no run {run_id} in {root}")
    for name in _FILES:
        if not (folder / name).is_file():
            raise RunError(f"run {run_id} is incomplete: {name} is missing")
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if compute_run_id(manifest.get("inputs", {})) != run_id:
        raise RunError(f"run {run_id}: the manifest inputs do not match the folder name")
    if _vectors_sha256(folder / "vectors.npy") != manifest.get("vectors_sha256"):
        raise RunError(f"run {run_id}: vectors.npy fails its checksum")
    vectors = np.load(folder / "vectors.npy")
    rows = [json.loads(line) for line in (folder / "passages.jsonl").read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != manifest["count"] or len(rows) != vectors.shape[0]:
        raise RunError(f"run {run_id}: {len(rows)} rows, {vectors.shape[0]} vectors, manifest says {manifest['count']}")
    return Run(run_id, manifest, rows, vectors)
