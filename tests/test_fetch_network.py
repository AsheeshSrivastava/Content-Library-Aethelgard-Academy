"""Spec 03 network acceptance test: re-fetch approved sources at their recorded SHA and compare hashes.

Not part of the default run. The owner runs it, for example in PowerShell:

    $env:RAGCORE_LIB3 = "D:\\01-Project\\unsloth\\aethelgard-academy\\.worktrees\\library3\\content\\library"
    $env:RAGCORE_PIN_SOURCES = "pytest-docs,mcp-spec"
    $env:RAGCORE_SNAPSHOTS = "<folder that holds <id>\\<sha>\\ snapshots>"   # optional, adds a stored-copy check
    uv run pytest -m network -q tests/test_fetch_network.py

It only downloads from the hosts in the library3 hosts.yaml. It needs git and the internet, and no API key.
"""
import os
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from ragcore.fetch import fetch_git, snapshot_hash
from ragcore.register import load_record

pytestmark = pytest.mark.network

LIB3 = os.environ.get("RAGCORE_LIB3")
IDS = [i.strip() for i in os.environ.get("RAGCORE_PIN_SOURCES", "").split(",") if i.strip()]
SNAPSHOTS = os.environ.get("RAGCORE_SNAPSHOTS")


@pytest.mark.skipif(not LIB3 or not IDS, reason="set RAGCORE_LIB3 and RAGCORE_PIN_SOURCES")
@pytest.mark.parametrize("source_id", IDS)
def test_refetch_at_pinned_sha_matches_stored_hashes(source_id, tmp_path):
    lib = Path(LIB3)
    hosts = frozenset(yaml.safe_load((lib / "hosts.yaml").read_text(encoding="utf-8"))["hosts"])
    record = load_record(lib / "sources" / f"{source_id}.yaml")
    assert record.fetch and record.fetch.git, f"{source_id} is not a git source"
    assert record.version_ref and record.content_hash, f"{source_id} has no pinned version"
    pinned = replace(record, fetch=replace(record.fetch, commit=record.version_ref))

    result = fetch_git(pinned, tmp_path, hosts)

    assert result.ref == record.version_ref
    assert result.licence_hash == record.licence.file_hash, "licence file differs from the stored hash"
    assert result.content_hash == record.content_hash, "content differs from the stored hash"
    if SNAPSHOTS and (Path(SNAPSHOTS) / source_id / record.version_ref).is_dir():
        stored = snapshot_hash(Path(SNAPSHOTS) / source_id / record.version_ref)
        assert stored == result.content_hash, "the stored snapshot folder differs from the fresh copy"
