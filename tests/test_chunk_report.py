"""Spec 09: the chunking report for a snapshot. Written before the code."""
import pytest

from ragcore.chunk_report import chunk_snapshot, main
from ragcore.chunking import ChunkConfig
from ragcore.gate import LicenceRules
from ragcore.register import Licence, SourceRecord, save_record

PARA = "A sentence about retrieval and passages that is long enough to matter. " * 4


def record(status="approved"):
    return SourceRecord(
        id="s1", title="T", owner="O", url="https://example.org", kind="docs", area="python",
        licence=Licence(id="MIT", file_url="https://example.org/LICENSE", covers="both", basis="detected"),
        version_ref="v1", fetched=None, content_hash="h", status=status)


def setup(tmp_path, files, status="approved"):
    library = tmp_path / "library"
    (library / "sources").mkdir(parents=True)
    (library / "licence-rules.yaml").write_text("allowed: [MIT]\n", encoding="utf-8")
    save_record(record(status), library / "sources" / "s1.yaml")
    snap = tmp_path / "snapshots" / "s1" / "v1"
    for name, text in files.items():
        (snap / name).parent.mkdir(parents=True, exist_ok=True)
        (snap / name).write_text(text, encoding="utf-8")
    return library, tmp_path / "snapshots", snap


def rules(tmp_path):
    return LicenceRules.load(tmp_path / "library" / "licence-rules.yaml")


def test_stats_for_a_small_snapshot(tmp_path):
    files = {"a.md": f"# A\n\n{PARA}\n\n## B\n\n{PARA}\n", "b.rst": f"Title\n=====\n\n{PARA}\n", "tiny.md": "# x\n"}
    _, _, snap = setup(tmp_path, files)
    stats = chunk_snapshot(snap, record(), rules(tmp_path), ChunkConfig(max_chars=500), max_skip_share=0.5)
    assert stats.sources == 2 and stats.skipped == {"tiny.md": "stub"}
    assert stats.passages == 3 and stats.dropped_empty == 0
    assert stats.failures == []
    assert stats.under_min == 0 and stats.over_max == 0
    assert 200 < stats.median_chars < 400 and stats.p90_chars >= stats.median_chars


def test_counts_passages_over_max_chars_and_under_the_minimum(tmp_path):
    files = {"a.md": f"# A\n\n{PARA}{PARA}{PARA}\n\n# B\n\nshort one.\n"}
    _, _, snap = setup(tmp_path, files)
    stats = chunk_snapshot(snap, record(), rules(tmp_path), ChunkConfig(max_chars=100, hard_limit=6000))
    assert stats.over_max == 1 and stats.under_min == 1


def test_main_prints_and_exits_zero_when_all_checks_pass(tmp_path, capsys):
    library, snaps, _ = setup(tmp_path, {"a.md": f"# A\n\n{PARA}\n"})
    code = main(["--source", "s1", "--register", str(library / "sources"), "--snapshots", str(snaps)])
    out = capsys.readouterr().out
    assert code == 0 and "s1" in out and "passages 1" in out


def test_main_exit_codes_for_bad_input(tmp_path):
    assert main([]) == 2
    library, snaps, _ = setup(tmp_path, {"a.md": PARA}, status="quarantine")
    assert main(["--source", "s1", "--register", str(library / "sources"), "--snapshots", str(snaps)]) == 2
    assert main(["--source", "nope", "--register", str(library / "sources"), "--snapshots", str(snaps)]) == 2


def test_main_exits_one_when_a_chunker_check_fails(tmp_path, monkeypatch):
    library, snaps, _ = setup(tmp_path, {"a.md": f"# A\n\n{PARA}\n"})
    import ragcore.chunk_report as report
    from ragcore.chunking import ChunkingError

    def broken(source, config):
        raise ChunkingError("bad chunker")
    monkeypatch.setattr(report, "chunk_source", broken)
    assert main(["--source", "s1", "--register", str(library / "sources"), "--snapshots", str(snaps)]) == 1


def test_a_narrow_console_does_not_stop_the_run(tmp_path, monkeypatch):
    import io
    import sys
    library, snaps, _ = setup(tmp_path, {"a.md": f"# A\n\n{PARA}\n"})
    monkeypatch.setattr(sys, "stdout", io.TextIOWrapper(io.BytesIO(), encoding="cp1252", errors="strict"))
    assert main(["--source", "s1", "--register", str(library / "sources"), "--snapshots", str(snaps)]) == 0


@pytest.mark.parametrize("flag", ["--max-chars", "--hard-limit"])
def test_size_flags_change_the_chunker_version(tmp_path, capsys, flag):
    library, snaps, _ = setup(tmp_path, {"a.md": f"# A\n\n{PARA}\n"})
    main(["--source", "s1", "--register", str(library / "sources"), "--snapshots", str(snaps), flag, "2000"])
    assert "heading-pack-1/" in capsys.readouterr().out
