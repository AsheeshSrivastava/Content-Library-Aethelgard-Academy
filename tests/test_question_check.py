"""Spec 12: the question check command. Written before the code."""
from ragcore.question_check import main
from ragcore.register import Licence, SourceRecord, save_record

PARA = "A sentence about retrieval and passages that is long enough to matter. " * 4


FILES = {"install.md": PARA, **{f"p{i}.md": PARA for i in range(5)}}


def record(rid="s1", status="approved"):
    return SourceRecord(
        id=rid, title="T", owner="O", url="https://example.org", kind="docs", area="python",
        licence=Licence(id="MIT", file_url="https://example.org/LICENSE", covers="both", basis="detected"),
        version_ref="v1", fetched=None, content_hash="h", status=status)


def setup(tmp_path, questions, sources=(("s1", "approved", FILES),)):
    library = tmp_path / "library"
    (library / "sources").mkdir(parents=True)
    (library / "licence-rules.yaml").write_text("allowed: [MIT]\n", encoding="utf-8")
    for rid, status, files in sources:
        save_record(record(rid, status), library / "sources" / f"{rid}.yaml")
        if files is not None:
            snap = tmp_path / "snapshots" / rid / "v1"
            for name, text in files.items():
                (snap / name).parent.mkdir(parents=True, exist_ok=True)
                (snap / name).write_text(text, encoding="utf-8")
    qdir = tmp_path / "questions"
    qdir.mkdir()
    (qdir / "set.yaml").write_text(questions, encoding="utf-8")
    return ["--questions", str(qdir), "--register", str(library / "sources"), "--snapshots", str(tmp_path / "snapshots")]


def test_sound_questions_exit_zero(tmp_path, capsys):
    args = setup(tmp_path, "questions:\n- {q: 'How?', expect: [install]}\noutside: ['Weather?']\n")
    assert main(args) == 0
    out = capsys.readouterr().out
    assert "1 question files, 1 questions, 1 outside" in out and "6 sources loaded" in out and "no problems" in out


def test_problems_exit_one_and_are_listed(tmp_path, capsys):
    args = setup(tmp_path, "questions:\n- {q: 'How?', expect: [install, nothere]}\n")
    assert main(args) == 1
    out = capsys.readouterr().out
    assert "no-match" in out and "nothere" in out and "set" in out


def test_absent_and_refused_snapshots_are_named(tmp_path, capsys):
    args = setup(tmp_path, "questions:\n- {q: 'How?', expect: [install]}\n", sources=(
        ("s1", "approved", FILES), ("gone", "approved", None), ("held", "quarantine", {"a.md": PARA})))
    assert main(args) == 0
    out = capsys.readouterr().out
    assert "gone: not on disk" in out and "held: not loaded" in out and "6 sources loaded" in out


def test_bad_input_exits_two(tmp_path, capsys):
    assert main([]) == 2
    args = setup(tmp_path, "nonsense")
    assert main(args) == 2
    assert "Cannot check" in capsys.readouterr().err


def test_missing_question_folder_exits_two(tmp_path):
    args = setup(tmp_path, "questions:\n- {q: 'How?', expect: [install]}\n")
    args[1] = str(tmp_path / "nope")
    assert main(args) == 2


def test_broad_share_can_be_set(tmp_path, capsys):
    args = setup(tmp_path, "questions:\n- {q: 'How?', expect: [md]}\n")
    assert main(args) == 1 and "broad" in capsys.readouterr().out
    assert main(args + ["--broad-share", "1.0"]) == 0


def test_a_snapshot_with_too_many_skipped_files_is_named_not_loaded(tmp_path, capsys):
    stubs = {"ok.md": PARA, **{f"s{i}.md": "# x\n" for i in range(3)}}
    args = setup(tmp_path, "questions:\n- {q: 'How?', expect: [install]}\n", sources=(
        ("s1", "approved", FILES), ("thin", "approved", stubs)))
    main(args)
    assert "thin: not loaded" in capsys.readouterr().out
