"""Stage runner (spec 13 section 6): corpus, building runs, the printed estimate, the report. Fake embedder only."""
import numpy as np
import pytest

from ragcore.chunking import ChunkConfig
from ragcore.contracts import Source
from ragcore.embed.base import FakeEmbedder
from ragcore.evalset import load_question_dir
from ragcore.runs import load_run
from ragcore.stage import (CHUNKER_NAMES, build_run, chunk_corpus, chunker_label, corpus_fingerprint, estimate,
                           evaluate_runs, load_corpus, main, questions_fingerprint, render_estimate)

PARA = "Retrieval finds passages that answer a question, and ranking orders them by cosine similarity. "
CONFIG = ChunkConfig(max_chars=400, hard_limit=2000)


def source(name, text, version="v1"):
    return Source(f"s1/{name}", version, text, "0" * 64, "md-1", "MIT", "declared")


def sources():
    return [source("a.md", f"# Alpha\n\n{PARA * 3}\n\n## Beta\n\nBeta passages talk about zebras and quartz.\n"),
            source("b.md", f"# Gamma\n\n{PARA * 2}\n\nGamma explains volcano glass and obsidian knives.\n")]


class Counting(FakeEmbedder):
    def __init__(self):
        super().__init__()
        self.calls = []

    def embed(self, texts, *, kind):
        self.calls.append((kind, len(texts)))
        return super().embed(texts, kind=kind)


def test_chunker_names():
    assert CHUNKER_NAMES == ("rules", "fixed", "merge-small")


def test_labels_name_the_chunker_and_change_with_settings():
    labels = {n: chunker_label(n, CONFIG) for n in CHUNKER_NAMES}
    assert len(set(labels.values())) == 3
    assert chunker_label("rules", CONFIG) != chunker_label("rules", ChunkConfig(max_chars=500, hard_limit=2000))
    assert "400" in labels["fixed"] and "300" in labels["merge-small"]
    for name in CHUNKER_NAMES:
        assert chunker_label(name, CONFIG) != chunker_label(name, ChunkConfig(max_chars=400, hard_limit=3000))


def test_unknown_chunker_refused():
    with pytest.raises(ValueError, match="unknown chunker"):
        chunk_corpus(sources(), "nope", CONFIG)


@pytest.mark.parametrize("name", CHUNKER_NAMES)
def test_chunk_corpus_covers_every_source_in_order(name):
    passages, dropped = chunk_corpus(sources(), name, CONFIG)
    assert passages and dropped >= 0
    assert [p.source_id for p in passages] == sorted(p.source_id for p in passages)
    assert {p.source_id for p in passages} == {"s1/a.md", "s1/b.md"}


def test_chunk_corpus_orders_by_source_id_whatever_the_input_order():
    ordered, _ = chunk_corpus(sources(), "rules", CONFIG)
    flipped, _ = chunk_corpus(list(reversed(sources())), "rules", CONFIG)
    assert [p.passage_id for p in ordered] == [p.passage_id for p in flipped]


def test_corpus_fingerprint_ignores_order_and_sees_changes():
    a, b = sources()
    assert corpus_fingerprint([a, b]) == corpus_fingerprint([b, a])
    assert corpus_fingerprint([a, b]) != corpus_fingerprint([a])
    assert corpus_fingerprint([a, b]) != corpus_fingerprint([a, source("b.md", b.text + "x")])
    assert corpus_fingerprint([a, b]) != corpus_fingerprint([a, source("b.md", b.text, version="v2")])


def test_questions_fingerprint_sees_a_renamed_file(tmp_path):
    (tmp_path / "a.yaml").write_text("questions: []\n", encoding="utf-8")
    before = questions_fingerprint(tmp_path)
    (tmp_path / "a.yaml").rename(tmp_path / "b.yaml")
    assert questions_fingerprint(tmp_path) != before


def test_questions_fingerprint(tmp_path):
    (tmp_path / "a.yaml").write_text("questions: []\n", encoding="utf-8")
    first = questions_fingerprint(tmp_path)
    assert first == questions_fingerprint(tmp_path)
    (tmp_path / "a.yaml").write_text("questions: [] \n", encoding="utf-8")
    assert questions_fingerprint(tmp_path) != first
    (tmp_path / "a.yaml").write_text("questions: []\n", encoding="utf-8")
    (tmp_path / "b.yaml").write_text("questions: []\n", encoding="utf-8")
    assert questions_fingerprint(tmp_path) != first


def test_build_run_vectors_match_passages_in_order(tmp_path):
    emb = Counting()
    built = build_run(tmp_path, "rules", sources(), emb, CONFIG, questions_fp="q", batch=2)
    run = load_run(tmp_path, built.run_id)
    texts = [p.text for p in built.passages]
    assert run.vectors.shape == (len(texts), emb.dim)
    assert np.allclose(run.vectors, FakeEmbedder().embed(texts, kind="passage"))
    assert [r["passage_id"] for r in run.rows] == [p.passage_id for p in built.passages]
    assert all(kind == "passage" and n <= 2 for kind, n in emb.calls) and len(emb.calls) > 1
    assert built.reused is False
    assert run.manifest["inputs"]["chunker"] == chunker_label("rules", CONFIG)
    assert run.manifest["inputs"]["embedder"] == emb.fingerprint
    assert run.manifest["extra"]["dropped_empty"] == built.dropped_empty


def test_second_build_is_reused_without_embedding(tmp_path):
    build_run(tmp_path, "rules", sources(), Counting(), CONFIG, questions_fp="q")
    again = Counting()
    built = build_run(tmp_path, "rules", sources(), again, CONFIG, questions_fp="q")
    assert built.reused is True and again.calls == []


def test_run_id_depends_on_every_input(tmp_path):
    base = build_run(tmp_path, "rules", sources(), FakeEmbedder(), CONFIG, questions_fp="q").run_id
    others = [
        build_run(tmp_path, "fixed", sources(), FakeEmbedder(), CONFIG, questions_fp="q").run_id,
        build_run(tmp_path, "rules", sources(), FakeEmbedder(dim=16), CONFIG, questions_fp="q").run_id,
        build_run(tmp_path, "rules", sources()[:1], FakeEmbedder(), CONFIG, questions_fp="q").run_id,
        build_run(tmp_path, "rules", sources(), FakeEmbedder(), CONFIG, questions_fp="other").run_id,
    ]
    assert len({base, *others}) == 5


def test_truncated_count_is_recorded_when_the_embedder_has_one(tmp_path):
    emb = FakeEmbedder()
    emb.truncated = 7
    built = build_run(tmp_path, "rules", sources(), emb, CONFIG, questions_fp="q")
    assert load_run(tmp_path, built.run_id).manifest["extra"]["truncated"] == 7


def test_a_source_with_nothing_to_embed_is_refused(tmp_path):
    with pytest.raises(ValueError, match="nothing to embed"):
        build_run(tmp_path, "rules", [source("h.md", "# Only a heading\n")], FakeEmbedder(), CONFIG, questions_fp="q")


def test_estimate_counts_and_marks_existing_runs(tmp_path):
    emb = FakeEmbedder()
    build_run(tmp_path, "rules", sources(), emb, CONFIG, questions_fp="q")
    ticks = iter([0.0, 2.0])
    result = estimate(sources(), ["rules", "fixed"], [emb], CONFIG, tmp_path, questions_fp="q", probe_n=2,
                      clock=lambda: next(ticks))
    rules_run, fixed_run = result["runs"]
    expected = chunk_corpus(sources(), "rules", CONFIG)[0]
    assert rules_run["passages"] == len(expected) and rules_run["exists"] is True and fixed_run["exists"] is False
    assert rules_run["tokens"] == sum(emb.count_tokens(p.text) for p in expected)
    assert rules_run["disk_mb"] == pytest.approx(len(expected) * emb.dim * 4 / 1e6, abs=1e-4)
    assert result["probe"]["fake"]["texts"] == 2
    rate = result["probe"]["fake"]["tokens_per_second"]
    assert fixed_run["seconds"] == pytest.approx(fixed_run["tokens"] / rate)
    assert rules_run["seconds"] == 0.0  # already built, will be skipped
    assert result["total_seconds"] == pytest.approx(fixed_run["seconds"])
    assert result["total_runs"] == 2 and result["to_build"] == 1
    assert result["total_disk_mb"] == pytest.approx(fixed_run["disk_mb"])


def test_render_estimate_is_plain_and_says_no_money(tmp_path):
    emb = FakeEmbedder()
    text = render_estimate(estimate(sources(), ["rules"], [emb], CONFIG, tmp_path, questions_fp="q", probe_n=2))
    assert "rules" in text and "fake" in text and "passages" in text and "minutes" in text
    assert "no paid" in text.lower()


def test_estimate_refuses_an_empty_corpus(tmp_path):
    with pytest.raises(ValueError, match="corpus is empty"):
        estimate([], ["rules"], [FakeEmbedder()], CONFIG, tmp_path, questions_fp="q")


# --- corpus on disk ----------------------------------------------------------------------------

import dataclasses  # noqa: E402

from ragcore.register import save_record  # noqa: E402
from test_chunk_report import PARA as DISK_PARA, record, setup  # noqa: E402


def test_load_corpus_reads_the_register_and_filters(tmp_path):
    library, snaps, _ = setup(tmp_path, {"a.md": f"# A\n\n{DISK_PARA}\n", "b.md": f"# B\n\n{DISK_PARA}\n"})
    other = dataclasses.replace(record(), id="s2")
    save_record(other, library / "sources" / "s2.yaml")
    (snaps / "s2" / "v1").mkdir(parents=True)
    (snaps / "s2" / "v1" / "z.md").write_text(f"# Z\n\n{DISK_PARA}\n", encoding="utf-8")
    loaded, notes = load_corpus(library / "sources", snaps)
    assert [s.source_id for s in loaded] == ["s1/a.md", "s1/b.md", "s2/z.md"] and notes == []
    only, _ = load_corpus(library / "sources", snaps, only=["s2"])
    assert [s.source_id for s in only] == ["s2/z.md"]


def test_load_corpus_unknown_only_and_missing_snapshot(tmp_path):
    library, snaps, _ = setup(tmp_path, {"a.md": f"# A\n\n{DISK_PARA}\n"})
    with pytest.raises(ValueError, match="nope"):
        load_corpus(library / "sources", snaps, only=["nope"])
    loaded, notes = load_corpus(library / "sources", tmp_path / "elsewhere")
    assert loaded == [] and any("s1" in n and "not on disk" in n for n in notes)


# --- the report --------------------------------------------------------------------------------

def test_evaluate_runs_finds_the_right_file(tmp_path):
    srcs = sources()
    emb = FakeEmbedder()
    questions = tmp_path / "q"
    questions.mkdir()
    (questions / "a.yaml").write_text(
        "questions:\n  - q: 'Beta passages talk about zebras and quartz.'\n    expect: ['a.md']\n"
        "  - q: 'Gamma explains volcano glass and obsidian knives.'\n    expect: ['b.md']\n", encoding="utf-8")
    runs = tmp_path / "runs"
    ids = {"rules": build_run(runs, "rules", srcs, emb, CONFIG, questions_fp="q").run_id,
           "fixed": build_run(runs, "fixed", srcs, emb, CONFIG, questions_fp="q").run_id}
    results = evaluate_runs(runs, ids, emb, load_question_dir(questions))
    assert set(results) == {"rules", "fixed"}
    assert all(r.first for r in results["rules"]) and len(results["fixed"]) == 2


# --- the command -------------------------------------------------------------------------------

def test_main_estimate_and_run_and_report(tmp_path, capsys):
    library, snaps, _ = setup(tmp_path, {"a.md": f"# A\n\n{DISK_PARA}\n\n## Zed\n\nZebras and quartz.\n",
                                         "b.md": f"# B\n\n{DISK_PARA}\n"})
    questions = tmp_path / "questions"
    questions.mkdir()
    (questions / "a.yaml").write_text("questions:\n  - q: 'Zebras and quartz.'\n    expect: ['a.md']\n", encoding="utf-8")
    common = ["--register", str(library / "sources"), "--snapshots", str(snaps), "--questions", str(questions),
              "--runs", str(tmp_path / "runs"), "--embedders", "fake", "--chunkers", "rules,fixed"]
    assert main(["estimate", *common]) == 0
    assert "No run was written" in capsys.readouterr().out
    assert not (tmp_path / "runs").exists()
    assert main(["run", *common]) == 0
    assert "built" in capsys.readouterr().out
    assert main(["report", *common, "--baseline", "rules"]) == 0
    out = capsys.readouterr().out
    assert "| Run |" in out and "fake/rules" in out and "baseline" in out
    assert "Secondary" not in out
    assert main(["report", *common, "--baseline", "fixed"]) == 0
    out = capsys.readouterr().out
    assert "| fake/fixed |" in out and out.count("baseline") == 1 and "fake/fixed" in out.split("baseline")[0].splitlines()[-1]
    assert main(["run", *common]) == 0
    assert "reused" in capsys.readouterr().out
    (questions / "rag-questions.yaml").write_text("questions:\n  - q: 'Zebras.'\n    expect: ['a.md']\n", encoding="utf-8")
    assert main(["run", *common]) == 0
    assert main(["report", *common]) == 0
    assert "Secondary questions" in capsys.readouterr().out


def test_main_bad_input(tmp_path, capsys):
    assert main([]) == 2
    assert main(["estimate"]) == 2
    library, snaps, _ = setup(tmp_path, {"a.md": f"# A\n\n{DISK_PARA}\n"})
    questions = tmp_path / "questions"
    questions.mkdir()
    (questions / "a.yaml").write_text("questions:\n  - q: 'x?'\n    expect: ['a.md']\n", encoding="utf-8")
    base = ["--register", str(library / "sources"), "--snapshots", str(snaps), "--questions", str(questions),
            "--runs", str(tmp_path / "runs")]
    assert main(["estimate", *base, "--embedders", "fake", "--chunkers", "nope"]) == 2
    assert main(["estimate", *base, "--embedders", "nope", "--chunkers", "rules"]) == 2
    assert main(["estimate", *base, "--embedders", "fake", "--chunkers", "rules", "--only", "zzz"]) == 2
    assert main(["estimate", *base, "--embedders", "", "--chunkers", "rules"]) == 2
    assert main(["estimate", "--register", str(library / "sources"), "--snapshots", str(tmp_path / "none"),
                 "--questions", str(questions), "--embedders", "fake", "--chunkers", "rules"]) == 2
    assert "not on disk" in capsys.readouterr().err
    assert main(["estimate", *base, "--embedders", "fake", "--chunkers", ""]) == 2
    assert main(["estimate", *base, "--embedders", "fake", "--chunkers", "rules"]) == 0
    assert main(["estimate", "--register", str(library / "sources"), "--snapshots", str(snaps),
                 "--questions", str(tmp_path / "nope")]) == 2
