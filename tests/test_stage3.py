"""Stage 3 command (spec 15 section 3.4): cells, passage texts, evaluation, saved results, report, estimate. Fakes only."""
import json

import pytest

from ragcore.compare import QResult
from ragcore.embed.base import FakeEmbedder
from ragcore.evalset import Question, load_question_dir
from ragcore.rerank import FakeReranker
from ragcore.retrieve import PIPELINES
from ragcore.runs import RunError, load_run
from ragcore.stage import build_run, corpus_fingerprint, main as stage_main
from ragcore.stage3 import (DEFAULT_CELLS, VariantResult, estimate_cell, evaluate_variants, load_cell, load_cell_run,
                            main, parse_cells, passage_texts, render_cell_report, render_estimate, save_cell)
from test_chunk_report import PARA as DISK_PARA, setup
from test_stage import CONFIG, sources

QUESTIONS = [Question("Beta passages talk about zebras and quartz.", ["a.md"]),
             Question("Gamma explains volcano glass and obsidian knives.", ["b.md"])]


def built(tmp_path):
    srcs, emb = sources(), FakeEmbedder()
    build_run(tmp_path / "runs", "rules", srcs, emb, CONFIG, questions_fp="q")
    run = load_cell_run(tmp_path / "runs", emb, "rules", CONFIG, corpus_fingerprint(srcs), "q")
    return srcs, emb, run


# --- cells ---------------------------------------------------------------------------------------

def test_default_cells_are_the_two_chosen_ones():
    assert parse_cells(DEFAULT_CELLS) == [("jina-v2-base-en", "llamaindex"), ("bge-m3", "merge-small")]


@pytest.mark.parametrize("text, message", [
    ("", "no cells"),
    ("bge-m3", "embedder:chunker"),
    ("nope:rules", "embedder:chunker"),
    ("bge-m3:nope", "plain chunker"),
    ("bge-m3:semantic-jina-v2-base-en", "plain chunker"),
    ("bge-m3:rules,bge-m3:rules", "twice"),
])
def test_bad_cells_are_refused(text, message):
    with pytest.raises(ValueError, match=message):
        parse_cells(text)


# --- passage text ----------------------------------------------------------------------------------

def test_passage_texts_are_the_exact_source_slices(tmp_path):
    srcs, _, run = built(tmp_path)
    texts = passage_texts(run, srcs, corpus_fingerprint(srcs))
    by_id = {s.source_id: s.text for s in srcs}
    assert len(texts) == len(run.rows) > 0
    assert all(t == by_id[r["source_id"]][r["start"]:r["end"]] and t for t, r in zip(texts, run.rows))


def test_passage_texts_refuse_another_corpus(tmp_path):
    srcs, _, run = built(tmp_path)
    with pytest.raises(RunError, match="corpus"):
        passage_texts(run, srcs, "0" * 16)


def test_a_missing_cell_run_says_to_build_it_first(tmp_path):
    with pytest.raises(RunError, match="stage 1"):
        load_cell_run(tmp_path / "runs", FakeEmbedder(), "rules", CONFIG, "c", "q")


# --- evaluation ------------------------------------------------------------------------------------

def test_all_five_variants_are_scored_on_every_question(tmp_path):
    srcs, emb, run = built(tmp_path)
    texts = passage_texts(run, srcs, corpus_fingerprint(srcs))
    out = evaluate_variants(run, texts, emb, FakeReranker(), QUESTIONS)
    assert tuple(out) == PIPELINES
    for variant in PIPELINES:
        assert [r.q for r in out[variant].results] == [q.q for q in QUESTIONS]
        assert out[variant].seconds_per_question >= 0
    assert all(r.first for r in out["L"].results)  # the questions quote their passages


def test_saved_variants_are_not_scored_again_and_each_is_saved(tmp_path):
    srcs, emb, run = built(tmp_path)
    texts = passage_texts(run, srcs, corpus_fingerprint(srcs))
    first = evaluate_variants(run, texts, emb, FakeReranker(), QUESTIONS, variants=("V", "L"))
    saves = []
    calls = []

    class Counting(FakeReranker):
        def score(self, query, passages):
            calls.append(query)
            return super().score(query, passages)

    out = evaluate_variants(run, texts, emb, Counting(), QUESTIONS, done=first, save=lambda all_: saves.append(set(all_)))
    assert out["V"] is first["V"] and out["L"] is first["L"]
    assert saves == [{"V", "L", "H"}, {"V", "L", "H", "VR"}, {"V", "L", "H", "VR", "HR"}]
    assert len(calls) == 2 * len(QUESTIONS)  # only VR and HR rerank


def test_truncated_pairs_are_counted_for_reranked_variants_only(tmp_path):
    srcs, emb, run = built(tmp_path)
    texts = passage_texts(run, srcs, corpus_fingerprint(srcs))

    class Cutting(FakeReranker):
        def score(self, query, passages):
            self.truncated += 1
            return super().score(query, passages)

    out = evaluate_variants(run, texts, emb, Cutting(), QUESTIONS)
    assert [out[v].truncated_pairs for v in PIPELINES] == [0, 0, 0, len(QUESTIONS), len(QUESTIONS)]


def test_seconds_per_question_is_the_time_of_a_variant_over_its_questions(tmp_path):
    srcs, emb, run = built(tmp_path)
    texts = passage_texts(run, srcs, corpus_fingerprint(srcs))
    now = iter(range(0, 100, 4))  # every variant takes 4 s for 2 questions
    out = evaluate_variants(run, texts, emb, FakeReranker(), QUESTIONS, clock=lambda: float(next(now)))
    assert [out[v].seconds_per_question for v in PIPELINES] == [2.0] * 5


# --- saved results ---------------------------------------------------------------------------------

def variant_result(first=True):
    return VariantResult([QResult("q1", first, True, 1.0, True), QResult("q2", False, True, 0.5, True)], 0.25, 3)


def test_saved_results_come_back_identical(tmp_path):
    path = tmp_path / "out" / "c.json"
    save_cell(path, "run1", "qfp", "rr", {"V": variant_result()})
    assert load_cell(path, "run1", "qfp", "rr") == {"V": variant_result()}
    assert load_cell(path, "run1", "qfp", None) == {"V": variant_result()}


def test_results_made_from_other_inputs_are_ignored(tmp_path):
    path = tmp_path / "c.json"
    save_cell(path, "run1", "qfp", "rr", {"V": variant_result()})
    assert load_cell(tmp_path / "missing.json", "run1", "qfp", "rr") == {}
    assert load_cell(path, "run2", "qfp", "rr") == {}
    assert load_cell(path, "run1", "other", "rr") == {}
    assert load_cell(path, "run1", "qfp", "other-reranker") == {}


# --- report ----------------------------------------------------------------------------------------

def all_variants():
    return {v: variant_result() for v in PIPELINES}


GROUPS = [("Primary questions", [Question("q1", ["a"]), Question("q2", ["b"])])]


def test_report_has_a_table_with_v_as_baseline_and_the_timings():
    text = render_cell_report("fake/rules", all_variants(), GROUPS)
    assert "Primary questions, fake/rules" in text
    assert "| fake/rules V |" in text and text.count("baseline") == 1
    assert next(l for l in text.splitlines() if l.startswith("| fake/rules V |")).endswith("| baseline |")
    assert all(f"| fake/rules {v} |" in text for v in PIPELINES)
    assert "Seconds per question, fake/rules: V 0.250" in text
    assert "VR 3, HR 3" in text


def test_report_slices_results_by_group():
    variants = {v: VariantResult([QResult("q1", True, True, 1.0, True), QResult("q2", False, False, 0.0, False),
                                  QResult("s1", True, True, 1.0, True)], 0.1, 0) for v in PIPELINES}
    groups = [("Primary questions", GROUPS[0][1]), ("Secondary questions", [Question("s1", ["a"])])]
    text = render_cell_report("c", variants, groups)
    primary, secondary = text.split("Secondary questions, c")
    assert "| c V | 2 | 1 | 1 |" in primary
    assert "| c V | 1 | 1 | 1 |" in secondary


def test_report_refuses_missing_variants_and_other_questions():
    partial = {"V": variant_result()}
    with pytest.raises(ValueError, match="no results for L, H, VR, HR"):
        render_cell_report("c", partial, GROUPS)
    with pytest.raises(ValueError, match="other questions"):
        render_cell_report("c", all_variants(), [("Primary", [Question("zzz", ["a"]), Question("q2", ["b"])])])


# --- estimate --------------------------------------------------------------------------------------

def test_estimate_counts_pairs_and_times_a_sample(tmp_path):
    srcs, _, run = built(tmp_path)
    texts = passage_texts(run, srcs, corpus_fingerprint(srcs))
    ticks = iter([0.0, 2.0, 10.0, 11.0])  # index took 2 s; the sample took 1 s
    e = estimate_cell(run, texts, FakeReranker(), 548, clock=lambda: next(ticks))
    assert e["pairs"] == 548 * 30 * 2 and e["index_seconds"] == 2.0
    assert e["pairs_per_second"] == pytest.approx(min(64, len(texts)))
    text = render_estimate({"fake/rules": e})
    assert "fake/rules" in text and "32880 pairs" in text and "no money" in text


# --- the command -----------------------------------------------------------------------------------

def command_args(tmp_path):
    library, snaps, _ = setup(tmp_path, {"a.md": f"# A\n\n{DISK_PARA}\n\n## Zed\n\nZebras and quartz.\n",
                                         "b.md": f"# B\n\n{DISK_PARA}\n\n## Quokka\n\nQuokkas smile at obsidian.\n"})
    questions = tmp_path / "questions"
    questions.mkdir()
    (questions / "a.yaml").write_text("questions:\n  - q: 'Zebras and quartz.'\n    expect: ['a.md']\n"
                                      "  - q: 'Quokkas smile at obsidian.'\n    expect: ['b.md']\n", encoding="utf-8")
    common = ["--register", str(library / "sources"), "--snapshots", str(snaps), "--questions", str(questions),
              "--runs", str(tmp_path / "runs")]
    return common, [*common, "--out", str(tmp_path / "out"), "--cells", "fake:rules", "--reranker", "fake"]


def test_estimate_run_and_report_end_to_end(tmp_path, capsys):
    common, args = command_args(tmp_path)
    assert stage_main(["run", *common, "--embedders", "fake", "--chunkers", "rules"]) == 0
    capsys.readouterr()
    assert main(["estimate", *args]) == 0
    out = capsys.readouterr().out
    assert "fake/rules" in out and "No result was written" in out and not (tmp_path / "out").exists()
    assert main(["run", *args]) == 0
    assert "0 of 5 variants already saved" in capsys.readouterr().out
    assert (tmp_path / "out" / "fake__rules.json").is_file()
    assert main(["run", *args]) == 0
    assert "5 of 5 variants already saved" in capsys.readouterr().out
    assert main(["report", *args]) == 0
    out = capsys.readouterr().out
    assert "| fake/rules V |" in out and "| fake/rules HR |" in out and "Seconds per question" in out


def test_report_before_run_fails_and_says_so(tmp_path, capsys):
    common, args = command_args(tmp_path)
    assert stage_main(["run", *common, "--embedders", "fake", "--chunkers", "rules"]) == 0
    capsys.readouterr()
    assert main(["report", *args]) == 1
    assert "run stage 3 first" in capsys.readouterr().err


def test_report_never_loads_the_model(tmp_path, capsys):
    common, args = command_args(tmp_path)
    assert stage_main(["run", *common, "--embedders", "fake", "--chunkers", "rules"]) == 0
    assert main(["run", *args]) == 0
    capsys.readouterr()
    real = [*args[:-2], "--reranker", "bge-reranker-v2-m3", "--rerankers", str(tmp_path / "no-such-file.yaml")]
    assert main(["report", *real]) == 0
    assert "| fake/rules V |" in capsys.readouterr().out


def test_nothing_to_search_is_refused(tmp_path, capsys):
    _, args = command_args(tmp_path)
    i = args.index("--snapshots")
    args[i + 1] = str(tmp_path / "elsewhere")
    assert main(["run", *args]) == 2
    assert "no sources loaded" in capsys.readouterr().err


def test_a_cell_without_a_stage_1_run_fails(tmp_path, capsys):
    _, args = command_args(tmp_path)
    assert main(["run", *args]) == 1
    assert "stage 1" in capsys.readouterr().err


def test_bad_input_is_refused(tmp_path, capsys):
    _, args = command_args(tmp_path)
    assert main(["run"]) == 2
    assert main(["run", *args[:-6], "--cells", "bge-m3", "--reranker", "fake"]) == 2
    assert "embedder:chunker" in capsys.readouterr().err
    assert main(["run", *args[:-6], "--cells", "fake:rules", "--reranker", "nope"]) == 1


def test_the_saved_file_names_its_inputs(tmp_path, capsys):
    common, args = command_args(tmp_path)
    assert stage_main(["run", *common, "--embedders", "fake", "--chunkers", "rules"]) == 0
    assert main(["run", *args]) == 0
    saved = json.loads((tmp_path / "out" / "fake__rules.json").read_text(encoding="utf-8"))
    assert set(saved["variants"]) == set(PIPELINES) and saved["reranker"] == "fake"
    assert load_run(tmp_path / "runs", saved["run_id"]).run_id == saved["run_id"]
