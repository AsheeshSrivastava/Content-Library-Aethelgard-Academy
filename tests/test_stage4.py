"""Stage 4 command (spec 16): prepare questions, evaluate V and HR on passages, report, estimate. Fakes only."""
import pytest

from ragcore.compare import QResult
from ragcore.embed.base import FakeEmbedder
from ragcore.passage_eval import QuoteSpan
from ragcore.passage_questions import PassageQuestion
from ragcore.rerank import FakeReranker
from ragcore.runs import RunError
from ragcore.stage import corpus_fingerprint, main as stage_main
from ragcore.stage3 import VariantResult, passage_texts, load_cell_run
from ragcore.stage import build_run
from ragcore.stage4 import (VARIANTS, estimate_cell, evaluate_cell, main, passage_questions_fingerprint, prepare,
                            render_estimate, render_report)
from test_chunk_report import PARA as DISK_PARA, setup
from test_stage import CONFIG, sources

QUOTE_A = "Beta passages talk about zebras and quartz and nothing else at all, said the guide."
QUOTE_B = "Gamma explains volcano glass and obsidian knives in much more detail than anyone needs."
TEXTS = {"s1/a.md": f"# Alpha\n\n{QUOTE_A}\n\nMore text about retrieval.\n", "s1/b.md": f"# Gamma\n\n{QUOTE_B}\n\nElse.\n"}


def pq(q, source, quote, kind="fact"):
    return PassageQuestion(q, source, quote, kind)


QUESTIONS = [pq("Which animals and stones does the first guide mention?", "a.md", QUOTE_A),
             pq("What is made of volcanic glass according to the second text?", "b.md", QUOTE_B, "code")]


def write_questions(folder, questions=QUESTIONS):
    folder.mkdir(exist_ok=True)
    body = "questions:\n" + "".join(f"- q: '{x.q}'\n  source: {x.source}\n  quote: '{x.quote}'\n  kind: {x.kind}\n" for x in questions)
    (folder / "set.yaml").write_text(body, encoding="utf-8")
    return folder


# --- preparing the questions ------------------------------------------------------------------------

def test_prepare_returns_questions_and_their_places(tmp_path):
    folder = write_questions(tmp_path / "pq")
    questions, spans = prepare(folder, TEXTS, check_set=False)
    assert questions == QUESTIONS and spans[0] == QuoteSpan("s1/a.md", TEXTS["s1/a.md"].index(QUOTE_A), TEXTS["s1/a.md"].index(QUOTE_A) + len(QUOTE_A))


def test_prepare_lists_the_problems_and_stops(tmp_path):
    bad = [*QUESTIONS, pq("Why?", "nope", QUOTE_A)]
    with pytest.raises(ValueError, match="no source matches 'nope'"):
        prepare(write_questions(tmp_path / "pq", bad), TEXTS, check_set=False)


def test_prepare_checks_the_mix_of_the_whole_set_unless_told_not_to(tmp_path):
    folder = write_questions(tmp_path / "pq")
    with pytest.raises(ValueError, match="about 150"):
        prepare(folder, TEXTS)


def test_the_fingerprint_follows_the_content(tmp_path):
    one = passage_questions_fingerprint(write_questions(tmp_path / "pq"))
    assert one == passage_questions_fingerprint(tmp_path / "pq")
    other = passage_questions_fingerprint(write_questions(tmp_path / "pq2", QUESTIONS[:1]))
    assert one != other


# --- evaluation ------------------------------------------------------------------------------------

def built(tmp_path):
    srcs, emb = sources(), FakeEmbedder()
    build_run(tmp_path / "runs", "rules", srcs, emb, CONFIG, questions_fp="q")
    run = load_cell_run(tmp_path / "runs", emb, "rules", CONFIG, corpus_fingerprint(srcs), "q")
    return srcs, emb, run, passage_texts(run, srcs, corpus_fingerprint(srcs))


def test_both_variants_are_scored_and_saved_one_by_one(tmp_path):
    srcs, emb, run, texts = built(tmp_path)
    by_id = {s.source_id: s.text for s in srcs}
    qs = [pq("Which animals and stones are discussed in the first file?", "a.md", "Beta passages talk about zebras and quartz.")]
    spans = [QuoteSpan("s1/a.md", by_id["s1/a.md"].index(qs[0].quote), by_id["s1/a.md"].index(qs[0].quote) + len(qs[0].quote))]
    saves = []
    out = evaluate_cell(run, texts, emb, FakeReranker(), qs, spans, save=lambda all_: saves.append(set(all_)))
    assert tuple(out) == VARIANTS and saves == [{"V"}, {"V", "HR"}]
    assert all(len(v.results) == 1 and v.seconds_per_question >= 0 for v in out.values())
    again = evaluate_cell(run, texts, emb, FakeReranker(), qs, spans, done={"V": out["V"]})
    assert again["V"] is out["V"]


# --- the report ------------------------------------------------------------------------------------

def result(*flags, seconds=0.1, cut=0):
    return VariantResult([QResult(f"q{i}", f, f, 1.0 if f else 0.0, f) for i, f in enumerate(flags)], seconds, cut)


def test_report_shows_each_cell_the_cells_against_each_other_kinds_and_cuts():
    qs = [pq(f"q{i}", "a", "x", "code" if i < 2 else "fact") for i in range(4)]
    cells = {"jina/llamaindex": {"V": result(True, False, False, False), "HR": result(True, True, True, False, seconds=0.2, cut=3)},
             "bge/merge-small": {"V": result(True, True, False, False), "HR": result(True, True, True, True, seconds=0.3)}}
    cuts = {"jina/llamaindex": [False, False, True, True], "bge/merge-small": [False, False, False, True]}
    text = render_report(cells, qs, cuts)
    assert "| jina/llamaindex V |" in text and "| jina/llamaindex HR |" in text and text.count("baseline") >= 4
    assert "Between the cells, hybrid + reranker" in text and "Between the cells, vector only" in text
    assert "quote cut by the chunking" in text.lower() and "jina/llamaindex: 2 of 4" in text and "bge/merge-small: 1 of 4" in text
    assert "code" in text and "fact" in text
    assert "Seconds per question, jina/llamaindex: V 0.100, HR 0.200" in text
    assert "Pairs cut to fit the reranker limit: jina/llamaindex 3" in text


def test_report_refuses_missing_results_and_other_questions():
    qs = [pq("q0", "a", "x")]
    with pytest.raises(ValueError, match="no results for HR"):
        render_report({"c": {"V": result(True)}}, qs, {"c": [False]})
    with pytest.raises(ValueError, match="other questions"):
        render_report({"c": {"V": result(True), "HR": result(True)}}, [pq("zzz", "a", "x")], {"c": [False]})


def test_report_with_one_cell_has_no_between_cell_tables():
    qs = [pq("q0", "a", "x")]
    text = render_report({"c": {"V": result(True), "HR": result(True)}}, qs, {"c": [False]})
    assert "Between the cells" not in text


# --- estimate ----------------------------------------------------------------------------------------

def test_estimate_counts_pairs_for_hr_only(tmp_path):
    _, _, run, texts = built(tmp_path)
    ticks = iter([0.0, 2.0, 10.0, 11.0])
    e = estimate_cell(run, texts, FakeReranker(), 150, clock=lambda: next(ticks))
    assert e["pairs"] == 150 * 30
    text = render_estimate({"c": e})
    assert "4500 pairs" in text and "no money" in text


# --- the command -----------------------------------------------------------------------------------

def command_args(tmp_path):
    files = {"a.md": f"# A\n\n{DISK_PARA}\n\n## Zed\n\nZebras and quartz are discussed here at some length by the guide.\n",
             "b.md": f"# B\n\n{DISK_PARA}\n\n## Quokka\n\nQuokkas smile at obsidian rocks whenever the weather turns cold.\n"}
    library, snaps, _ = setup(tmp_path, files)
    folder = tmp_path / "pq"
    folder.mkdir()
    (folder / "set.yaml").write_text(
        "questions:\n- q: 'Which animals are happy near dark volcanic glass?'\n  source: b.md\n"
        "  quote: 'Quokkas smile at obsidian rocks whenever the weather turns cold.'\n  kind: fact\n"
        "- q: 'What are the striped creatures and the crystal discussed alongside?'\n  source: a.md\n"
        "  quote: 'Zebras and quartz are discussed here at some length by the guide.'\n  kind: fact\n", encoding="utf-8")
    common = ["--register", str(library / "sources"), "--snapshots", str(snaps), "--questions", str(tmp_path / "unused"),
              "--runs", str(tmp_path / "runs")]
    (tmp_path / "unused").mkdir()
    (tmp_path / "unused" / "q.yaml").write_text("questions:\n  - q: 'x?'\n    expect: ['a.md']\n", encoding="utf-8")
    args = [*common, "--passage-questions", str(folder), "--out", str(tmp_path / "out"), "--cells", "fake:rules",
            "--reranker", "fake", "--small-set-for-tests"]
    return common, args


def test_estimate_run_and_report_end_to_end(tmp_path, capsys):
    common, args = command_args(tmp_path)
    assert stage_main(["run", *common, "--embedders", "fake", "--chunkers", "rules"]) == 0
    capsys.readouterr()
    assert main(["estimate", *args]) == 0
    assert "No result was written" in capsys.readouterr().out and not (tmp_path / "out").exists()
    assert main(["run", *args]) == 0
    assert "0 of 2 variants already saved" in capsys.readouterr().out
    assert main(["run", *args]) == 0
    assert "2 of 2 variants already saved" in capsys.readouterr().out
    assert main(["report", *args]) == 0
    out = capsys.readouterr().out
    assert "| fake/rules V |" in out and "| fake/rules HR |" in out and "quote cut" in out.lower()


def test_report_before_run_fails_and_a_missing_stage_1_run_fails(tmp_path, capsys):
    common, args = command_args(tmp_path)
    assert main(["run", *args]) == 1
    assert "stage 1" in capsys.readouterr().err
    assert stage_main(["run", *common, "--embedders", "fake", "--chunkers", "rules"]) == 0
    assert main(["report", *args]) == 1
    assert "run stage 4 first" in capsys.readouterr().err


def test_a_real_sized_set_is_refused_unless_it_has_the_right_mix(tmp_path, capsys):
    _, args = command_args(tmp_path)
    assert main(["run", *[a for a in args if a != "--small-set-for-tests"]]) == 2
    assert "about 150" in capsys.readouterr().err


def test_bad_input(tmp_path, capsys):
    assert main([]) == 2
    _, args = command_args(tmp_path)
    i = args.index("--cells")
    args[i + 1] = "bge-m3"
    assert main(["run", *args]) == 2
    assert "embedder:chunker" in capsys.readouterr().err

# --- tests added after the mutation check ---------------------------------------------------------------

def test_several_problems_are_listed_together(tmp_path):
    bad = [pq("Why one?", "nope", QUOTE_A), pq("Why two?", "nada", QUOTE_B)]
    with pytest.raises(ValueError, match="2 problems.*nope.*nada"):
        prepare(write_questions(tmp_path / "pq", bad), TEXTS, check_set=False)


def test_more_problems_than_shown_are_counted(tmp_path):
    bad = [pq(f"Why number {i} indeed?", f"none{i}", QUOTE_A) for i in range(12)]
    # 12 unknown sources and one repeated quote make 13 problems; 10 are shown.
    with pytest.raises(ValueError, match=r"13 problems.*\(and 3 more\)"):
        prepare(write_questions(tmp_path / "pq", bad), TEXTS, check_set=False)


def test_the_fingerprint_sees_a_renamed_file(tmp_path):
    folder = write_questions(tmp_path / "pq")
    before = passage_questions_fingerprint(folder)
    (folder / "set.yaml").rename(folder / "renamed.yaml")
    assert passage_questions_fingerprint(folder) != before


def test_seconds_per_question_and_cut_pairs_are_measured(tmp_path):
    srcs, emb, run, texts = built(tmp_path)
    by_id = {s.source_id: s.text for s in srcs}
    quote = "Beta passages talk about zebras and quartz."
    qs = [pq(f"Which animals and stones are discussed in file number {i}?", "a.md", quote) for i in range(2)]
    start = by_id["s1/a.md"].index(quote)
    spans = [QuoteSpan("s1/a.md", start, start + len(quote))] * 2
    ticks = iter(float(i * 4) for i in range(10))  # each variant takes 4 s for 2 questions

    class Cutting(FakeReranker):
        def score(self, query, passages):
            self.truncated += 1
            return super().score(query, passages)

    out = evaluate_cell(run, texts, emb, Cutting(), qs, spans, clock=lambda: next(ticks))
    assert out["V"].seconds_per_question == 2.0 and out["HR"].seconds_per_question == 2.0
    assert out["V"].truncated_pairs == 0 and out["HR"].truncated_pairs == 2


def test_each_table_has_its_own_baseline_row():
    qs = [pq("q0", "a", "x"), pq("q1", "a", "x")]
    cells = {"one": {"V": result(True, False), "HR": result(True, True)}, "two": {"V": result(False, False), "HR": result(True, False)}}
    text = render_report(cells, qs, {"one": [False, False], "two": [False, False]})
    lines = text.splitlines()
    assert next(l for l in lines if l.startswith("| one V |")).endswith("| baseline |")
    assert next(l for l in lines if l.startswith("| two V |") and "baseline" in l)
    between = text.split("Between the cells, hybrid + reranker")[1].split("Between the cells, vector only")[0]
    assert next(l for l in between.splitlines() if l.startswith("| one HR |")).endswith("| baseline |")
    assert not next(l for l in between.splitlines() if l.startswith("| two HR |")).endswith("| baseline |")


def test_results_are_ignored_when_the_questions_change_and_the_report_ignores_the_reranker(tmp_path, capsys):
    import json
    common, args = command_args(tmp_path)
    assert stage_main(["run", *common, "--embedders", "fake", "--chunkers", "rules"]) == 0
    assert main(["run", *args]) == 0
    capsys.readouterr()
    saved = tmp_path / "out" / "fake__rules.json"
    original = saved.read_text(encoding="utf-8")
    data = json.loads(original)
    data["reranker"] = "some other reranker"
    saved.write_text(json.dumps(data), encoding="utf-8")
    assert main(["report", *args]) == 0 and "| fake/rules HR |" in capsys.readouterr().out
    saved.write_text(original, encoding="utf-8")
    assert main(["run", *args]) == 0 and "2 of 2 variants already saved" in capsys.readouterr().out
    (tmp_path / "pq" / "set.yaml").write_text((tmp_path / "pq" / "set.yaml").read_text(encoding="utf-8") + "# edited\n", encoding="utf-8")
    assert main(["run", *args]) == 0
    assert "0 of 2 variants already saved" in capsys.readouterr().out


def test_the_report_uses_the_cut_flags_of_the_run(tmp_path, capsys, monkeypatch):
    common, args = command_args(tmp_path)
    assert stage_main(["run", *common, "--embedders", "fake", "--chunkers", "rules"]) == 0
    assert main(["run", *args]) == 0
    capsys.readouterr()
    monkeypatch.setattr("ragcore.stage4.cut_flags", lambda run, spans: [True] * len(spans))
    assert main(["report", *args]) == 0
    assert "fake/rules: 2 of 2" in capsys.readouterr().out
