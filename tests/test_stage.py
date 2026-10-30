"""Stage runner (spec 13 section 6): corpus, building runs, the printed estimate, the report. Fake embedder only."""
import importlib.util

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
    assert CHUNKER_NAMES == ("rules", "fixed", "merge-small", "parent-child", "llamaindex")


def test_labels_name_the_chunker_and_change_with_settings():
    labels = {n: chunker_label(n, CONFIG) for n in CHUNKER_NAMES}
    assert len(set(labels.values())) == 5
    assert chunker_label("rules", CONFIG) != chunker_label("rules", ChunkConfig(max_chars=500, hard_limit=2000))
    assert "400" in labels["fixed"] and "300" in labels["merge-small"]
    for name in CHUNKER_NAMES:
        assert chunker_label(name, CONFIG) != chunker_label(name, ChunkConfig(max_chars=400, hard_limit=3000))


def test_unknown_chunker_refused():
    with pytest.raises(ValueError, match="unknown chunker"):
        chunk_corpus(sources(), "nope", CONFIG)


HAS_LLAMA = importlib.util.find_spec("llama_index") is not None
CHUNKERS_HERE = [n if n != "llamaindex" else pytest.param(n, marks=pytest.mark.skipif(not HAS_LLAMA, reason="needs the variants group"))
                 for n in CHUNKER_NAMES]


@pytest.mark.parametrize("name", CHUNKERS_HERE)
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


def test_parent_child_run_stores_the_parent_of_every_child(tmp_path):
    built = build_run(tmp_path, "parent-child", sources(), FakeEmbedder(), CONFIG, questions_fp="q")
    run = load_run(tmp_path, built.run_id)
    assert all(r["parent"] for r in run.rows) and len(run.rows) == built.count
    rules_ids = {p.passage_id for p in chunk_corpus(sources(), "rules", CONFIG)[0]}
    assert {r["parent"] for r in run.rows} <= rules_ids
    plain = build_run(tmp_path, "rules", sources(), FakeEmbedder(), CONFIG, questions_fp="q")
    assert all(r["parent"] is None for r in load_run(tmp_path, plain.run_id).rows)


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


def test_parent_child_label_matches_the_children_version():
    from ragcore.chunking import chunk_parent_child
    version = chunk_parent_child(sources()[0], CONFIG).children[0].chunker_version
    assert chunker_label("parent-child", CONFIG) == version


def test_parent_child_dropped_count_reaches_the_manifest(tmp_path):
    srcs = [source("c.md", "# A\n\n" + PARA + "\n\n## B\n\n<!-- hidden -->\n")]
    built = build_run(tmp_path, "parent-child", srcs, FakeEmbedder(), CONFIG, questions_fp="q")
    assert built.dropped_empty >= 1
    assert load_run(tmp_path, built.run_id).manifest["extra"]["dropped_empty"] == built.dropped_empty


# --- semantic chunkers (steered by an embedder) ---------------------------------------------------

from ragcore.semantic import SemanticChunker  # noqa: E402
from ragcore.stage import is_semantic, semantic_name  # noqa: E402

BIG = ChunkConfig(max_chars=1500, hard_limit=2000)
A = "apple banana cherry grape melon peach plum pear " * 3
B = "quartz basalt granite obsidian marble slate flint chalk " * 3


def embedded(emb):
    return sum(n for _, n in emb.calls)


def topic_sources():
    text = f"# One\n\n{A}\n\n{A}\n\n{B}\n\n{B}\n"
    return [source("t.md", text)]


class Boom(FakeEmbedder):
    def embed(self, texts, *, kind):
        raise AssertionError("the steering embedder must not be called")


def test_semantic_names():
    assert semantic_name("bge-m3") == "semantic-bge-m3"
    assert is_semantic("semantic-jina-v2-base-en") and not is_semantic("rules")


def test_semantic_label_needs_its_steering_embedder():
    with pytest.raises(ValueError, match="steering"):
        chunker_label("semantic-fake", BIG)
    steer = FakeEmbedder()
    assert chunker_label("semantic-fake", BIG, {"fake": steer}) == SemanticChunker(steer).version(BIG)
    assert chunker_label("semantic-fake", BIG, {"fake": FakeEmbedder(dim=16)}) != chunker_label(
        "semantic-fake", BIG, {"fake": steer})


def test_semantic_run_uses_the_steering_cuts_and_is_reused(tmp_path):
    steer = {"fake": FakeEmbedder()}
    built = build_run(tmp_path, "semantic-fake", topic_sources(), FakeEmbedder(), BIG, questions_fp="q", steerers=steer)
    plain = build_run(tmp_path, "rules", topic_sources(), FakeEmbedder(), BIG, questions_fp="q")
    assert built.count > plain.count
    again = build_run(tmp_path, "semantic-fake", topic_sources(), FakeEmbedder(), BIG, questions_fp="q",
                      steerers={"fake": Boom()})
    assert again.reused is True and again.run_id == built.run_id  # same fingerprint, so no chunking and no steering


def test_the_chunking_is_cached_on_disk_for_the_other_embedder(tmp_path):
    steer = Counting()
    build_run(tmp_path, "semantic-fake", topic_sources(), FakeEmbedder(), BIG, questions_fp="q", steerers={"fake": steer})
    used = embedded(steer)
    assert used > 0 and len(list((tmp_path / "_chunks").iterdir())) == 1
    other = build_run(tmp_path, "semantic-fake", topic_sources(), FakeEmbedder(dim=16), BIG, questions_fp="q",
                      steerers={"fake": steer})
    assert embedded(steer) == used and other.count > 0


def test_a_cached_chunking_works_without_the_steering_model(tmp_path):
    build_run(tmp_path, "semantic-fake", topic_sources(), FakeEmbedder(), BIG, questions_fp="q",
              steerers={"fake": FakeEmbedder()})
    for run in tmp_path.iterdir():
        if run.name != "_chunks":
            import shutil
            shutil.rmtree(run)
    built = build_run(tmp_path, "semantic-fake", topic_sources(), FakeEmbedder(), BIG, questions_fp="q",
                      steerers={"fake": Boom()})
    assert built.count > 0 and built.reused is False


def test_cache_is_keyed_by_corpus_and_settings(tmp_path):
    steer = {"fake": FakeEmbedder()}
    build_run(tmp_path, "semantic-fake", topic_sources(), FakeEmbedder(), BIG, questions_fp="q", steerers=steer)
    build_run(tmp_path, "semantic-fake", sources(), FakeEmbedder(), BIG, questions_fp="q", steerers=steer)
    assert len(list((tmp_path / "_chunks").iterdir())) == 2


def test_estimate_for_semantic_uses_rules_as_a_stand_in_and_never_embeds_blocks(tmp_path):
    emb = FakeEmbedder()
    steer = Counting()
    result = estimate(sources(), ["semantic-fake"], [emb], BIG, tmp_path, questions_fp="q", probe_n=2,
                      steerers={"fake": steer})
    run = result["runs"][0]
    rules = chunk_corpus(sources(), "rules", BIG)[0]
    assert run["approx"] is True and run["passages"] == len(rules)
    assert embedded(steer) <= 2  # only the speed sample
    assert result["steering"]["semantic-fake"] >= 0.0
    assert "semantic-fake" in render_estimate(result) and "about" in render_estimate(result)


def test_main_runs_semantic_and_reports_it(tmp_path, capsys):
    library, snaps, _ = setup(tmp_path, {"a.md": f"# A\n\n{DISK_PARA}\n\n## Zed\n\nZebras and quartz.\n"})
    questions = tmp_path / "questions"
    questions.mkdir()
    (questions / "a.yaml").write_text("questions:\n  - q: 'Zebras and quartz.'\n    expect: ['a.md']\n", encoding="utf-8")
    common = ["--register", str(library / "sources"), "--snapshots", str(snaps), "--questions", str(questions),
              "--runs", str(tmp_path / "runs"), "--embedders", "fake", "--chunkers", "rules,semantic-fake"]
    assert main(["estimate", *common]) == 0
    assert main(["run", *common]) == 0
    assert "fake/semantic-fake" in capsys.readouterr().out
    assert main(["report", *common]) == 0
    assert "| fake/semantic-fake |" in capsys.readouterr().out
    assert main(["estimate", *common[:-1], "rules,semantic-nope"]) == 2


def _drop_runs(root):
    import shutil
    for run in root.iterdir():
        if run.name != "_chunks":
            shutil.rmtree(run)


def test_a_cached_chunking_keeps_its_dropped_count(tmp_path):
    srcs = [source("c.md", "# A\n\n" + A + "\n\n## B\n\n<!-- hidden -->\n")]
    first = build_run(tmp_path, "semantic-fake", srcs, FakeEmbedder(), BIG, questions_fp="q", steerers={"fake": FakeEmbedder()})
    assert first.dropped_empty >= 1
    _drop_runs(tmp_path)
    second = build_run(tmp_path, "semantic-fake", srcs, FakeEmbedder(), BIG, questions_fp="q", steerers={"fake": Boom()})
    assert second.dropped_empty == first.dropped_empty


def test_a_cache_that_does_not_fit_the_corpus_is_ignored(tmp_path):
    import json
    build_run(tmp_path, "semantic-fake", topic_sources(), FakeEmbedder(), BIG, questions_fp="q", steerers={"fake": FakeEmbedder()})
    _drop_runs(tmp_path)
    (cache,) = (tmp_path / "_chunks").iterdir()
    data = json.loads(cache.read_text(encoding="utf-8"))
    data["spans"].append(["s1/other.md", 0, 5, []])
    cache.write_text(json.dumps(data), encoding="utf-8")
    steer = Counting()
    build_run(tmp_path, "semantic-fake", topic_sources(), FakeEmbedder(), BIG, questions_fp="q", steerers={"fake": steer})
    assert embedded(steer) > 0


def test_estimate_knows_when_the_steering_pass_is_already_done_or_not_needed(tmp_path):
    steer = {"fake": FakeEmbedder()}
    build_run(tmp_path, "semantic-fake", topic_sources(), FakeEmbedder(), BIG, questions_fp="q", steerers=steer)
    done = estimate(topic_sources(), ["semantic-fake"], [FakeEmbedder()], BIG, tmp_path, questions_fp="q", steerers=steer)
    assert done["steering"] == {} and done["to_build"] == 0
    _drop_runs(tmp_path)
    cached = estimate(topic_sources(), ["semantic-fake"], [FakeEmbedder()], BIG, tmp_path, questions_fp="q", steerers=steer)
    assert cached["steering"] == {"semantic-fake": 0.0} and cached["to_build"] == 1


def test_estimate_times_the_steering_pass_from_the_steerers_speed(tmp_path):
    emb, steer = FakeEmbedder(), FakeEmbedder()
    ticks = iter([0.0, 2.0])
    result = estimate(topic_sources(), ["semantic-fake"], [emb], BIG, tmp_path, questions_fp="q", probe_n=2,
                      clock=lambda: next(ticks), steerers={"fake": steer})
    rate = result["probe"]["fake"]["tokens_per_second"]
    tokens = sum(steer.count_tokens(p.text) for p in chunk_corpus(topic_sources(), "rules", BIG)[0])
    assert result["steering"]["semantic-fake"] == pytest.approx(tokens / rate)
    assert result["total_seconds"] == pytest.approx(result["runs"][0]["seconds"] + tokens / rate)


def test_a_steerer_that_is_not_an_embedder_gets_its_own_speed_sample(tmp_path):
    class Other(FakeEmbedder):
        id = "other"
    result = estimate(topic_sources(), ["semantic-other"], [FakeEmbedder()], BIG, tmp_path, questions_fp="q", probe_n=2,
                      steerers={"other": Other()})
    assert set(result["probe"]) == {"fake", "other"}


def test_main_makes_one_model_per_id(tmp_path, monkeypatch):
    import ragcore.stage as stage
    library, snaps, _ = setup(tmp_path, {"a.md": f"# A\n\n{DISK_PARA}\n"})
    questions = tmp_path / "questions"
    questions.mkdir()
    (questions / "a.yaml").write_text("questions:\n  - q: 'x?'\n    expect: ['a.md']\n", encoding="utf-8")
    made = []
    real = stage.make_embedder
    monkeypatch.setattr(stage, "make_embedder", lambda i, m: made.append(i) or real(i, m))
    assert main(["estimate", "--register", str(library / "sources"), "--snapshots", str(snaps), "--questions", str(questions),
                 "--runs", str(tmp_path / "runs"), "--embedders", "fake", "--chunkers", "rules,semantic-fake"]) == 0
    assert made == ["fake"]


def test_llamaindex_label_works_without_llamaindex_installed():
    assert chunker_label("llamaindex", CONFIG) == "llamaindex-1/375/400/2000"


@pytest.mark.skipif(not HAS_LLAMA, reason="needs the variants group")
def test_llamaindex_run_is_built_and_stops_when_too_many_nodes_are_unmapped(tmp_path, monkeypatch):
    built = build_run(tmp_path, "llamaindex", sources(), FakeEmbedder(), CONFIG, questions_fp="q")
    assert built.count > 0
    import ragcore.llamaindex_chunker as li
    monkeypatch.setattr(li.LlamaIndexChunker, "_node_texts", lambda self, text: ["never in the text " + t for t in ["a", "b"]])
    with pytest.raises(RuntimeError, match="1%"):
        build_run(tmp_path, "llamaindex", sources(), FakeEmbedder(dim=16), CONFIG, questions_fp="q")


# --- exclusions (spec 14) -------------------------------------------------------------------------

from ragcore.exclusions import Rule  # noqa: E402


def test_load_corpus_applies_exclusions_and_says_so(tmp_path):
    library, snaps, _ = setup(tmp_path, {"a.md": f"# A\n\n{DISK_PARA}\n", "b.md": f"# B\n\n{DISK_PARA}\n",
                                         "keep/c.md": f"# C\n\n{DISK_PARA}\n"})
    rules = {"s1": [Rule("exclude", "*.md", "all"), Rule("include", "keep/*", ""), Rule("exclude", "typo/*", "x")]}
    loaded, notes = load_corpus(library / "sources", snaps, exclusions=rules)
    assert [s.source_id for s in loaded] == ["s1/keep/c.md"]
    assert "exclusions: 2 files left out of s1" in notes
    assert "exclusion rule matches no file: s1 'typo/*'" in notes


def test_rules_for_sources_that_were_not_loaded_are_not_reported(tmp_path):
    library, snaps, _ = setup(tmp_path, {"a.md": f"# A\n\n{DISK_PARA}\n"})
    loaded, notes = load_corpus(library / "sources", snaps, exclusions={"elsewhere": [Rule("exclude", "*", "x")]})
    assert len(loaded) == 1 and notes == []


def test_the_run_id_changes_when_the_corpus_changes_through_exclusions(tmp_path):
    library, snaps, _ = setup(tmp_path, {"a.md": f"# A\n\n{DISK_PARA}\n", "b.md": f"# B\n\n{DISK_PARA} more\n"})
    full, _ = load_corpus(library / "sources", snaps)
    less, _ = load_corpus(library / "sources", snaps, exclusions={"s1": [Rule("exclude", "b.md", "noise")]})
    one = build_run(tmp_path / "runs", "rules", full, FakeEmbedder(), CONFIG, questions_fp="q")
    two = build_run(tmp_path / "runs", "rules", less, FakeEmbedder(), CONFIG, questions_fp="q")
    assert one.run_id != two.run_id and two.count < one.count


def _main_args(tmp_path, library, snaps):
    questions = tmp_path / "questions"
    questions.mkdir(exist_ok=True)
    (questions / "a.yaml").write_text("questions:\n  - q: 'x?'\n    expect: ['a.md']\n", encoding="utf-8")
    return ["--register", str(library / "sources"), "--snapshots", str(snaps), "--questions", str(questions),
            "--runs", str(tmp_path / "runs"), "--embedders", "fake", "--chunkers", "rules"]


def test_main_exclusions_file_none_and_missing(tmp_path, capsys):
    library, snaps, _ = setup(tmp_path, {"a.md": f"# A\n\n{DISK_PARA}\n", "b.md": f"# B\n\n{DISK_PARA}\n"})
    ex = tmp_path / "ex.yaml"
    ex.write_text("s1:\n  - exclude: b.md\n    reason: noise\n", encoding="utf-8")
    args = _main_args(tmp_path, library, snaps)
    assert main(["estimate", *args, "--exclusions", str(ex)]) == 0
    out = capsys.readouterr().out
    assert "exclusions: 1 files left out of s1" in out and "Corpus: 1 sources" in out
    assert main(["estimate", *args, "--exclusions", "none"]) == 0
    assert "Corpus: 2 sources" in capsys.readouterr().out
    assert main(["estimate", *args, "--exclusions", str(tmp_path / "missing.yaml")]) == 2
    ex.write_text("s1:\n  - exclude: b.md\n", encoding="utf-8")
    assert main(["estimate", *args, "--exclusions", str(ex)]) == 2


def test_main_uses_the_shipped_file_by_default(tmp_path, capsys):
    library, snaps, _ = setup(tmp_path, {"a.md": f"# A\n\n{DISK_PARA}\n"})
    assert main(["estimate", *_main_args(tmp_path, library, snaps)]) == 0
    assert "Corpus: 1 sources" in capsys.readouterr().out  # no rule names source s1, and rules for other sources stay quiet
