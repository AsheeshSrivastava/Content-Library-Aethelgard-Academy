"""Stage 3 (spec 15 section 3.4): hybrid search and the reranker on cells that stage 1 already built.

`estimate` prints what will be reranked and a timed sample; `run` evaluates the five pipelines V, L, H, VR, HR on every
question and saves the per-question results; `report` prints the paired tables with V of the same cell as baseline.
Nothing here builds a run or calls a paid service."""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from ragcore.chunking import ChunkConfig
from ragcore.compare import QResult, markdown_table, score_questions, split_sets
from ragcore.contracts import Source
from ragcore.errors import RegisterError
from ragcore.evalset import Question, load_question_dir
from ragcore.rerank import FakeReranker, load_reranker_specs
from ragcore.retrieve import PIPELINES, RERANK_DEPTH, keyword_index, make_pipeline
from ragcore.runs import Run, RunError, compute_run_id, load_run
from ragcore.stage import (DEFAULT_MODELS, DEFAULT_RUNS, EMBEDDER_IDS, _csv, _exclusions, _safe_print, corpus_fingerprint,
                           is_semantic, load_corpus, make_embedder, questions_fingerprint, run_inputs, valid_chunker)

DEFAULT_RERANKERS = Path(__file__).resolve().parent.parent.parent / "configs" / "rerankers.yaml"
DEFAULT_OUT = Path.home() / ".ragcore" / "stage3"
DEFAULT_CELLS = "jina-v2-base-en:llamaindex,bge-m3:merge-small"
RERANKER_IDS = ("fake", "bge-reranker-v2-m3")
RERANKED = ("VR", "HR")
SAMPLE_PAIRS = 64


def parse_cells(text: str) -> list[tuple[str, str]]:
    """`embedder:chunker,embedder:chunker` as (embedder, chunker) pairs."""
    cells = []
    for part in _csv(text):
        embedder, _, chunker = part.partition(":")
        if embedder not in EMBEDDER_IDS or not chunker:
            raise ValueError(f"cell {part!r}: write it as embedder:chunker, with an embedder from {', '.join(EMBEDDER_IDS)}")
        if not valid_chunker(chunker) or is_semantic(chunker):
            raise ValueError(f"cell {part!r}: {chunker!r} is not a plain chunker name")
        if (embedder, chunker) in cells:
            raise ValueError(f"cell {part!r} is listed twice")
        cells.append((embedder, chunker))
    if not cells:
        raise ValueError("no cells chosen")
    return cells


def load_cell_run(root: Path, embedder, chunker: str, config: ChunkConfig, corpus_fp: str, questions_fp: str) -> Run:
    run_id = compute_run_id(run_inputs(chunker, config, embedder, corpus_fp, questions_fp))
    try:
        return load_run(root, run_id)
    except RunError as exc:
        raise RunError(f"{embedder.id}/{chunker}: {exc}; build it with stage 1 first") from None


def passage_texts(run: Run, sources: list[Source], corpus_fp: str) -> list[str]:
    """The text of every passage of the run, cut from the sources it was built from."""
    built_from = run.manifest["inputs"]["corpus"]
    if built_from != corpus_fp:
        raise RunError(f"run {run.run_id} was built from corpus {built_from} but this corpus is {corpus_fp}")
    by_id = {s.source_id: s.text for s in sources}
    return [by_id[row["source_id"]][row["start"]:row["end"]] for row in run.rows]


@dataclass
class VariantResult:
    results: list[QResult]
    seconds_per_question: float
    truncated_pairs: int


def evaluate_variants(run: Run, texts: list[str], embedder, reranker, questions: list[Question],
                      variants: tuple[str, ...] = PIPELINES, done: dict[str, VariantResult] | None = None,
                      save=None, clock=time.perf_counter) -> dict[str, VariantResult]:
    """Score each variant on all questions. `done` holds variants already scored; `save(all)` is called after each."""
    out = dict(done or {})
    index = keyword_index(run, texts)
    for variant in variants:
        if variant in out:
            continue
        retriever = make_pipeline(variant, run, texts, embedder=embedder, reranker=reranker, index=index)
        before = getattr(reranker, "truncated", 0)
        start = clock()
        results = score_questions(questions, retriever)
        seconds = clock() - start
        truncated = getattr(reranker, "truncated", 0) - before
        out[variant] = VariantResult(results, seconds / max(1, len(questions)), truncated)
        if save:
            save(out)
    return out


# --- saved results -----------------------------------------------------------------------------

def _cell_path(out_dir: Path, embedder_id: str, chunker: str) -> Path:
    return out_dir / f"{embedder_id}__{chunker}.json"


def save_cell(path: Path, run_id: str, questions_fp: str, reranker_fp: str, variants: dict[str, VariantResult]) -> None:
    data = {"run_id": run_id, "questions": questions_fp, "reranker": reranker_fp,
            "variants": {v: {"seconds_per_question": r.seconds_per_question, "truncated_pairs": r.truncated_pairs,
                             "results": [[x.q, x.first, x.top_k, x.rr, x.recall10] for x in r.results]}
                         for v, r in variants.items()}}
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data), encoding="utf-8")
    temp.replace(path)


def load_cell(path: Path, run_id: str, questions_fp: str, reranker_fp: str | None) -> dict[str, VariantResult]:
    """Saved variants of a cell, or none when the file is missing or was made from other inputs (any reranker when\n    eranker_fp is None, which the report uses so that it never loads the model)."""
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if (data["run_id"], data["questions"]) != (run_id, questions_fp) or reranker_fp not in (None, data["reranker"]):
        return {}
    return {v: VariantResult([QResult(q, first, top, rr, rec) for q, first, top, rr, rec in item["results"]],
                             item["seconds_per_question"], item["truncated_pairs"])
            for v, item in data["variants"].items()}


# --- the report --------------------------------------------------------------------------------

def render_cell_report(name: str, variants: dict[str, VariantResult], groups: list[tuple[str, list[Question]]]) -> str:
    missing = [v for v in PIPELINES if v not in variants]
    if missing:
        raise ValueError(f"{name}: no results for {', '.join(missing)}; run stage 3 first")
    lines, start = [], 0
    for title, questions in groups:
        stop = start + len(questions)
        runs = {f"{name} {v}": variants[v].results[start:stop] for v in PIPELINES}
        for results in runs.values():
            if [r.q for r in results] != [q.q for q in questions]:
                raise ValueError(f"{name}: the saved results are for other questions; run stage 3 again")
        lines += [f"\n{title}, {name}", markdown_table(runs, baseline=f"{name} V")]
        start = stop
    lines.append(f"\nSeconds per question, {name}: " + ", ".join(f"{v} {variants[v].seconds_per_question:.3f}" for v in PIPELINES))
    lines.append("Pairs cut to fit the reranker limit: " + ", ".join(f"{v} {variants[v].truncated_pairs}" for v in RERANKED))
    return "\n".join(lines)


# --- the estimate ------------------------------------------------------------------------------

def estimate_cell(run: Run, texts: list[str], reranker, n_questions: int, clock=time.perf_counter) -> dict:
    start = clock()
    keyword_index(run, texts)
    index_seconds = clock() - start
    sample = texts[:SAMPLE_PAIRS]
    start = clock()
    reranker.score("how do I do this", sample)
    seconds = clock() - start
    rate = len(sample) / seconds if seconds > 0 else float("inf")
    pairs = n_questions * RERANK_DEPTH * len(RERANKED)
    return {"passages": len(texts), "questions": n_questions, "pairs": pairs, "index_seconds": index_seconds,
            "pairs_per_second": rate, "rerank_minutes": pairs / rate / 60 if rate != float("inf") else 0.0}


def render_estimate(cells: dict[str, dict], depth: int = RERANK_DEPTH) -> str:
    lines = ["Stage 3 estimate (local models only, no money spent)"]
    for name, e in cells.items():
        lines.append(f"- {name}: {e['passages']} passages, {e['questions']} questions; keyword index in "
                     f"{e['index_seconds']:.1f} s; up to {e['pairs']} pairs to rerank ({e['questions']} x {depth} x "
                     f"{len(RERANKED)} variants), about {e['pairs_per_second']:.0f} pairs/s measured on a sample, "
                     f"so about {e['rerank_minutes']:.1f} min")
    total = sum(e["rerank_minutes"] for e in cells.values())
    lines.append(f"Total reranking time: about {total:.1f} min, plus a few minutes for the vector and keyword steps.")
    return "\n".join(lines)


# --- the command -------------------------------------------------------------------------------

def make_reranker(reranker_id: str, path: Path):
    if reranker_id == "fake":
        return FakeReranker(), "fake"
    if reranker_id not in RERANKER_IDS:
        raise ValueError(f"unknown reranker {reranker_id!r}; choose one of {', '.join(RERANKER_IDS)}")
    from ragcore.rerank import HFReranker  # Needs the `embed` dependency group.
    reranker = HFReranker(load_reranker_specs(path)[reranker_id])
    return reranker, reranker.fingerprint


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 3: hybrid search and reranking (spec 15).")
    parser.add_argument("command", choices=("estimate", "run", "report"))
    parser.add_argument("--register")
    parser.add_argument("--snapshots")
    parser.add_argument("--questions")
    parser.add_argument("--rules")
    parser.add_argument("--runs", default=str(DEFAULT_RUNS))
    parser.add_argument("--models", default=str(DEFAULT_MODELS))
    parser.add_argument("--rerankers", default=str(DEFAULT_RERANKERS))
    parser.add_argument("--reranker", default="bge-reranker-v2-m3")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="where per-question results are saved")
    parser.add_argument("--cells", default=DEFAULT_CELLS, help="embedder:chunker pairs, comma separated")
    parser.add_argument("--exclusions", default="", help="corpus exclusions file (default: configs/corpus-exclusions.yaml)")
    parser.add_argument("--only", default="")
    parser.add_argument("--max-chars", type=int, default=ChunkConfig.max_chars)
    parser.add_argument("--hard-limit", type=int, default=ChunkConfig.hard_limit)
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
    except SystemExit as exc:
        return int(exc.code or 0)
    if not (args.register and args.snapshots and args.questions):
        print("Give --register, --snapshots and --questions.", file=sys.stderr)
        return 2
    try:
        config = ChunkConfig(args.max_chars, args.hard_limit)
        cells = parse_cells(args.cells)
        sets = load_question_dir(Path(args.questions))
        sources, notes = load_corpus(Path(args.register), Path(args.snapshots), Path(args.rules) if args.rules else None,
                                     _csv(args.only) or None, _exclusions(args.exclusions))
        if not sources:
            raise ValueError("no sources loaded: " + "; ".join(notes))
    except (ValueError, RegisterError, OSError) as exc:
        print(f"Cannot start: {exc}", file=sys.stderr)
        return 2
    primary, secondary = split_sets(sets)
    groups = [(title, [q for s in group for q in s.questions])
              for title, group in (("Primary questions", primary), ("Secondary questions (rag-questions)", secondary))]
    groups = [(title, questions) for title, questions in groups if questions]
    questions = [q for _, group in groups for q in group]
    corpus_fp, questions_fp = corpus_fingerprint(sources), questions_fingerprint(Path(args.questions))
    out_dir = Path(args.out)
    try:
        made = {e: make_embedder(e, Path(args.models)) for e in dict.fromkeys(e for e, _ in cells)}
        reranker, reranker_fp = make_reranker(args.reranker, Path(args.rerankers)) if args.command != "report" else (None, None)
        estimates = {}
        for embedder_id, chunker in cells:
            name = f"{embedder_id}/{chunker}"
            run = load_cell_run(Path(args.runs), made[embedder_id], chunker, config, corpus_fp, questions_fp)
            texts = passage_texts(run, sources, corpus_fp)
            path = _cell_path(out_dir, embedder_id, chunker)
            if args.command == "estimate":
                estimates[name] = estimate_cell(run, texts, reranker, len(questions))
            elif args.command == "run":
                done = load_cell(path, run.run_id, questions_fp, reranker_fp)
                _safe_print(f"{name}: {len(done)} of {len(PIPELINES)} variants already saved")
                evaluate_variants(run, texts, made[embedder_id], reranker, questions, done=done,
                                  save=lambda all_, p=path, r=run.run_id: save_cell(p, r, questions_fp, reranker_fp, all_))
                _safe_print(f"{name}: done, saved to {path}")
            else:
                saved = load_cell(path, run.run_id, questions_fp, reranker_fp)
                _safe_print(render_cell_report(name, saved, groups))
        if args.command == "estimate":
            _safe_print(render_estimate(estimates))
            _safe_print("No result was written.")
    except (RunError, ValueError) as exc:
        print(f"Cannot continue: {exc}", file=sys.stderr)
        return 1
    return 0

