"""Stage 4 (spec 16): passage-level questions on the cells of stage 3, for V (vector only) and HR (hybrid + reranker).

`estimate` prints what will be reranked; `run` scores both pipelines on every question and saves the results;
`report` prints the paired tables, the two cells against each other, the kinds of question and the quotes the chunking
cut. Nothing here builds a run or calls a paid service."""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
from collections import defaultdict
from pathlib import Path

from ragcore.chunking import ChunkConfig
from ragcore.compare import markdown_table
from ragcore.errors import RegisterError
from ragcore.passage_eval import QuoteSpan, cut_flags, quote_span, score_passage_questions
from ragcore.passage_questions import PassageQuestion, check_mix, check_questions, load_passage_questions
from ragcore.retrieve import RERANK_DEPTH, keyword_index, make_passage_pipeline
from ragcore.runs import Run, RunError
from ragcore.stage import (DEFAULT_MODELS, DEFAULT_RUNS, _csv, _exclusions, _safe_print, corpus_fingerprint, load_corpus,
                           make_embedder, questions_fingerprint)
from ragcore.stage3 import (DEFAULT_CELLS, DEFAULT_RERANKERS, SAMPLE_PAIRS, VariantResult, load_cell, load_cell_run,
                            make_reranker, parse_cells, passage_texts, save_cell)

DEFAULT_OUT = Path.home() / ".ragcore" / "stage4"
DEFAULT_QUESTIONS = Path(__file__).resolve().parent.parent.parent / "eval" / "passage-questions"
VARIANTS = ("V", "HR")
SHOWN_PROBLEMS = 10


def prepare(folder: Path, sources: dict[str, str], check_set: bool = True) -> tuple[list[PassageQuestion], list[QuoteSpan]]:
    """Load the passage-level questions and where their quotes sit; refuse a set with any problem."""
    questions = load_passage_questions(folder)
    problems = check_questions(questions, sources) + (check_mix(questions, sources) if check_set else [])
    if problems:
        more = f" (and {len(problems) - SHOWN_PROBLEMS} more)" if len(problems) > SHOWN_PROBLEMS else ""
        raise ValueError(f"{len(problems)} problems in the passage-level questions: "
                         + "; ".join(problems[:SHOWN_PROBLEMS]) + more)
    return questions, [quote_span(q, sources) for q in questions]


def passage_questions_fingerprint(folder: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(Path(folder).glob("*.yaml")):
        digest.update(path.name.encode("utf-8") + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()[:16]


def evaluate_cell(run: Run, texts: list[str], embedder, reranker, questions: list[PassageQuestion],
                  spans: list[QuoteSpan], done: dict[str, VariantResult] | None = None, save=None,
                  clock=time.perf_counter) -> dict[str, VariantResult]:
    out = dict(done or {})
    index = keyword_index(run, texts)
    for variant in VARIANTS:
        if variant in out:
            continue
        retriever = make_passage_pipeline(variant, run, texts, embedder=embedder, reranker=reranker, index=index)
        before = getattr(reranker, "truncated", 0)
        start = clock()
        results = score_passage_questions(questions, spans, retriever)
        seconds = clock() - start
        out[variant] = VariantResult(results, seconds / max(1, len(questions)), getattr(reranker, "truncated", 0) - before)
        if save:
            save(out)
    return out


# --- the report --------------------------------------------------------------------------------

def _check_cell(name: str, variants: dict[str, VariantResult], questions: list[PassageQuestion]) -> None:
    missing = [v for v in VARIANTS if v not in variants]
    if missing:
        raise ValueError(f"{name}: no results for {', '.join(missing)}; run stage 4 first")
    for v in VARIANTS:
        if [r.q for r in variants[v].results] != [q.q for q in questions]:
            raise ValueError(f"{name}: the saved results are for other questions; run stage 4 again")


def _kinds(name: str, variants: dict[str, VariantResult], questions: list[PassageQuestion]) -> str:
    rows = defaultdict(list)
    for i, q in enumerate(questions):
        rows[q.kind].append(i)
    lines = [f"| Kind ({name}) | Questions | V right first | V top 3 | HR right first | HR top 3 |", "|---|---|---|---|---|---|"]
    for kind in sorted(rows):
        pick = lambda v, f: sum(getattr(variants[v].results[i], f) for i in rows[kind])
        lines.append(f"| {kind} | {len(rows[kind])} | {pick('V', 'first')} | {pick('V', 'top_k')} | "
                     f"{pick('HR', 'first')} | {pick('HR', 'top_k')} |")
    return "\n".join(lines)


def render_report(cells: dict[str, dict[str, VariantResult]], questions: list[PassageQuestion],
                  cuts: dict[str, list[bool]]) -> str:
    for name, variants in cells.items():
        _check_cell(name, variants, questions)
    lines = []
    for name, variants in cells.items():
        runs = {f"{name} {v}": variants[v].results for v in VARIANTS}
        lines += [f"\nPassage-level questions ({len(questions)}), {name}", markdown_table(runs, baseline=f"{name} V"),
                  "", _kinds(name, variants, questions)]
    names = list(cells)
    if len(names) > 1:
        for v, title in (("HR", "hybrid + reranker"), ("V", "vector only")):
            runs = {f"{n} {v}": cells[n][v].results for n in names}
            lines += [f"\nBetween the cells, {title} (first cell is the baseline)", markdown_table(runs, baseline=f"{names[0]} {v}")]
    lines.append("\nQuote cut by the chunking (no passage holds the whole quote), questions affected: "
                 + ", ".join(f"{n}: {sum(cuts[n])} of {len(cuts[n])}" for n in names))
    for name, variants in cells.items():
        lines.append(f"Seconds per question, {name}: " + ", ".join(f"{v} {variants[v].seconds_per_question:.3f}" for v in VARIANTS))
    lines.append("Pairs cut to fit the reranker limit: " + ", ".join(f"{n} {cells[n]['HR'].truncated_pairs}" for n in names))
    return "\n".join(lines)


# --- the estimate ------------------------------------------------------------------------------

def estimate_cell(run: Run, texts: list[str], reranker, n_questions: int, clock=time.perf_counter) -> dict:
    start = clock()
    keyword_index(run, texts)
    index_seconds = clock() - start
    start = clock()
    reranker.score("how do I do this", texts[:SAMPLE_PAIRS])
    seconds = clock() - start
    sample = min(SAMPLE_PAIRS, len(texts))
    rate = sample / seconds if seconds > 0 else float("inf")
    pairs = n_questions * RERANK_DEPTH
    return {"passages": len(texts), "questions": n_questions, "pairs": pairs, "index_seconds": index_seconds,
            "pairs_per_second": rate, "rerank_minutes": pairs / rate / 60 if rate != float("inf") else 0.0}


def render_estimate(cells: dict[str, dict]) -> str:
    lines = ["Stage 4 estimate (local models only, no money spent)"]
    for name, e in cells.items():
        lines.append(f"- {name}: {e['passages']} passages, {e['questions']} questions; keyword index in "
                     f"{e['index_seconds']:.1f} s; up to {e['pairs']} pairs to rerank, about "
                     f"{e['pairs_per_second']:.0f} pairs/s measured on a sample, so about {e['rerank_minutes']:.1f} min")
    lines.append(f"Total reranking time: about {sum(e['rerank_minutes'] for e in cells.values()):.1f} min.")
    return "\n".join(lines)


# --- the command -------------------------------------------------------------------------------

def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 4: passage-level questions (spec 16).")
    parser.add_argument("command", choices=("estimate", "run", "report"))
    parser.add_argument("--register")
    parser.add_argument("--snapshots")
    parser.add_argument("--questions", help="the file-level questions folder (it names the stage 1 runs)")
    parser.add_argument("--passage-questions", default=str(DEFAULT_QUESTIONS))
    parser.add_argument("--rules")
    parser.add_argument("--runs", default=str(DEFAULT_RUNS))
    parser.add_argument("--models", default=str(DEFAULT_MODELS))
    parser.add_argument("--rerankers", default=str(DEFAULT_RERANKERS))
    parser.add_argument("--reranker", default="bge-reranker-v2-m3")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--cells", default=DEFAULT_CELLS)
    parser.add_argument("--exclusions", default="")
    parser.add_argument("--only", default="")
    parser.add_argument("--max-chars", type=int, default=ChunkConfig.max_chars)
    parser.add_argument("--hard-limit", type=int, default=ChunkConfig.hard_limit)
    parser.add_argument("--small-set-for-tests", action="store_true", help="skip the mix rule of the full set (tests only)")
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
        sources, notes = load_corpus(Path(args.register), Path(args.snapshots), Path(args.rules) if args.rules else None,
                                     _csv(args.only) or None, _exclusions(args.exclusions))
        if not sources:
            raise ValueError("no sources loaded: " + "; ".join(notes))
        questions, spans = prepare(Path(args.passage_questions), {s.source_id: s.text for s in sources},
                                   check_set=not args.small_set_for_tests)
    except (ValueError, RegisterError, OSError) as exc:
        print(f"Cannot start: {exc}", file=sys.stderr)
        return 2
    corpus_fp, questions_fp = corpus_fingerprint(sources), questions_fingerprint(Path(args.questions))
    set_fp = passage_questions_fingerprint(Path(args.passage_questions))
    out_dir = Path(args.out)
    try:
        made = {e: make_embedder(e, Path(args.models)) for e in dict.fromkeys(e for e, _ in cells)}
        reranker, reranker_fp = make_reranker(args.reranker, Path(args.rerankers)) if args.command != "report" else (None, None)
        estimates, results, cuts = {}, {}, {}
        for embedder_id, chunker in cells:
            name = f"{embedder_id}/{chunker}"
            run = load_cell_run(Path(args.runs), made[embedder_id], chunker, config, corpus_fp, questions_fp)
            texts = passage_texts(run, sources, corpus_fp)
            path = out_dir / f"{embedder_id}__{chunker}.json"
            if args.command == "estimate":
                estimates[name] = estimate_cell(run, texts, reranker, len(questions))
            elif args.command == "run":
                done = load_cell(path, run.run_id, set_fp, reranker_fp)
                _safe_print(f"{name}: {len(done)} of {len(VARIANTS)} variants already saved")
                evaluate_cell(run, texts, made[embedder_id], reranker, questions, spans, done=done,
                              save=lambda all_, p=path, r=run.run_id: save_cell(p, r, set_fp, reranker_fp, all_))
                _safe_print(f"{name}: done, saved to {path}")
            else:
                results[name] = load_cell(path, run.run_id, set_fp, None)
                cuts[name] = cut_flags(run, spans)
        if args.command == "estimate":
            _safe_print(render_estimate(estimates))
            _safe_print("No result was written.")
        elif args.command == "report":
            _safe_print(render_report(results, questions, cuts))
    except (RunError, ValueError) as exc:
        print(f"Cannot continue: {exc}", file=sys.stderr)
        return 1
    return 0
