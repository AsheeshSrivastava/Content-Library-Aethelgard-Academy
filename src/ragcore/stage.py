"""Stage runner (spec 13 section 6): load the corpus, build runs, print the estimate first, report the comparison.

    uv run python scripts/run_stage.py estimate|run|report --register DIR --snapshots DIR --questions DIR ...
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from ragcore.chunking import (CHILD_CHARS, ChunkConfig, chunk_fixed, chunk_merge_small, chunk_parent_child,
                              chunk_source)
from ragcore.compare import markdown_table, score_questions, split_sets
from ragcore.contracts import Passage, Source
from ragcore.embed.base import FakeEmbedder, estimate_seconds, load_model_specs, probe
from ragcore.errors import LoadRefused, RegisterError, SkipShareExceeded
from ragcore.evalset import QuestionSet, load_question_dir
from ragcore.gate import LicenceRules
from ragcore.loaders.snapshot import load_snapshot
from ragcore.register import load_record
from ragcore.retrieve import make_retriever
from ragcore.runs import RunError, compute_run_id, load_run, write_run

CHUNKER_NAMES = ("rules", "fixed", "merge-small", "parent-child")
MERGE_MIN_CHARS = 300
DEFAULT_RUNS = Path.home() / ".ragcore" / "runs"
DEFAULT_MODELS = Path(__file__).resolve().parent.parent.parent / "configs" / "models.yaml"
EMBEDDER_IDS = ("fake", "bge-m3", "jina-v2-base-en")


# --- chunking ----------------------------------------------------------------------------------

def chunker_label(name: str, config: ChunkConfig) -> str:
    if name == "rules":
        return config.version
    if name == "fixed":
        return f"fixed-1/{config.max_chars}/{config.hard_limit}"
    if name == "merge-small":
        return f"merge-small-1/{config.max_chars}/{config.hard_limit}/{MERGE_MIN_CHARS}"
    if name == "parent-child":
        return f"parent-child-1/{config.max_chars}/{config.hard_limit}/{CHILD_CHARS}"
    raise ValueError(f"unknown chunker {name!r}; choose one of {', '.join(CHUNKER_NAMES)}")


def chunk_corpus_full(sources: list[Source], name: str, config: ChunkConfig) -> tuple[list[Passage], int, dict]:
    """(passages to embed, dropped, parent map). The map is empty except for parent-child (child id -> parent id)."""
    chunkers = {"rules": lambda s: chunk_source(s, config), "fixed": lambda s: chunk_fixed(s, config),
                "merge-small": lambda s: chunk_merge_small(s, config, MERGE_MIN_CHARS)}
    if name not in chunkers and name != "parent-child":
        chunker_label(name, config)  # raises the "unknown chunker" message
    passages, dropped, parents = [], 0, {}
    for source in sorted(sources, key=lambda s: s.source_id):
        if name == "parent-child":
            pc = chunk_parent_child(source, config, CHILD_CHARS)
            passages += pc.children
            dropped += pc.dropped_empty
            parents.update(pc.parent_of)
        else:
            result = chunkers[name](source)
            passages += result.passages
            dropped += result.dropped_empty
    return passages, dropped, parents


def chunk_corpus(sources: list[Source], name: str, config: ChunkConfig) -> tuple[list[Passage], int]:
    passages, dropped, _ = chunk_corpus_full(sources, name, config)
    return passages, dropped


# --- the corpus and the questions --------------------------------------------------------------

def corpus_fingerprint(sources: list[Source]) -> str:
    lines = sorted(f"{s.source_id}\0{s.version}\0{s.text_sha256}" for s in sources)
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()[:16]


def questions_fingerprint(folder: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(Path(folder).glob("*.yaml")):
        digest.update(path.name.encode("utf-8") + b"\0" + hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()[:16]


def load_corpus(register: Path, snapshots: Path, rules_path: Path | None = None,
                only: list[str] | None = None) -> tuple[list[Source], list[str]]:
    register, snapshots = Path(register), Path(snapshots)
    rules = LicenceRules.load(Path(rules_path) if rules_path else register.parent / "licence-rules.yaml")
    records = [load_record(p) for p in sorted(register.glob("*.yaml"))]
    if only:
        unknown = sorted(set(only) - {r.id for r in records})
        if unknown:
            raise ValueError(f"not in the register: {', '.join(unknown)}")
        records = [r for r in records if r.id in only]
    sources, notes = [], []
    for record in records:
        folder = snapshots / record.id / record.version_ref
        if not folder.is_dir():
            notes.append(f"{record.id}: not on disk")
            continue
        try:
            sources += load_snapshot(folder, record, rules).sources
        except (LoadRefused, SkipShareExceeded) as exc:
            notes.append(f"{record.id}: not loaded ({exc})")
    return sorted(sources, key=lambda s: s.source_id), notes


# --- runs --------------------------------------------------------------------------------------

@dataclass
class BuiltRun:
    run_id: str
    count: int
    dropped_empty: int = 0
    reused: bool = False
    seconds: float = 0.0
    passages: list = field(default_factory=list)  # Empty when the run was reused.


def run_inputs(name: str, config: ChunkConfig, embedder, corpus_fp: str, questions_fp: str, mode: str = "normal") -> dict:
    return {"chunker": chunker_label(name, config), "mode": mode, "embedder": embedder.fingerprint,
            "corpus": corpus_fp, "questions": questions_fp}


def build_run(root: Path, name: str, sources: list[Source], embedder, config: ChunkConfig, *, questions_fp: str,
              batch: int = 256, mode: str = "normal",
              progress: Callable[[int, int], None] | None = None) -> BuiltRun:
    inputs = run_inputs(name, config, embedder, corpus_fingerprint(sources), questions_fp, mode)
    run_id = compute_run_id(inputs)
    try:
        existing = load_run(root, run_id)
        return BuiltRun(run_id, existing.manifest["count"], existing.manifest["extra"].get("dropped_empty", 0), True)
    except RunError:
        pass
    passages, dropped, parents = chunk_corpus_full(sources, name, config)
    if not passages:
        raise ValueError("nothing to embed: the chunker produced no passages")
    start = time.perf_counter()
    texts = [p.text for p in passages]
    parts = []
    for i in range(0, len(texts), batch):
        parts.append(embedder.embed(texts[i:i + batch], kind="passage"))
        if progress:
            progress(min(i + batch, len(texts)), len(texts))
    seconds = time.perf_counter() - start
    extra = {"dropped_empty": dropped, "embed_seconds": round(seconds, 2), "passages": len(passages)}
    if hasattr(embedder, "truncated"):
        extra["truncated"] = embedder.truncated
    write_run(root, inputs, passages, np.vstack(parts).astype(np.float32), parents=parents, extra=extra)
    return BuiltRun(run_id, len(passages), dropped, False, seconds, passages)


# --- the estimate printed before anything runs -------------------------------------------------

def estimate(sources: list[Source], chunkers: list[str], embedders: list, config: ChunkConfig, root: Path, *,
             questions_fp: str, probe_n: int = 256, clock: Callable[[], float] = time.perf_counter) -> dict:
    if not sources:
        raise ValueError("nothing to estimate: the corpus is empty")
    corpus_fp = corpus_fingerprint(sources)
    chunked = {name: chunk_corpus(sources, name, config)[0] for name in chunkers}
    sample = [p.text for p in chunked[chunkers[0]][:probe_n]]
    probes = {e.id: probe(e, sample, clock=clock) for e in embedders}
    runs = []
    for name in chunkers:
        passages = chunked[name]
        for emb in embedders:
            tokens = sum(emb.count_tokens(p.text) for p in passages)
            exists = True
            try:
                load_run(root, compute_run_id(run_inputs(name, config, emb, corpus_fp, questions_fp)))
            except RunError:
                exists = False
            runs.append({"name": f"{emb.id}/{name}", "chunker": name, "embedder": emb.id, "passages": len(passages),
                         "tokens": tokens, "exists": exists, "disk_mb": len(passages) * emb.dim * 4 / 1e6,
                         "seconds": 0.0 if exists else estimate_seconds(tokens, probes[emb.id]["tokens_per_second"])})
    return {"runs": runs, "probe": probes, "sources": len(sources), "total_runs": len(runs),
            "to_build": sum(not r["exists"] for r in runs), "total_seconds": sum(r["seconds"] for r in runs),
            "total_disk_mb": sum(r["disk_mb"] for r in runs if not r["exists"])}


def render_estimate(result: dict) -> str:
    lines = [f"Corpus: {result['sources']} sources. Runs planned: {result['total_runs']}, to build: {result['to_build']} "
             f"(the others already exist and are skipped)."]
    for p_id, p in result["probe"].items():
        lines.append(f"Speed sample for {p_id}: {p['texts']} passages, {p['tokens']} tokens in {p['seconds']:.1f} s "
                     f"= {p['tokens_per_second']:.0f} tokens per second.")
    for r in result["runs"]:
        state = "exists, skipped" if r["exists"] else f"about {r['seconds'] / 60:.1f} minutes"
        lines.append(f"  {r['name']}: {r['passages']} passages, {r['tokens']} tokens, {r['disk_mb']:.0f} MB; {state}")
    lines.append(f"Total: about {result['total_seconds'] / 60:.1f} minutes and {result['total_disk_mb']:.0f} MB of disk. "
                 "No paid service is used; everything runs on this computer.")
    return "\n".join(lines)


# --- the report --------------------------------------------------------------------------------

def evaluate_runs(root: Path, run_ids: dict[str, str], embedder, sets: list[QuestionSet], *, k: int = 3) -> dict:
    """Per-question results for each named run; all runs must have been made with `embedder`."""
    questions = [q for s in sets for q in s.questions]
    return {name: score_questions(questions, make_retriever(load_run(root, run_id), embedder), k=k)
            for name, run_id in run_ids.items()}


# --- the command -------------------------------------------------------------------------------

def make_embedder(embedder_id: str, models_path: Path):
    if embedder_id == "fake":
        return FakeEmbedder()
    if embedder_id not in EMBEDDER_IDS:
        raise ValueError(f"unknown embedder {embedder_id!r}; choose one of {', '.join(EMBEDDER_IDS)}")
    specs = load_model_specs(models_path)
    from ragcore.embed.hf import HFEncoder  # Needs the `embed` dependency group.
    return HFEncoder(specs[embedder_id])


def _csv(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a stage of the chunker and embedder comparison (spec 13).")
    parser.add_argument("command", choices=("estimate", "run", "report"))
    parser.add_argument("--register")
    parser.add_argument("--snapshots")
    parser.add_argument("--questions")
    parser.add_argument("--rules")
    parser.add_argument("--runs", default=str(DEFAULT_RUNS))
    parser.add_argument("--models", default=str(DEFAULT_MODELS))
    parser.add_argument("--embedders", default="bge-m3,jina-v2-base-en")
    parser.add_argument("--chunkers", default=",".join(CHUNKER_NAMES))
    parser.add_argument("--only", default="", help="comma list of source ids (stage 0 uses one small source)")
    parser.add_argument("--baseline", default="rules")
    parser.add_argument("--max-chars", type=int, default=ChunkConfig.max_chars)
    parser.add_argument("--hard-limit", type=int, default=ChunkConfig.hard_limit)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--probe-n", type=int, default=256)
    return parser


def _safe_print(text: str) -> None:
    if getattr(sys.stdout, "encoding", None) and sys.stdout.encoding.lower().replace("-", "") != "utf8":
        sys.stdout.reconfigure(errors="replace")  # A narrow console must not stop the run.
    print(text, flush=True)


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
        chunkers = _csv(args.chunkers)
        for name in chunkers:
            chunker_label(name, config)
        if not chunkers:
            raise ValueError("no chunkers chosen")
        embedder_ids = _csv(args.embedders)
        unknown = [e for e in embedder_ids if e not in EMBEDDER_IDS]
        if unknown or not embedder_ids:
            raise ValueError(f"unknown embedder {', '.join(unknown) or '(none)'}; choose from {', '.join(EMBEDDER_IDS)}")
        sets = load_question_dir(Path(args.questions))
        sources, notes = load_corpus(Path(args.register), Path(args.snapshots), Path(args.rules) if args.rules else None,
                                     _csv(args.only) or None)
        if not sources:
            raise ValueError("no sources loaded: " + "; ".join(notes))
    except (ValueError, RegisterError, OSError) as exc:
        print(f"Cannot start: {exc}", file=sys.stderr)
        return 2
    for note in notes:
        _safe_print(f"  note: {note}")
    questions_fp, root = questions_fingerprint(Path(args.questions)), Path(args.runs)
    embedders = {e: make_embedder(e, Path(args.models)) for e in embedder_ids}
    if args.command == "estimate":
        _safe_print(render_estimate(estimate(sources, chunkers, list(embedders.values()), config, root,
                                             questions_fp=questions_fp, probe_n=args.probe_n)))
        _safe_print("No run was written.")
        return 0
    if args.command == "run":
        for emb_id, emb in embedders.items():
            for name in chunkers:
                built = build_run(root, name, sources, emb, config, questions_fp=questions_fp, batch=args.batch)
                state = "reused" if built.reused else f"built in {built.seconds / 60:.1f} min"
                _safe_print(f"{emb_id}/{name}: {built.count} passages, {state} [{built.run_id}]")
        return 0
    corpus_fp = corpus_fingerprint(sources)
    primary, secondary = split_sets(sets)
    try:
        for emb_id, emb in embedders.items():
            ids = {f"{emb_id}/{n}": compute_run_id(run_inputs(n, config, emb, corpus_fp, questions_fp)) for n in chunkers}
            for title, group in (("Primary questions", primary), ("Secondary questions (rag-questions)", secondary)):
                if not any(s.questions for s in group):
                    continue
                results = evaluate_runs(root, ids, emb, group)
                baseline = f"{emb_id}/{args.baseline}"
                _safe_print(f"\n{title}, {emb_id}\n" + markdown_table(results, baseline=baseline))
    except (RunError, ValueError) as exc:
        print(f"Cannot report: {exc}", file=sys.stderr)
        return 1
    return 0
