"""The index commands (spec 17 sections 5 and 7): sync, status, search and check, run through `scripts/index.ps1`.

    uv run --group embed --group variants python scripts/run_index.py sync|status|search|check ...

Models and the chunker are built only when a command needs them: a sync with nothing new, and `status`, load none.
Exit code 0 = done, 1 = could not continue (the index is unchanged or still valid), 2 = bad input."""
from __future__ import annotations

import argparse
import contextlib
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

from ragcore.chunking import ChunkConfig
from ragcore.contracts import Source
from ragcore.errors import RegisterError
from ragcore.index_retriever import Retriever
from ragcore.index_store import IndexSetup, IndexStoreError, open_index
from ragcore.index_sync import SyncError, sync
from ragcore.passage_eval import QuoteSpan, score_passage_questions
from ragcore.passage_questions import PassageQuestion
from ragcore.retrieve import PassageHit

DEFAULT_INDEX = Path.home() / ".ragcore" / "index" / "index.db"
DEFAULT_EMBEDDER = "jina-v2-base-en"
DEFAULT_RERANKER = "bge-reranker-v2-m3"
DEFAULT_QUESTIONS = Path(__file__).resolve().parent.parent.parent / "eval" / "passage-questions"
STAGE4_FIRST, STAGE4_TOP3 = 100, 121  # jina-v2-base-en + llamaindex + HR on the 137 questions (report section 9.2)
PROGRESS_EVERY = 100
SNIPPET_CHARS = 200


@dataclass(frozen=True)
class Services:
    """Everything that touches the corpus, the models or LlamaIndex, so the commands can be tested without them."""
    load_sources: Callable[[argparse.Namespace], tuple[list[Source], dict[str, str], list[str]]]
    setup: Callable[[argparse.Namespace], IndexSetup]
    make_chunker: Callable[[argparse.Namespace], Callable[[Source], object]]  # callable(source) -> ChunkResult, + verify()
    make_embedder: Callable[[argparse.Namespace], object]
    make_reranker: Callable[[argparse.Namespace], object]
    load_questions: Callable[[argparse.Namespace], tuple[list[PassageQuestion], list[QuoteSpan]]] | None = None


class _Lazy:
    def __init__(self, build: Callable[[], object]) -> None:
        self._build, self._made = build, None

    @property
    def made(self) -> bool:
        return self._made is not None

    def get(self):
        if self._made is None:
            self._made = self._build()
        return self._made


class _LazyEmbedder:
    def __init__(self, build: Callable[[], object]) -> None:
        self._lazy = _Lazy(build)

    def embed(self, texts, kind="passage"):
        return self._lazy.get().embed(texts, kind=kind)


def real_services() -> Services:
    from ragcore.embed.base import load_model_specs
    from ragcore.llamaindex_chunker import LlamaIndexChunker, llamaindex_version
    from ragcore.register import load_record
    from ragcore.rerank import load_reranker_specs
    from ragcore.stage import _exclusions, load_corpus, make_embedder
    from ragcore.stage4 import prepare
    from ragcore.stage3 import make_reranker

    config = ChunkConfig()

    def load_sources(args):
        register = Path(args.register)
        sources, notes = load_corpus(register, Path(args.snapshots), Path(args.rules) if args.rules else None, None,
                                     _exclusions(args.exclusions))
        urls = {record.id: record.url for record in (load_record(p) for p in sorted(register.glob("*.yaml")))}
        return sources, urls, notes

    def load_questions(args):
        sources, notes = load_corpus(Path(args.register), Path(args.snapshots), Path(args.rules) if args.rules else None,
                                     None, _exclusions(args.exclusions))
        if not sources:
            raise ValueError("no sources loaded: " + "; ".join(notes))
        return prepare(Path(args.passage_questions), {s.source_id: s.text for s in sources})

    def setup(args):
        try:
            llama = metadata.version("llama-index-core")
        except metadata.PackageNotFoundError:
            raise ValueError("LlamaIndex is not installed: run with `--group variants`") from None
        model = load_model_specs(Path(args.models))[args.embedder]
        reranker = load_reranker_specs(Path(args.rerankers))[args.reranker]
        revision = model.revision + (f"+{model.code_revision}" if model.code_revision else "")
        return IndexSetup(model.id, revision, "llamaindex", llamaindex_version(config),
                          f"{config.max_chars}/{config.hard_limit}", llama, reranker.id, reranker.revision)

    class Bound:
        def __init__(self) -> None:
            self.chunker = LlamaIndexChunker()

        def __call__(self, source):
            return self.chunker(source, config)

        def verify(self) -> None:
            self.chunker.verify()

    return Services(load_sources, setup, lambda args: Bound(), lambda args: make_embedder(args.embedder, Path(args.models)),
                    lambda args: make_reranker(args.reranker, Path(args.rerankers))[0], load_questions)


def _say(text: str = "") -> None:
    if getattr(sys.stdout, "encoding", None) and sys.stdout.encoding.lower().replace("-", "") != "utf8":
        sys.stdout.reconfigure(errors="replace")  # A narrow console must not stop the command.
    print(text, flush=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sync, inspect and search the index (spec 17).")
    parser.add_argument("command", choices=("sync", "status", "search", "check"))
    parser.add_argument("question", nargs="?", help="search: the question")
    parser.add_argument("--index", default=str(DEFAULT_INDEX))
    parser.add_argument("--register")
    parser.add_argument("--snapshots")
    parser.add_argument("--rules")
    parser.add_argument("--exclusions", default="", help="corpus exclusions file (default: configs/corpus-exclusions.yaml)")
    parser.add_argument("--models", default=str(Path(__file__).resolve().parent.parent.parent / "configs" / "models.yaml"))
    parser.add_argument("--rerankers", default=str(Path(__file__).resolve().parent.parent.parent / "configs" / "rerankers.yaml"))
    parser.add_argument("--embedder", default=DEFAULT_EMBEDDER)
    parser.add_argument("--reranker", default=DEFAULT_RERANKER)
    parser.add_argument("--batch", type=int, default=256, help="sync: texts embedded at once (the comparison used 256)")
    parser.add_argument("--dry-run", action="store_true", help="sync: show what would change, change nothing")
    parser.add_argument("--allow-mass-delete", action="store_true",
                        help="sync: allow removing half or more of the stored sources")
    parser.add_argument("--passage-questions", default=str(DEFAULT_QUESTIONS), help="check: the frozen questions folder")
    parser.add_argument("--expect-first", type=int, default=STAGE4_FIRST,
                        help="check: right-first count the stage 4 result gave")
    parser.add_argument("--expect-top3", type=int, default=STAGE4_TOP3,
                        help="check: right-in-first-3 count the stage 4 result gave")
    parser.add_argument("-k", type=int, default=5, help="search: how many passages")
    parser.add_argument("--max-per-file", type=int, default=None, help="search: at most this many passages per source")
    return parser


def _plain(exc: Exception) -> str:
    return str(exc).replace("allow_mass_delete", "--allow-mass-delete")


def _sync(args, services: Services) -> int:
    if not (args.register and args.snapshots):
        print("Give --register and --snapshots.", file=sys.stderr)
        return 2
    sources, urls, notes = services.load_sources(args)
    for note in notes:
        _say(note)
    if not sources:
        print("Cannot start: no sources loaded" + (": " + "; ".join(notes) if notes else ""), file=sys.stderr)
        return 2
    setup = services.setup(args)
    chunker = _Lazy(lambda: services.make_chunker(args))
    embedder = _LazyEmbedder(lambda: services.make_embedder(args))

    def progress(done: int, total: int, source_id: str) -> None:
        if done % PROGRESS_EVERY == 0 or done == total:
            _say(f"  {done} of {total} sources")

    path = Path(args.index)
    with contextlib.ExitStack() as stack:
        if args.dry_run and not path.exists():  # A dry run must not leave a file behind.
            path = Path(stack.enter_context(tempfile.TemporaryDirectory())) / "index.db"
        store = stack.enter_context(open_index(path, setup))
        report = sync(sources, store, lambda source: chunker.get()(source), embedder, urls=urls, batch=args.batch,
                      dry_run=args.dry_run, allow_mass_delete=args.allow_mass_delete, progress=progress)
    _say(report.render())
    if chunker.made and hasattr(chunker.get(), "verify"):
        try:
            chunker.get().verify()
        except RuntimeError as exc:
            print(f"Warning: {exc}\nThe index was written, but do not trust it until this is understood.", file=sys.stderr)
            return 1
    return 0


def _status(args, services: Services) -> int:
    path = Path(args.index)
    if not path.exists():
        print(f"{path} does not exist; run `index.ps1 sync` first.", file=sys.stderr)
        return 1
    with open_index(path, services.setup(args), read_only=True) as store:
        meta, counts, sources = store.meta(), store.counts(), store.sources()
        _say(f"Index: {path} ({path.stat().st_size / 1_000_000:.1f} MB)")
        _say(f"  {counts['sources']} sources, {counts['passages']} passages, {counts['vectors']} vectors")
        _say(f"  last sync: {max((s.loaded_at for s in sources.values()), default='never')}")
        _say(f"  schema {meta.get('schema_version')}, created {meta.get('created_at')}, vector size {meta.get('dim', '-')}")
        _say("  built with:")
        for key in ("embedder_id", "embedder_revision", "chunker", "chunker_version", "chunk_config",
                    "llama_index_version", "reranker_id", "reranker_revision"):
            _say(f"    {key}: {meta.get(key)}")
    return 0


def _search(args, services: Services) -> int:
    if not args.question:
        print("Give the question to search for: index.ps1 search \"your question\"", file=sys.stderr)
        return 2
    path = Path(args.index)
    if not path.exists():
        print(f"{path} does not exist; run `index.ps1 sync` first.", file=sys.stderr)
        return 1
    with Retriever.open(path, services.setup(args), services.make_embedder(args), services.make_reranker(args)) as retriever:
        results = retriever.search(args.question, k=args.k, max_per_file=args.max_per_file)
    if not results:
        _say("No passages found.")
    for r in results:
        snippet = " ".join(r.text.split())[:SNIPPET_CHARS]
        _say(f"{r.rank}. [{r.score:.2f}] {' > '.join(r.heading_path)}")
        _say(f"   {r.url or r.source_id}  ({r.source_id}, {r.licence})")
        _say(f"   {snippet}")
    return 0


def _check(args, services: Services) -> int:
    """Spec 17 test 10: score the frozen passage-level questions through `Retriever.search`; the counts must equal the
    stage 4 result, and a second run must give exactly the same answers."""
    if not (args.register and args.snapshots):
        print("Give --register and --snapshots.", file=sys.stderr)
        return 2
    questions, spans = services.load_questions(args)
    if not questions:
        print("Cannot start: no questions found in " + str(args.passage_questions), file=sys.stderr)
        return 2
    path = Path(args.index)
    if not path.exists():
        print(f"{path} does not exist; run `index.ps1 sync` first.", file=sys.stderr)
        return 1
    with Retriever.open(path, services.setup(args), services.make_embedder(args), services.make_reranker(args)) as retriever:
        def retrieve(question: str, n: int) -> list[PassageHit]:
            return [PassageHit(r.source_id, r.start, r.end, r.score) for r in retriever.search(question, k=n)]

        first = score_passage_questions(questions, spans, retrieve)
        second = score_passage_questions(questions, spans, retrieve)
    total = len(questions)
    right_first, right_top3 = sum(r.first for r in first), sum(r.top_k for r in first)
    repeatable = first == second
    matches = (right_first, right_top3) == (args.expect_first, args.expect_top3)
    _say(f"Checked {total} questions through Retriever.search")
    _say(f"  right first: {right_first} of {total}")
    _say(f"  right in the first 3: {right_top3} of {total}")
    _say(f"  right within 10: {sum(r.recall10 for r in first)} of {total}")
    _say(f"  mean reciprocal rank: {sum(r.rr for r in first) / total:.3f}")
    _say(f"  same result on a second run: {'yes' if repeatable else 'no'}")
    if matches:
        _say(f"  the stage 4 result ({args.expect_first} first, {args.expect_top3} in the first 3) matches")
    else:
        _say(f"  differs from the stage 4 result: expected {args.expect_first} first and {args.expect_top3} in the "
             f"first 3")
    return 0 if matches and repeatable else 1


def main(argv: list[str] | None = None, services: Services | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
    except SystemExit as exc:
        return int(exc.code or 0)
    if args.k < 1 or (args.max_per_file is not None and args.max_per_file < 1) or args.batch < 1:
        print("-k, --max-per-file and --batch must be 1 or more.", file=sys.stderr)
        return 2
    try:
        services = services or real_services()
        return {"sync": _sync, "status": _status, "search": _search, "check": _check}[args.command](args, services)
    except (SyncError, IndexStoreError) as exc:
        print(f"Cannot continue: {_plain(exc)}", file=sys.stderr)
        return 1
    except (ValueError, RegisterError, OSError, KeyError) as exc:
        print(f"Cannot start: {exc}", file=sys.stderr)
        return 2
