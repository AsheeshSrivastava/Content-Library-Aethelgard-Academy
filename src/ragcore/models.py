"""The pinned models, built from pins that travel inside the package (spec 19).

`load_models` is the public way for an installed `ragcore` to get the embedder, the reranker and the matching
`IndexSetup`. Importing this module needs no torch; the models load inside `load_models`."""
from __future__ import annotations

from pathlib import Path

from ragcore.chunking import ChunkConfig
from ragcore.embed.base import ModelSpec, load_model_specs
from ragcore.index_store import IndexSetup
from ragcore.llamaindex_chunker import llamaindex_version
from ragcore.rerank import RerankerSpec, load_reranker_specs

PINS_DIR = Path(__file__).resolve().parent / "pins"
MODELS_PINS = PINS_DIR / "models.yaml"
RERANKER_PINS = PINS_DIR / "rerankers.yaml"
# The index file records the LlamaIndex version it was built with. A caller that only reads an index does not install
# LlamaIndex, so the pin is written here; a test keeps it equal to the `variants` pin in pyproject.toml.
LLAMA_INDEX_PIN = "0.14.25"
DEFAULT_EMBEDDER = "jina-v2-base-en"
DEFAULT_RERANKER = "bge-reranker-v2-m3"


def make_setup(embedder: ModelSpec, reranker: RerankerSpec, llama_index_version: str,
               config: ChunkConfig | None = None) -> IndexSetup:
    """The one place that turns pins into an `IndexSetup`, used by the index command and by `load_models`."""
    config = config or ChunkConfig()
    revision = embedder.revision + (f"+{embedder.code_revision}" if embedder.code_revision else "")
    return IndexSetup(embedder.id, revision, "llamaindex", llamaindex_version(config),
                      f"{config.max_chars}/{config.hard_limit}", llama_index_version, reranker.id, reranker.revision)


def _pick(specs: dict, wanted: str, kind: str):
    if wanted not in specs:
        raise ValueError(f"unknown {kind} {wanted!r}; choose one of {', '.join(sorted(specs))}")
    return specs[wanted]


def load_models(embedder_id: str = DEFAULT_EMBEDDER, reranker_id: str = DEFAULT_RERANKER, *,
                local_files_only: bool = False):
    """Return `(embedder, reranker, setup)` for the pinned models; pass them to `Retriever.open`.

    Raises ValueError for an id that is not pinned, ImportError when `ragcore[embed]` is not installed, and the
    RuntimeError from the reviewed-code check when the Jina code folder is missing or changed."""
    embedder_spec = _pick(load_model_specs(MODELS_PINS), embedder_id, "embedder")
    reranker_spec = _pick(load_reranker_specs(RERANKER_PINS), reranker_id, "reranker")
    try:
        from ragcore.embed.hf import HFEncoder
        from ragcore.rerank import HFReranker
        embedder = HFEncoder(embedder_spec, local_files_only=local_files_only)
        reranker = HFReranker(reranker_spec, local_files_only=local_files_only)
    except ModuleNotFoundError as exc:
        if (exc.name or "").split(".")[0] not in ("torch", "transformers"):
            raise  # Some other missing module (for example one the Jina code needs) is reported as it is.
        raise ImportError('the models need torch and transformers: install them with `pip install "ragcore[embed]"`') from exc
    return embedder, reranker, make_setup(embedder_spec, reranker_spec, LLAMA_INDEX_PIN)
