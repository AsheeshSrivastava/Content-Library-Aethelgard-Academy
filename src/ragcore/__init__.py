"""rag-core: citation-grounded retrieval. The public surface is the names in `__all__` (spec 18 section 3);
everything else in this package is internal and may change."""
from importlib import metadata as _metadata

from ragcore.index_retriever import Result, Retriever
from ragcore.index_store import IndexSetup, IndexStoreError, open_index

try:
    __version__ = _metadata.version("ragcore")
except _metadata.PackageNotFoundError:  # a source checkout that was never installed
    __version__ = "0.1.0"

__all__ = ["Retriever", "Result", "IndexSetup", "open_index", "IndexStoreError", "__version__"]
