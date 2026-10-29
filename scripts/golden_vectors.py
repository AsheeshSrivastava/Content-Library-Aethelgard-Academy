"""Make the reference vectors the wrapper is checked against (spec 13 section 4.1). Run once in a throwaway environment
so the reference libraries never touch the project:

    uv run --no-project --with "sentence-transformers>=3,<4" --with "transformers>=4.50,<5" --with pyyaml \
        python scripts/golden_vectors.py bge-m3 jina-v2-base-en

bge-m3 is encoded by sentence-transformers (which applies the pooling set in the model's own 1_Pooling/config.json).
jina-v2-base-en is encoded by the model's own `encode` method (its README's reference usage).
Output: ~/.ragcore/golden/<id>.json with the texts and the unit-length reference vectors (float32 as lists)."""
import json
import sys
from pathlib import Path

import yaml

TEXTS = [
    "How do I run a function again if it fails with a timeout?",
    "Chunking splits long documents into passages before they are embedded.",
    "def add(a, b):\n    return a + b",
    "The MCP specification defines how a client discovers the tools a server offers.",
    "Reranking reorders the first results with a slower, more careful model.",
    "# Installing\n\nRun `uv sync` and then the tests.",
]


def main(ids: list[str]) -> None:
    specs = yaml.safe_load((Path(__file__).resolve().parent.parent / "configs" / "models.yaml").read_text(encoding="utf-8"))
    out_dir = Path.home() / ".ragcore" / "golden"
    out_dir.mkdir(parents=True, exist_ok=True)
    for model_id in ids:
        spec = specs[model_id]
        if model_id == "bge-m3":
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(spec["repo"], revision=spec["revision"], device="cpu")
            vectors = model.encode(TEXTS, normalize_embeddings=True)
            how = "sentence-transformers " + __import__("sentence_transformers").__version__
        else:
            from transformers import AutoModel
            model = AutoModel.from_pretrained(spec["repo"], revision=spec["revision"], trust_remote_code=True,
                                              code_revision=spec["code"]["revision"]).eval()
            vectors = model.encode(TEXTS, normalize_embeddings=True)
            how = "the model's own encode()"
        vectors = [list(map(float, v)) for v in vectors]
        (out_dir / f"{model_id}.json").write_text(json.dumps(
            {"model": model_id, "revision": spec["revision"], "reference": how, "texts": TEXTS, "vectors": vectors}),
            encoding="utf-8")
        print(model_id, "->", out_dir / f"{model_id}.json", "via", how, "dim", len(vectors[0]))


if __name__ == "__main__":
    main(sys.argv[1:] or ["bge-m3", "jina-v2-base-en"])
