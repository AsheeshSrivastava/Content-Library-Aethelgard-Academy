"""Make the reference scores the reranker wrapper is checked against (spec 15 section 3.3). Run once in a throwaway
environment so the reference library never touches the project:

    uv run --no-project --with "sentence-transformers>=3,<4" --with "transformers>=4.50,<5" --with pyyaml \
        python scripts/golden_rerank.py

The reference is sentence-transformers `CrossEncoder` with an identity activation (raw logits).
Output: ~/.ragcore/golden/bge-reranker-v2-m3.json with the pairs and the reference scores."""
import json
from pathlib import Path

import torch
import yaml

PAIRS = [
    ("how do I retry a failed call", "Use a decorator that runs the function again when it raises a timeout error."),
    ("how do I retry a failed call", "Chunking splits long documents into passages before they are embedded."),
    ("what does a reranker do", "Reranking reorders the first results with a slower, more careful model."),
    ("what does a reranker do", "# Installing\n\nRun `uv sync` and then the tests."),
    ("how does a client find tools", "The MCP specification defines how a client discovers the tools a server offers."),
]
ID = "bge-reranker-v2-m3"


def main() -> None:
    spec = yaml.safe_load((Path(__file__).resolve().parent.parent / "configs" / "rerankers.yaml").read_text(encoding="utf-8"))[ID]
    from sentence_transformers import CrossEncoder
    model = CrossEncoder(spec["repo"], revision=spec["revision"], device="cpu", max_length=spec["max_pair_tokens"],
                         default_activation_function=torch.nn.Identity())
    scores = [float(s) for s in model.predict(PAIRS)]
    out_dir = Path.home() / ".ragcore" / "golden"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{ID}.json").write_text(json.dumps(
        {"model": ID, "revision": spec["revision"], "reference": "sentence-transformers CrossEncoder, identity activation",
         "pairs": PAIRS, "scores": scores}), encoding="utf-8")
    print(ID, "->", out_dir / f"{ID}.json", scores)


if __name__ == "__main__":
    main()
