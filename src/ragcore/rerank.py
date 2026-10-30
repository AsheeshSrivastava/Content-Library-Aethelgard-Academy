"""The reranker: pinned model spec, ordering rule, a fake for tests, and the real cross-encoder (spec 15 section 3.3).

Nothing outside `HFReranker` needs a GPU; that class needs the `embed` dependency group."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import yaml

_SHA1 = re.compile(r"[0-9a-f]{40}")
_REQUIRED = ("repo", "revision", "license", "max_pair_tokens", "trust_remote_code")
_ALLOWED = set(_REQUIRED) | {"weights"}


@dataclass(frozen=True)
class RerankerSpec:
    id: str
    repo: str
    revision: str
    license: str
    max_pair_tokens: int
    weights: str | None = None


def _spec(reranker_id: str, raw: object) -> RerankerSpec:
    def bad(why: str) -> ValueError:
        return ValueError(f"{reranker_id}: {why}")

    if not isinstance(raw, dict):
        raise bad("must be a mapping")
    for key in raw:
        if key not in _ALLOWED:
            raise bad(f"unknown key {key!r}")
    for key in _REQUIRED:
        if key not in raw:
            raise bad(f"missing {key}")
    for key in ("repo", "license"):
        if not isinstance(raw[key], str) or not raw[key].strip():
            raise bad(f"{key} must be text")
    if not isinstance(raw["revision"], str) or not _SHA1.fullmatch(raw["revision"]):
        raise bad("revision must be a full 40-character commit hash, not a branch name")
    tokens = raw["max_pair_tokens"]
    if not isinstance(tokens, int) or isinstance(tokens, bool) or tokens <= 0:
        raise bad("max_pair_tokens must be a positive whole number")
    if raw["trust_remote_code"] is not False:
        raise bad("a reranker must not need remote code (trust_remote_code: false)")
    return RerankerSpec(reranker_id, raw["repo"], raw["revision"], raw["license"], tokens, raw.get("weights"))


def load_reranker_specs(path: Path) -> dict[str, RerankerSpec]:
    try:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"{Path(path).name}: not valid YAML ({exc})") from None
    if not isinstance(data, dict) or not data:
        raise ValueError(f"{Path(path).name}: needs a mapping of reranker ids to pins")
    return {reranker_id: _spec(reranker_id, raw) for reranker_id, raw in data.items()}


class Reranker(Protocol):
    truncated: int  # Pairs whose passage was cut to fit the pair limit, since this reranker was made.

    def score(self, query: str, passages: list[str]) -> list[float]: ...


def rerank(reranker: Reranker, query: str, passages: list[str]) -> list[int]:
    """Indices of `passages`, best first. Equal scores keep the order they came in."""
    if not passages:
        return []
    scores = reranker.score(query, passages)
    if len(scores) != len(passages):
        raise ValueError(f"reranker gave {len(scores)} scores for {len(passages)} passages")
    return sorted(range(len(passages)), key=lambda i: -scores[i])  # Python's sort is stable, so ties keep input order.


_WORD = re.compile(r"\w+")


class FakeReranker:
    """Stand-in for tests: the score is how many different query words the passage holds."""

    def __init__(self) -> None:
        self.truncated = 0

    def score(self, query: str, passages: list[str]) -> list[float]:
        wanted = {w.lower() for w in _WORD.findall(query)}
        return [float(len(wanted & {w.lower() for w in _WORD.findall(p)})) for p in passages]


class HFReranker:
    """bge-reranker-v2-m3 as a pinned cross-encoder: one logit per (query, passage) pair, fp16 on the GPU."""

    def __init__(self, spec: RerankerSpec, *, device: str | None = None, dtype: str = "float16",
                 batch_pairs: int = 16, local_files_only: bool = False) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.spec, self.id, self.max_pair_tokens, self.batch_pairs = spec, spec.id, spec.max_pair_tokens, batch_pairs
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype if self.device != "cpu" else "float32"
        self.truncated = 0
        self.fingerprint = f"{spec.repo}@{spec.revision[:12]}|{self.dtype}|{spec.max_pair_tokens}"
        self.tokenizer = AutoTokenizer.from_pretrained(spec.repo, revision=spec.revision, local_files_only=local_files_only)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            spec.repo, revision=spec.revision, trust_remote_code=False, torch_dtype=getattr(torch, self.dtype),
            local_files_only=local_files_only).to(self.device).eval()

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        full = self.tokenizer([query] * len(passages), list(passages))["input_ids"]
        self.truncated += sum(1 for ids in full if len(ids) > self.max_pair_tokens)
        order = sorted(range(len(passages)), key=lambda i: -len(full[i]))
        out = [0.0] * len(passages)
        for start in range(0, len(order), self.batch_pairs):
            chosen = order[start:start + self.batch_pairs]
            enc = self.tokenizer([query] * len(chosen), [passages[i] for i in chosen], padding=True,
                                 truncation="only_second", max_length=self.max_pair_tokens, return_tensors="pt")
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with self.torch.inference_mode():
                logits = self.model(**enc).logits.view(-1).float().cpu().tolist()
            for i, value in zip(chosen, logits):
                out[i] = value
        return out
