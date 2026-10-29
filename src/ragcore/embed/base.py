"""Embedder protocol, pinned model specs, a fake embedder for tests, and the throughput probe (spec 13, unit 13b).

Nothing here needs a GPU; the real encoder is in `ragcore.embed.hf`."""
from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

import numpy as np
import yaml

KINDS = ("passage", "query")
POOLING = ("cls", "mean")
_SHA1 = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-fA-F]{64}")
_REQUIRED = ("repo", "revision", "license", "dim", "max_tokens", "pooling", "trust_remote_code")
_ALLOWED = set(_REQUIRED) | {"weights", "code"}
_CODE_ALLOWED = {"repo", "revision", "sha256", "local_copy", "review"}


@dataclass(frozen=True)
class ModelSpec:
    id: str
    repo: str
    revision: str
    license: str
    dim: int
    max_tokens: int
    pooling: str
    trust_remote_code: bool
    weights: str | None = None
    code_repo: str | None = None
    code_revision: str | None = None
    code_sha256: dict = field(default_factory=dict)
    code_local_copy: str | None = None


def _spec(model_id: str, raw: object) -> ModelSpec:
    def bad(why: str) -> ValueError:
        return ValueError(f"{model_id}: {why}")

    if not isinstance(raw, dict):
        raise bad("must be a mapping")
    for key in raw:
        if key not in _ALLOWED:
            raise bad(f"unknown key {key!r}")
    for key in _REQUIRED:
        if key not in raw:
            raise bad(f"missing {key}")
    if not isinstance(raw["repo"], str) or not raw["repo"].strip():
        raise bad("repo must be text")
    if not isinstance(raw["license"], str) or not raw["license"].strip():
        raise bad("license must be text")
    if not isinstance(raw["revision"], str) or not _SHA1.fullmatch(raw["revision"]):
        raise bad("revision must be a full 40-character commit hash, not a branch name")
    for key in ("dim", "max_tokens"):
        if not isinstance(raw[key], int) or isinstance(raw[key], bool) or raw[key] <= 0:
            raise bad(f"{key} must be a positive whole number")
    if raw["pooling"] not in POOLING:
        raise bad(f"pooling must be one of {', '.join(POOLING)}")
    if not isinstance(raw["trust_remote_code"], bool):
        raise bad("trust_remote_code must be true or false")
    code = raw.get("code")
    if raw["trust_remote_code"] != (code is not None):
        raise bad("a model with trust_remote_code needs a pinned code section, and only such a model has one")
    code_repo = code_revision = code_local = None
    code_sha: dict = {}
    if code is not None:
        if not isinstance(code, dict):
            raise bad("code must be a mapping")
        for key in code:
            if key not in _CODE_ALLOWED:
                raise bad(f"unknown key code.{key}")
        code_repo, code_revision = code.get("repo"), code.get("revision")
        if not isinstance(code_repo, str) or not code_repo.strip():
            raise bad("code.repo must be text")
        if not isinstance(code_revision, str) or not _SHA1.fullmatch(code_revision):
            raise bad("code.revision must be a full 40-character commit hash")
        code_sha = code.get("sha256")
        if (not isinstance(code_sha, dict) or not code_sha
                or not all(isinstance(v, str) and _SHA256.fullmatch(v) for v in code_sha.values())):
            raise bad("code.sha256 must list a SHA-256 for every reviewed code file")
        code_local = code.get("local_copy")
    return ModelSpec(model_id, raw["repo"], raw["revision"], raw["license"], raw["dim"], raw["max_tokens"],
                     raw["pooling"], raw["trust_remote_code"], raw.get("weights"), code_repo, code_revision,
                     dict(code_sha), code_local)


def load_model_specs(path: Path) -> dict[str, ModelSpec]:
    try:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"{Path(path).name}: not valid YAML ({exc})") from None
    if not isinstance(data, dict) or not data:
        raise ValueError(f"{Path(path).name}: needs a mapping of model ids to pins")
    return {model_id: _spec(model_id, raw) for model_id, raw in data.items()}


@dataclass
class TokenStates:
    """Token vectors of one text and where each token sits in it, for late chunking."""
    vectors: np.ndarray  # (n_tokens, dim) float32, special tokens left out
    offsets: list  # (char start, char end) per token
    truncated: bool = False


class Embedder(Protocol):
    id: str
    fingerprint: str
    dim: int

    def embed(self, texts: list[str], *, kind: str) -> np.ndarray: ...
    def token_states(self, text: str) -> TokenStates: ...
    def count_tokens(self, text: str) -> int: ...


def check_kind(kind: str) -> None:
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {', '.join(KINDS)}, got {kind!r}")


def normalise(vectors: np.ndarray) -> np.ndarray:
    """Unit length rows; an all-zero row stays zero."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return (vectors / np.where(norms == 0, 1.0, norms)).astype(np.float32)


_WORD = re.compile(r"\w+")


class FakeEmbedder:
    """Deterministic stand-in for tests: words hash to vectors, each token is mixed with its neighbours (so context
    matters, as in a real model), and a text vector is the normalised mean of its token vectors."""

    id = "fake"

    def __init__(self, dim: int = 32, window: int = 2) -> None:
        self.dim, self.window = dim, window
        self.fingerprint = f"fake|dim={dim}|window={window}"

    def _word(self, word: str) -> np.ndarray:
        seed = int.from_bytes(hashlib.blake2b(word.lower().encode("utf-8"), digest_size=8).digest(), "big")
        return np.random.default_rng(seed).standard_normal(self.dim)

    def token_states(self, text: str) -> TokenStates:
        found = list(_WORD.finditer(text))
        base = np.array([self._word(m.group(0)) for m in found]).reshape(len(found), self.dim)
        vectors = base.copy()
        for i in range(len(found)):
            around = [base[j] for j in range(max(0, i - self.window), min(len(found), i + self.window + 1)) if j != i]
            if around:
                vectors[i] += 0.5 * np.mean(around, axis=0)
        return TokenStates(vectors.astype(np.float32), [(m.start(), m.end()) for m in found])

    def embed(self, texts: list[str], *, kind: str) -> np.ndarray:
        check_kind(kind)
        rows = [self.token_states(t).vectors.mean(axis=0) if _WORD.search(t) else np.zeros(self.dim) for t in texts]
        return normalise(np.array(rows, dtype=np.float32).reshape(len(texts), self.dim))

    def count_tokens(self, text: str) -> int:
        return len(_WORD.findall(text))


def probe(embedder: Embedder, texts: list[str], *, clock: Callable[[], float] = time.perf_counter) -> dict:
    """Time one real embedding pass over `texts`; the result sizes the estimate printed before a stage."""
    tokens = sum(embedder.count_tokens(t) for t in texts)
    if not tokens:
        raise ValueError("nothing to measure: no text with any tokens")
    start = clock()
    embedder.embed(texts, kind="passage")
    seconds = clock() - start
    rate = tokens / seconds if seconds > 0 else float("inf")
    return {"texts": len(texts), "tokens": tokens, "seconds": seconds, "tokens_per_second": rate}


def estimate_seconds(tokens: int, tokens_per_second: float) -> float:
    if tokens < 0 or tokens_per_second <= 0:
        raise ValueError("tokens must be at least 0 and the rate above 0")
    return tokens / tokens_per_second
