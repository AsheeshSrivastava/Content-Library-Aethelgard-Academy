"""The real encoder: a pinned Hugging Face model on the GPU (spec 13, unit 13b). Needs the `embed` dependency group."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from ragcore.embed.base import ModelSpec, TokenStates, check_kind, normalise


def verify_reviewed_code(spec: ModelSpec) -> None:
    """The remote code that will run must be the copy that was read: same commit, same file hashes."""
    if not spec.trust_remote_code:
        return
    folder = Path(spec.code_local_copy or "").expanduser()
    for name, pinned in spec.code_sha256.items():
        path = folder / name
        if not path.is_file():
            raise RuntimeError(f"{spec.id}: reviewed code file {path} is missing; download it at the pinned commit")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual.lower() != pinned.lower():
            raise RuntimeError(f"{spec.id}: {name} differs from the reviewed copy (sha256 {actual}); read it again before use")


class HFEncoder:
    def __init__(self, spec: ModelSpec, *, device: str | None = None, dtype: str = "float16",
                 max_batch_tokens: int = 32768, local_files_only: bool = False) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        verify_reviewed_code(spec)
        self.torch = torch
        self.spec, self.id, self.dim = spec, spec.id, spec.dim
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype if self.device != "cpu" else "float32"
        self.max_batch_tokens = max_batch_tokens
        self.truncated = 0  # Texts cut at max_tokens since this encoder was made.
        code = f"|code@{spec.code_revision[:12]}" if spec.code_revision else ""
        self.fingerprint = f"{spec.repo}@{spec.revision[:12]}{code}|{spec.pooling}|{self.dtype}|{spec.max_tokens}"
        self.tokenizer = AutoTokenizer.from_pretrained(spec.repo, revision=spec.revision,
                                                       local_files_only=local_files_only)
        extra = {"code_revision": spec.code_revision} if spec.trust_remote_code else {}
        self.model = AutoModel.from_pretrained(spec.repo, revision=spec.revision, trust_remote_code=spec.trust_remote_code,
                                               torch_dtype=getattr(torch, self.dtype), local_files_only=local_files_only,
                                               **extra).to(self.device).eval()

    # --- helpers -----------------------------------------------------------------------------

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def _hidden(self, ids: list[list[int]]) -> "torch.Tensor":  # noqa: F821
        torch = self.torch
        width = max(len(x) for x in ids)
        pad = self.tokenizer.pad_token_id or 0
        batch = torch.tensor([x + [pad] * (width - len(x)) for x in ids], device=self.device)
        mask = torch.tensor([[1] * len(x) + [0] * (width - len(x)) for x in ids], device=self.device)
        with torch.inference_mode():
            out = self.model(input_ids=batch, attention_mask=mask)
        return out[0].float(), mask

    def _pool(self, hidden, mask):
        if self.spec.pooling == "cls":
            return hidden[:, 0]
        weights = mask.unsqueeze(-1).float()
        return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)

    # --- the protocol ------------------------------------------------------------------------

    def embed(self, texts: list[str], *, kind: str) -> np.ndarray:
        check_kind(kind)
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        limit = self.spec.max_tokens
        ids = self.tokenizer(list(texts), truncation=True, max_length=limit)["input_ids"]
        for text, row in zip(texts, ids):
            if len(row) >= limit and len(self.tokenizer(text)["input_ids"]) > limit:
                self.truncated += 1
        order = sorted(range(len(texts)), key=lambda i: -len(ids[i]))
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        pos = 0
        while pos < len(order):
            width = len(ids[order[pos]])
            size = max(1, self.max_batch_tokens // width)
            chosen = order[pos:pos + size]
            hidden, mask = self._hidden([ids[i] for i in chosen])
            vectors = self.torch.nn.functional.normalize(self._pool(hidden, mask), dim=-1)
            out[chosen] = vectors.cpu().numpy()
            pos += size
        return normalise(out)

    def token_states(self, text: str) -> TokenStates:
        enc = self.tokenizer(text, return_offsets_mapping=True, return_special_tokens_mask=True)
        if len(enc["input_ids"]) > self.spec.max_tokens:
            raise ValueError(f"{len(enc['input_ids'])} tokens is over the model limit of {self.spec.max_tokens}; "
                             "long texts are cut into windows by the late chunking code")
        hidden, _ = self._hidden([enc["input_ids"]])
        keep = [i for i, special in enumerate(enc["special_tokens_mask"]) if not special]
        vectors = hidden[0][keep].cpu().numpy().astype(np.float32).reshape(len(keep), self.dim)
        return TokenStates(vectors, [tuple(enc["offset_mapping"][i]) for i in keep], truncated=False)
