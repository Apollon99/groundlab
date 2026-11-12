"""Cross-encoder reranking of the head of the hybrid list (D-027, D-028).

Fixed settings, never library defaults:
- model `cross-encoder/ms-marco-MiniLM-L6-v2` at the pinned revision, loaded from the verified
  local directory with `local_files_only=True` and `use_safetensors=True` (no network, no pickle);
- input: the raw query (no instruction) and the canonical document text as one 512-token pair,
  only the document truncated (`encode_cross_encoder_pair`, D-019a);
- score: the raw finite logit (`num_labels == 1`), no sigmoid;
- numerics: FP32, TF32 off, eager attention, `eval()` and `inference_mode`, CUDA only;
- batching: the pairs of ONE query are scored together, in fused order, in chunks of
  `batch_size`, each chunk right-padded to its longest pair. Pairs of different queries are
  never mixed. The two paths share the batching policy and use identical batches when
  their fused candidate lists match.
- only the first `min(rerank_depth, len(fused))` candidates are scored. They are reordered by
  logit descending, then doc ID ascending (string order). The tail keeps its order and
  membership exactly.
- run-file scores of the final list are rank-derived and strictly descending
  (`len(list) - position`); the head's logits and the tail's RRF scores live on different scales
  and are not mixed in one score column. Evaluation reads the rank column.

Nothing here receives qrels.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from groundlab.errors import CudaUnavailableError, VerificationError
from groundlab.models import ModelSpec, model_dir, verify_model
from groundlab.retrieval.dense import configure_torch_numerics
from groundlab.text import encode_cross_encoder_pair
from groundlab.types import Query, ScoredDoc

RERANK_DEPTH = 30
WEIGHTS_FILE = "model.safetensors"


@dataclass(frozen=True)
class RerankParams:
    model: str
    revision: str
    max_length: int
    rerank_depth: int
    batch_size: int
    attn_implementation: str
    dtype: str
    device: str

    def __post_init__(self) -> None:
        checks = {
            "attn_implementation": (self.attn_implementation, "eager"),
            "dtype": (self.dtype, "float32"),
            "device": (self.device, "cuda"),
        }
        for name, (value, required) in checks.items():
            if value != required:
                raise ValueError(f"rerank.{name} must be {required!r}, got {value!r}")
        if not 1 <= self.max_length <= 512:
            raise ValueError("rerank.max_length must be in 1..512")
        if self.rerank_depth <= 0 or self.batch_size <= 0:
            raise ValueError("rerank.rerank_depth and rerank.batch_size must be positive")


# --------------------------------------------------------------------------- pure ordering


def rerank_candidates(
    fused: Sequence[ScoredDoc],
    score_head: Callable[[list[str]], Sequence[float]],
    depth: int,
) -> list[ScoredDoc]:
    """Reorder only `fused[:depth]` by the scores `score_head` returns for those doc IDs.

    `score_head` is called once with the head's doc IDs in fused order, and not at all for an
    empty list. Scores must be finite, one per candidate. Ties (exactly equal scores) are
    broken by doc ID ascending. The returned scores are strictly descending rank scores.
    """
    if depth <= 0:
        raise ValueError(f"depth must be positive, got {depth}")
    n = min(depth, len(fused))
    if n == 0:
        return []
    head_ids = [hit.doc_id for hit in fused[:n]]
    logits = [float(x) for x in score_head(list(head_ids))]
    if len(logits) != n:
        raise VerificationError(f"reranker returned {len(logits)} scores for {n} candidates")
    if not all(math.isfinite(x) for x in logits):
        raise VerificationError("non-finite reranker score")
    order = sorted(range(n), key=lambda i: (-logits[i], head_ids[i]))
    ids = [head_ids[i] for i in order] + [hit.doc_id for hit in fused[n:]]
    return [ScoredDoc(doc_id, float(len(ids) - pos)) for pos, doc_id in enumerate(ids)]


def rerank_query(
    cross_encoder: CrossEncoder,
    query: Query,
    fused: Sequence[ScoredDoc],
    doc_texts: Mapping[str, str],
    depth: int,
) -> list[ScoredDoc]:
    """Rerank one query's fused list with the real model. Only the query text is used."""

    def score_head(doc_ids: list[str]) -> list[float]:
        missing = [d for d in doc_ids if d not in doc_texts]
        if missing:
            raise VerificationError(f"candidate documents not in the corpus: {missing[:3]}")
        return cross_encoder.score(query.text, [doc_texts[d] for d in doc_ids])

    return rerank_candidates(fused, score_head, depth)


# --------------------------------------------------------------------------- model


class CrossEncoder:
    """Pinned sequence-classification model over a verified local directory."""

    def __init__(
        self, spec: ModelSpec, models_dir: Path, params: RerankParams, device: str
    ) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise CudaUnavailableError("reranking requires CUDA, which is unavailable")
        if spec.revision != params.revision:
            raise VerificationError(
                f"config revision {params.revision} != pinned {spec.revision} for {spec.key}"
            )
        if WEIGHTS_FILE not in spec.files:
            raise VerificationError(f"{spec.key}: {WEIGHTS_FILE} is not pinned")
        self.model_file_sha256 = verify_model(spec, models_dir)
        self.spec = spec
        self.params = params
        self.device = device
        self.numerics = configure_torch_numerics()
        root = model_dir(spec, models_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            root,
            dtype=torch.float32,
            attn_implementation=params.attn_implementation,
            use_safetensors=True,
            local_files_only=True,
        )
        if int(self.model.config.num_labels) != 1:
            raise VerificationError(f"{spec.key}: expected one logit, got num_labels != 1")
        self.model.eval().to(device)
        self._pairs = 0
        self._truncated = 0
        self._max_tokens = 0

    def take_stats(self) -> dict[str, int]:
        """Pair counters since the last call (they are reset by this call)."""
        stats = {
            "pairs_scored": self._pairs,
            "pairs_document_truncated": self._truncated,
            "max_pair_tokens": self._max_tokens,
        }
        self._pairs = self._truncated = self._max_tokens = 0
        return stats

    def score(self, query: str, doc_texts: Sequence[str]) -> list[float]:
        """Raw logits, one per document, in input order."""
        import torch

        pairs = [
            encode_cross_encoder_pair(self.tokenizer, query, text, self.params.max_length)
            for text in doc_texts
        ]
        self._pairs += len(pairs)
        self._truncated += sum(1 for p in pairs if p.document_truncated)
        self._max_tokens = max([self._max_tokens, *(len(p.input_ids) for p in pairs)])
        pad_id = int(self.tokenizer.pad_token_id)
        out: list[float] = []
        bs = self.params.batch_size
        with torch.inference_mode():
            for start in range(0, len(pairs), bs):
                chunk = pairs[start : start + bs]
                width = max(len(p.input_ids) for p in chunk)
                ids = torch.full((len(chunk), width), pad_id, dtype=torch.long)
                types = torch.zeros((len(chunk), width), dtype=torch.long)
                mask = torch.zeros((len(chunk), width), dtype=torch.long)
                for row, p in enumerate(chunk):
                    n = len(p.input_ids)
                    ids[row, :n] = torch.tensor(p.input_ids, dtype=torch.long)
                    types[row, :n] = torch.tensor(p.token_type_ids, dtype=torch.long)
                    mask[row, :n] = torch.tensor(p.attention_mask, dtype=torch.long)
                logits = self.model(
                    input_ids=ids.to(self.device),
                    token_type_ids=types.to(self.device),
                    attention_mask=mask.to(self.device),
                ).logits
                if tuple(logits.shape) != (len(chunk), 1):
                    raise VerificationError(f"unexpected logits shape {tuple(logits.shape)}")
                out.extend(float(x) for x in logits[:, 0].float().cpu().tolist())
        if not all(math.isfinite(x) for x in out):
            raise VerificationError("non-finite reranker logit produced")
        return out

    def describe(self) -> dict[str, Any]:
        import tokenizers
        import torch
        import transformers

        info: dict[str, Any] = {
            "model": self.spec.key,
            "repo_id": self.spec.repo_id,
            "revision": self.spec.revision,
            "model_file_sha256": self.model_file_sha256,
            "params": asdict(self.params),
            "loading": {"local_files_only": True, "use_safetensors": True},
            "score": "raw logit (num_labels == 1), no activation",
            "pair_input": "raw query + canonical document text; document-only truncation",
            "batching": "pairs of one query per batch, fused order, right-padded to longest",
            "device": self.device,
            "numerics": self.numerics,
            "versions": {
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "tokenizers": tokenizers.__version__,
            },
        }
        if self.device.startswith("cuda"):
            info["device_name"] = torch.cuda.get_device_name(self.device)
        return info
