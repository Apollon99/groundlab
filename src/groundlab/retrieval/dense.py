"""Dense retrieval with a pinned bi-encoder (BAAI/bge-small-en-v1.5).

Every setting is explicit (D-019a, D-020, D-021, D-022), never a library default:
- documents: canonical text `title + "\\n" + text`, no instruction;
- queries: the fixed instruction from the config prepended (dense queries only);
- tokenizer from the pinned revision, right truncation at 512 tokens, padding to the
  longest sequence in the batch;
- encoder in FP32 with TF32 disabled, eager attention, CLS pooling, L2 normalization;
- embeddings run on CUDA only; there is no CPU fallback (`device="cuda"` is required
  by the config and checked before loading);
- search is an exact dot product on CPU in float64 over all documents, ranked by score
  descending then doc ID ascending.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from groundlab.errors import CudaUnavailableError, VerificationError
from groundlab.io_utils import loads_strict, sha256_file, write_json
from groundlab.models import ModelSpec, model_dir, verify_model
from groundlab.text import DOCUMENT_SEPARATOR, canonical_document_text
from groundlab.types import Document, Query, Run, ScoredDoc

EMBEDDINGS_FILE = "embeddings.npy"
DOC_IDS_FILE = "doc_ids.json"
NORM_TOLERANCE = 1e-3


@dataclass(frozen=True)
class DenseParams:
    model: str
    revision: str
    pooling: str
    normalize: bool
    max_length: int
    truncation_side: str
    query_instruction: str
    document_instruction: str
    document_separator: str
    attn_implementation: str
    dtype: str
    device: str
    batch_size: int

    def __post_init__(self) -> None:
        checks = {
            "pooling": (self.pooling, "cls"),
            "normalize": (self.normalize, True),
            "truncation_side": (self.truncation_side, "right"),
            "document_instruction": (self.document_instruction, ""),
            "document_separator": (self.document_separator, DOCUMENT_SEPARATOR),
            "attn_implementation": (self.attn_implementation, "eager"),
            "dtype": (self.dtype, "float32"),
            "device": (self.device, "cuda"),
        }
        for name, (value, required) in checks.items():
            if value != required:
                raise ValueError(f"dense.{name} must be {required!r}, got {value!r}")
        if not 1 <= self.max_length <= 512:
            raise ValueError("dense.max_length must be in 1..512")
        if self.batch_size <= 0:
            raise ValueError("dense.batch_size must be positive")


def configure_torch_numerics() -> dict[str, Any]:
    """Force true FP32 matmuls (no TF32) and return the settings for the manifest."""
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    return {
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }


class Encoder:
    """Bi-encoder over a verified local model directory.

    `device` is passed explicitly. The CLI only ever passes "cuda"; tests may pass "cpu"
    for a tiny synthetic model. Nothing falls back silently.
    """

    def __init__(self, spec: ModelSpec, models_dir: Path, params: DenseParams, device: str) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise CudaUnavailableError("dense encoding requires CUDA, which is unavailable")
        if spec.revision != params.revision:
            raise VerificationError(
                f"config revision {params.revision} != pinned {spec.revision} for {spec.key}"
            )
        self.model_file_sha256 = verify_model(spec, models_dir)
        self.spec = spec
        self.params = params
        self.device = device
        self.numerics = configure_torch_numerics()
        root = model_dir(spec, models_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(root)
        self.tokenizer.truncation_side = "right"
        self.tokenizer.padding_side = "right"
        self.model = AutoModel.from_pretrained(
            root, dtype=torch.float32, attn_implementation=params.attn_implementation
        )
        self.model.eval().to(device)
        self.dim = int(self.model.config.hidden_size)

    def token_lengths(self, texts: Sequence[str]) -> list[int]:
        """Untruncated token counts including special tokens."""
        enc = self.tokenizer(list(texts), add_special_tokens=True, truncation=False, verbose=False)
        return [len(ids) for ids in enc["input_ids"]]

    def encode(self, texts: Sequence[str], batch_size: int | None = None) -> np.ndarray:
        import torch

        bs = batch_size or self.params.batch_size
        out = np.empty((len(texts), self.dim), dtype=np.float32)
        with torch.inference_mode():
            for start in range(0, len(texts), bs):
                batch = list(texts[start : start + bs])
                enc = self.tokenizer(
                    batch,
                    padding="longest",
                    truncation=True,
                    max_length=self.params.max_length,
                    return_tensors="pt",
                ).to(self.device)
                hidden = self.model(**enc).last_hidden_state
                cls = hidden[:, 0]  # CLS pooling
                cls = torch.nn.functional.normalize(cls, p=2, dim=-1)
                out[start : start + len(batch)] = cls.float().cpu().numpy()
        if not np.all(np.isfinite(out)):
            raise VerificationError("non-finite embedding produced")
        return out

    def query_texts(self, queries: Sequence[Query]) -> list[str]:
        return [self.params.query_instruction + q.text for q in queries]

    def encode_queries(self, queries: Sequence[Query]) -> np.ndarray:
        return self.encode(self.query_texts(queries))

    def encode_documents(self, docs: Sequence[Document]) -> np.ndarray:
        return self.encode([canonical_document_text(d) for d in docs])

    def describe(self) -> dict[str, Any]:
        import tokenizers
        import torch
        import transformers

        info: dict[str, Any] = {
            "model": self.spec.key,
            "repo_id": self.spec.repo_id,
            "revision": self.spec.revision,
            "model_file_sha256": self.model_file_sha256,
            "dim": self.dim,
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


# --------------------------------------------------------------------------- GPU memory


class GpuMemoryProbe:
    """Peak CUDA memory (allocated/reserved by PyTorch) over a block of work."""

    def __init__(self, device: str) -> None:
        self.cuda = device.startswith("cuda")
        self.device = device

    def __enter__(self) -> GpuMemoryProbe:
        if self.cuda:
            import torch

            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        return self

    def __exit__(self, *exc: object) -> None:
        pass

    def result(self) -> dict[str, int] | None:
        if not self.cuda:
            return None
        import torch

        torch.cuda.synchronize(self.device)
        return {
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(self.device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(self.device)),
        }


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        import torch

        torch.cuda.synchronize(device)


def sync_device(device: str) -> None:
    """Public wrapper used by the hybrid timing code."""
    _sync(device)


# --------------------------------------------------------------------------- index


class DenseIndex:
    """Verified embedding cache + exact dot-product search."""

    def __init__(self, embeddings: np.ndarray, doc_ids: list[str], manifest: dict[str, Any]):
        if embeddings.dtype != np.float32 or embeddings.ndim != 2:
            raise VerificationError("embeddings must be a 2-D float32 array")
        if embeddings.shape[0] != len(doc_ids):
            raise VerificationError("embeddings and doc_ids disagree in length")
        self.embeddings = embeddings
        self._emb64 = embeddings.astype(np.float64)
        self.doc_ids = doc_ids
        self.manifest = manifest
        order = sorted(range(len(doc_ids)), key=lambda i: doc_ids[i])
        self._id_rank = np.empty(len(doc_ids), dtype=np.int64)
        self._id_rank[order] = np.arange(len(doc_ids))

    @classmethod
    def build(
        cls,
        encoder: Encoder,
        docs: Sequence[Document],
        corpus_sha256: str,
    ) -> DenseIndex:
        if not docs:
            raise ValueError("cannot index an empty corpus")
        texts = [canonical_document_text(d) for d in docs]
        lengths = encoder.token_lengths(texts)
        with GpuMemoryProbe(encoder.device) as probe:
            _sync(encoder.device)
            start = time.perf_counter()
            embeddings = encoder.encode(texts)
            _sync(encoder.device)
            seconds = time.perf_counter() - start
        manifest = {
            "kind": "dense",
            "params": asdict(encoder.params),
            "encoder": encoder.describe(),
            "corpus_sha256": corpus_sha256,
            "n_docs": len(docs),
            "dim": encoder.dim,
            "document_tokens": {
                "max_untruncated": max(lengths),
                "mean_untruncated": float(np.mean(lengths)),
                "n_truncated": sum(1 for n in lengths if n > encoder.params.max_length),
            },
            "encode_seconds": seconds,
            "gpu_memory": probe.result(),
        }
        return cls(embeddings, [d.doc_id for d in docs], manifest)

    def save(self, index_dir: Path) -> None:
        index_dir.mkdir(parents=True, exist_ok=True)
        (index_dir / "manifest.json").unlink(missing_ok=True)
        with (index_dir / EMBEDDINGS_FILE).open("wb") as handle:
            np.save(handle, self.embeddings, allow_pickle=False)
        write_json(index_dir / DOC_IDS_FILE, self.doc_ids)
        self.manifest["file_sha256"] = {
            name: sha256_file(index_dir / name) for name in (EMBEDDINGS_FILE, DOC_IDS_FILE)
        }
        write_json(index_dir / "manifest.json", self.manifest)

    @classmethod
    def load(
        cls,
        index_dir: Path,
        params: DenseParams,
        corpus_sha256: str,
        model_file_sha256: dict[str, str],
    ) -> DenseIndex:
        path = index_dir / "manifest.json"
        if not path.exists():
            raise VerificationError(f"no embedding cache at {index_dir}; run `groundlab build`")
        manifest = loads_strict(path.read_text(encoding="utf-8"))
        if manifest.get("kind") != "dense":
            raise VerificationError(f"{index_dir} is not a dense embedding cache")
        if manifest["params"] != asdict(params):
            raise VerificationError("embedding cache was built with different dense params")
        if manifest["corpus_sha256"] != corpus_sha256:
            raise VerificationError("embedding cache was built from a different corpus")
        if manifest["encoder"]["model_file_sha256"] != model_file_sha256:
            raise VerificationError("embedding cache was built with different model files")
        recorded = manifest.get("file_sha256")
        expected_names = {EMBEDDINGS_FILE, DOC_IDS_FILE}
        if not isinstance(recorded, dict) or set(recorded) != expected_names:
            raise VerificationError("embedding cache manifest lacks file hashes; rebuild")
        present = {p.name for p in index_dir.iterdir() if p.is_file() and p.name != "manifest.json"}
        if present != expected_names:
            raise VerificationError(f"unexpected files in embedding cache: {sorted(present)}")
        for name in sorted(expected_names):
            if sha256_file(index_dir / name) != recorded[name]:
                raise VerificationError(f"embedding cache file {name} differs from manifest")
        embeddings = np.load(index_dir / EMBEDDINGS_FILE, allow_pickle=False)
        doc_ids = loads_strict((index_dir / DOC_IDS_FILE).read_text(encoding="utf-8"))
        if embeddings.shape != (manifest["n_docs"], manifest["dim"]):
            raise VerificationError("embedding matrix shape does not match the manifest")
        norms = np.linalg.norm(embeddings.astype(np.float64), axis=1)
        if not np.all(np.isfinite(embeddings)) or np.max(np.abs(norms - 1.0)) > NORM_TOLERANCE:
            raise VerificationError("embeddings are not finite unit vectors")
        return cls(embeddings, list(doc_ids), manifest)

    def scores(self, query_embedding: np.ndarray) -> np.ndarray:
        return np.asarray(self._emb64 @ query_embedding.astype(np.float64), dtype=np.float64)

    def rank(self, scores: np.ndarray, depth: int) -> list[ScoredDoc]:
        if not np.all(np.isfinite(scores)):
            raise VerificationError("non-finite dense score")
        order = np.lexsort((self._id_rank, -scores))[:depth]
        return [ScoredDoc(self.doc_ids[i], float(scores[i])) for i in order]

    def search_embeddings(
        self, queries: Sequence[Query], query_embeddings: np.ndarray, depth: int
    ) -> Run:
        if depth <= 0:
            raise ValueError("depth must be positive")
        if query_embeddings.shape != (len(queries), self.embeddings.shape[1]):
            raise VerificationError("query embedding shape mismatch")
        all_scores = query_embeddings.astype(np.float64) @ self._emb64.T
        return {q.query_id: self.rank(all_scores[i], depth) for i, q in enumerate(queries)}


# --------------------------------------------------------------------------- latency


def _percentile(values: Sequence[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q, method="linear"))


def measure_latency(
    encoder: Encoder,
    index: DenseIndex,
    queries: Sequence[Query],
    depth: int,
    warmup: int,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Single-query end-to-end latency: tokenize + encode on GPU + exact search + rank.

    The first `warmup` queries are run once untimed. Then every query is timed once,
    one at a time, with CUDA synchronized before the clock is read.
    """
    texts = encoder.query_texts(queries)
    for text in texts[:warmup]:
        index.rank(index.scores(encoder.encode([text], batch_size=1)[0]), depth)
    _sync(encoder.device)
    timings_ms: list[float] = []
    single: list[np.ndarray] = []
    rankings: list[list[str]] = []
    for text in texts:
        start = clock()
        emb = encoder.encode([text], batch_size=1)[0]
        ranked = index.rank(index.scores(emb), depth)
        _sync(encoder.device)
        timings_ms.append(1000.0 * (clock() - start))
        # Diagnostics are collected after the clock is read, outside the timed region.
        single.append(emb)
        rankings.append([h.doc_id for h in ranked])
    return {
        "protocol": "single query, end to end (tokenize, encode, exact search, rank), "
        f"{warmup} untimed warm-up queries, each query timed once, CUDA synchronized",
        "n": len(timings_ms),
        "warmup": warmup,
        "mean_ms": float(np.mean(timings_ms)),
        "p50_ms": _percentile(timings_ms, 50),
        "p90_ms": _percentile(timings_ms, 90),
        "p99_ms": _percentile(timings_ms, 99),
        "min_ms": float(np.min(timings_ms)),
        "max_ms": float(np.max(timings_ms)),
        "single_query_embeddings": np.vstack(single),
        "single_query_rankings": rankings,
    }
