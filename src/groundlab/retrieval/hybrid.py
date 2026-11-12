"""Hybrid retrieval: reciprocal rank fusion of the BM25 and dense lists (D-018, D-027).

Fusion, fixed and not tuned:
- inputs are the first `parent_depth` (100) hits of each parent, exactly as the parent
  returned them. BM25 may return fewer (or none); nothing is padded and an absent
  contribution is zero. The parents' own retrieval and ranking code is not touched;
- RRF score of a document = sum over the parents that list it of 1 / (k + rank), with
  1-based ranks and k = 60;
- the sum is computed with exact rational arithmetic, so documents whose scores are
  mathematically equal tie exactly (floating point could separate them by one ulp);
- order: score descending, then doc ID ascending (string order); the top `depth` (100) are
  emitted. The stored score is the correctly rounded float of the exact sum.

`PhaseTimer` and `measure_pipeline_latency` give the hybrid arms a throughput pass and a
timed single-query pass with the same per-phase breakdown. Nothing here receives qrels.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import numpy as np

from groundlab.errors import VerificationError
from groundlab.retrieval.bm25 import BM25Index
from groundlab.retrieval.dense import DenseIndex, Encoder, GpuMemoryProbe, sync_device
from groundlab.types import Query, Run, ScoredDoc

RRF_K = 60
PARENT_DEPTH = 100


@dataclass(frozen=True)
class HybridParams:
    fusion: str
    rrf_k: int
    parent_depth: int

    def __post_init__(self) -> None:
        if self.fusion != "rrf":
            raise ValueError(f"hybrid.fusion must be 'rrf', got {self.fusion!r}")
        if self.rrf_k != RRF_K:
            raise ValueError(f"hybrid.rrf_k must be {RRF_K}, got {self.rrf_k}")
        if self.parent_depth != PARENT_DEPTH:
            raise ValueError(f"hybrid.parent_depth must be {PARENT_DEPTH}")


def rrf_fuse(
    bm25_hits: Sequence[ScoredDoc],
    dense_hits: Sequence[ScoredDoc],
    *,
    k: int = RRF_K,
    parent_depth: int = PARENT_DEPTH,
    depth: int = PARENT_DEPTH,
) -> list[ScoredDoc]:
    """Fuse two best-first lists. See the module docstring for the exact semantics."""
    if k < 0 or parent_depth <= 0 or depth <= 0:
        raise ValueError("k must be >= 0; parent_depth and depth must be positive")
    scores: dict[str, Fraction] = {}
    for hits in (bm25_hits, dense_hits):
        seen: set[str] = set()
        for rank, hit in enumerate(hits[:parent_depth], start=1):
            if hit.doc_id in seen:
                raise VerificationError(f"duplicate document {hit.doc_id!r} in a parent list")
            seen.add(hit.doc_id)
            scores[hit.doc_id] = scores.get(hit.doc_id, Fraction(0)) + Fraction(1, k + rank)
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return [ScoredDoc(doc_id, float(score)) for doc_id, score in ordered[:depth]]


# --------------------------------------------------------------------------- timing


def summarize_ms(values: Sequence[float]) -> dict[str, float | int]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": len(values),
        "mean_ms": float(np.mean(arr)),
        "p50_ms": float(np.percentile(arr, 50, method="linear")),
        "p90_ms": float(np.percentile(arr, 90, method="linear")),
        "p99_ms": float(np.percentile(arr, 99, method="linear")),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
    }


class PhaseTimer:
    """Wall time of named phases, with CUDA synchronized at both ends of each phase.

    With `track_memory`, the CUDA peak counters are reset when a phase starts and read as
    soon as it ends, before any later phase can reset them, so every phase keeps its own peak
    (maximum over its invocations).
    """

    def __init__(
        self,
        device: str,
        track_memory: bool = False,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.device = device
        self.track_memory = track_memory
        self.clock = clock
        self.samples_ms: dict[str, list[float]] = {}
        self.peaks: dict[str, dict[str, int]] = {}

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        probe = GpuMemoryProbe(self.device).__enter__() if self.track_memory else None
        sync_device(self.device)
        start = self.clock()
        yield
        sync_device(self.device)
        elapsed = self.clock() - start
        self.samples_ms.setdefault(name, []).append(1000.0 * elapsed)
        peak = probe.result() if probe is not None else None
        if peak is not None:
            seen = self.peaks.setdefault(name, {})
            for key, value in peak.items():
                seen[key] = max(seen.get(key, 0), value)

    def total_seconds(self) -> dict[str, float]:
        return {name: sum(v) / 1000.0 for name, v in self.samples_ms.items()}

    def overall_peak(self) -> dict[str, int] | None:
        if not self.peaks:
            return None
        keys = next(iter(self.peaks.values())).keys()
        return {key: max(p[key] for p in self.peaks.values()) for key in keys}


def measure_pipeline_latency(
    step: Callable[[Query, PhaseTimer], list[ScoredDoc]],
    queries: Sequence[Query],
    warmup: int,
    device: str,
    clock: Callable[[], float] = time.perf_counter,
) -> tuple[dict[str, Any], list[list[ScoredDoc]]]:
    """Single-query end-to-end latency of `step` (the full pipeline for one query).

    The first `warmup` queries are run once, untimed. Then every query is timed once, one at a
    time, with CUDA synchronized before the clock is read. Rankings are kept after the clock
    is read. Per-phase times come from the same timed calls.
    """
    for query in queries[:warmup]:
        step(query, PhaseTimer(device, clock=clock))
    sync_device(device)
    timer = PhaseTimer(device, clock=clock)
    timings_ms: list[float] = []
    rankings: list[list[ScoredDoc]] = []
    for query in queries:
        start = clock()
        ranked = step(query, timer)
        sync_device(device)
        timings_ms.append(1000.0 * (clock() - start))
        rankings.append(ranked)
    result: dict[str, Any] = {
        "protocol": "single query, full pipeline, "
        f"{warmup} untimed warm-up queries, each query timed once, CUDA synchronized; "
        "phases are timed inside the same calls",
        "warmup": warmup,
        **summarize_ms(timings_ms),
        "phases": {name: summarize_ms(v) for name, v in timer.samples_ms.items()},
    }
    return result, rankings


# --------------------------------------------------------------------------- candidates


class HybridCandidates:
    """Parent retrieval + fusion. Shared by the hybrid and the hybrid+rerank arms, so both
    see exactly the same fused candidate ordering."""

    def __init__(
        self, bm25: BM25Index, dense: DenseIndex, encoder: Encoder, params: HybridParams, depth: int
    ) -> None:
        self.bm25 = bm25
        self.dense = dense
        self.encoder = encoder
        self.params = params
        self.depth = depth

    def _fuse(
        self, bm25_hits: Sequence[ScoredDoc], dense_hits: Sequence[ScoredDoc]
    ) -> list[ScoredDoc]:
        return rrf_fuse(
            bm25_hits,
            dense_hits,
            k=self.params.rrf_k,
            parent_depth=self.params.parent_depth,
            depth=self.depth,
        )

    def batch(self, queries: Sequence[Query], timer: PhaseTimer) -> tuple[Run, dict[str, int]]:
        """Throughput path: batched query encoding and batched dense search."""
        p = self.params.parent_depth
        with timer.phase("bm25_search"):
            bm25_run = self.bm25.search(queries, p)
        with timer.phase("dense_query_encoding"):
            embeddings = self.encoder.encode_queries(queries)
        with timer.phase("dense_search"):
            dense_run = self.dense.search_embeddings(queries, embeddings, p)
        with timer.phase("fusion"):
            fused = {
                q.query_id: self._fuse(bm25_run[q.query_id], dense_run[q.query_id]) for q in queries
            }
        ids = [q.query_id for q in queries]
        stats = {
            "bm25_empty": sum(1 for i in ids if not bm25_run[i]),
            "bm25_fewer_than_parent_depth": sum(1 for i in ids if len(bm25_run[i]) < p),
            "dense_fewer_than_parent_depth": sum(1 for i in ids if len(dense_run[i]) < p),
            "fused_fewer_than_depth": sum(1 for i in ids if len(fused[i]) < self.depth),
        }
        return fused, stats

    def single(self, query: Query, timer: PhaseTimer) -> list[ScoredDoc]:
        """Timed single-query path: one query encoded and searched on its own."""
        p = self.params.parent_depth
        with timer.phase("bm25_search"):
            bm25_hits = self.bm25.search([query], p)[query.query_id]
        with timer.phase("dense_query_encoding"):
            embedding = self.encoder.encode(self.encoder.query_texts([query]), batch_size=1)[0]
        with timer.phase("dense_search"):
            dense_hits = self.dense.rank(self.dense.scores(embedding), p)
        with timer.phase("fusion"):
            fused: list[ScoredDoc] = self._fuse(bm25_hits, dense_hits)
        return fused
