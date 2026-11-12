"""Reciprocal rank fusion and phase timing. Pure Python, no model or GPU."""

from __future__ import annotations

import math
from typing import Any

import pytest

from groundlab.errors import VerificationError
from groundlab.retrieval.hybrid import (
    HybridParams,
    PhaseTimer,
    measure_pipeline_latency,
    rrf_fuse,
)
from groundlab.types import Query, ScoredDoc


def hits(*ids: str) -> list[ScoredDoc]:
    """A best-first parent list; the scores are irrelevant to RRF and deliberately odd."""
    return [ScoredDoc(i, 100.0 - n) for n, i in enumerate(ids)]


def ids(fused: list[ScoredDoc]) -> list[str]:
    return [h.doc_id for h in fused]


def test_overlapping_lists_sum_both_contributions() -> None:
    fused = rrf_fuse(hits("A", "B", "C"), hits("B", "D", "A"))
    scores = {h.doc_id: h.score for h in fused}
    assert scores["A"] == pytest.approx(1 / 61 + 1 / 63, abs=1e-15)
    assert scores["B"] == pytest.approx(1 / 62 + 1 / 61, abs=1e-15)
    assert scores["C"] == pytest.approx(1 / 63, abs=1e-15)
    assert scores["D"] == pytest.approx(1 / 62, abs=1e-15)
    assert ids(fused) == ["B", "A", "D", "C"]


def test_ranks_are_one_based_and_k_is_60() -> None:
    (only,) = rrf_fuse(hits("A"), [])
    assert only.score == 1 / 61  # not 1/60 (0-based) and not 1/62


def test_disjoint_lists_interleave_with_ties_by_doc_id() -> None:
    fused = rrf_fuse(hits("b1", "b2"), hits("d1", "d2"))
    # rank 1 of each list ties (1/61), rank 2 ties (1/62); tie order is doc ID ascending
    assert ids(fused) == ["b1", "d1", "b2", "d2"]
    assert fused[0].score == fused[1].score > fused[2].score == fused[3].score


def test_empty_bm25_list_gives_dense_order_without_padding() -> None:
    fused = rrf_fuse([], hits("x", "w", "y"))
    assert ids(fused) == ["x", "w", "y"]  # rank order, not doc-ID order
    assert [h.score for h in fused] == [1 / 61, 1 / 62, 1 / 63]


def test_empty_dense_list_and_both_empty() -> None:
    assert ids(rrf_fuse(hits("p", "o"), [])) == ["p", "o"]
    assert rrf_fuse([], []) == []


def test_absent_contribution_is_zero_not_a_penalty_rank() -> None:
    # B is only in BM25 (rank 2). If an absent hit were padded at rank 101 it would still lose
    # to C; the real check is the exact score.
    fused = rrf_fuse(hits("A", "B"), hits("A"))
    assert {h.doc_id: h.score for h in fused}["B"] == 1 / 62


def test_tied_scores_break_by_doc_id_string_order() -> None:
    # "10" < "9" in string order
    fused = rrf_fuse(hits("9"), hits("10"))
    assert ids(fused) == ["10", "9"]
    assert fused[0].score == fused[1].score


def test_mathematically_equal_sums_tie_exactly() -> None:
    # k=1: X = 1/(1+2) + 1/(1+5) = 1/2 and Y = 1/(1+1) = 1/2. Floating point may separate
    # them by one ulp; the exact rational sum never does, so the doc ID must decide.
    for x_id, y_id in (("a-two-terms", "z-one-term"), ("z-two-terms", "a-one-term")):
        bm25 = hits(y_id, x_id)
        dense = hits("f1", "f2", "f3", "f4", x_id)
        fused = rrf_fuse(bm25, dense, k=1)
        scores = {h.doc_id: h.score for h in fused}
        assert scores[x_id] == scores[y_id] == 0.5
        assert ids(fused).index(min(x_id, y_id)) < ids(fused).index(max(x_id, y_id))


def test_only_first_parent_depth_hits_enter_fusion() -> None:
    bm25 = hits(*[f"b{i:03d}" for i in range(150)])
    dense = hits(*[f"d{i:03d}" for i in range(150)])
    fused = rrf_fuse(bm25, dense, depth=500)
    assert len(fused) == 200
    assert not any(h.doc_id.endswith(("100", "149")) for h in fused)
    assert "b099" in ids(fused) and "d099" in ids(fused)


def test_output_is_cut_to_depth() -> None:
    bm25 = hits(*[f"b{i:03d}" for i in range(100)])
    dense = hits(*[f"d{i:03d}" for i in range(100)])
    assert len(rrf_fuse(bm25, dense)) == 100
    assert len(rrf_fuse(bm25, dense, depth=7)) == 7


def test_short_lists() -> None:
    assert ids(rrf_fuse(hits("A"), hits("B"))) == ["A", "B"]
    assert len(rrf_fuse(hits("A", "B"), hits("A", "B"))) == 2


def test_scores_are_finite_positive_and_non_increasing() -> None:
    bm25 = hits(*[f"d{i:03d}" for i in range(0, 100, 2)])
    dense = hits(*[f"d{i:03d}" for i in range(99, -1, -1)])
    fused = rrf_fuse(bm25, dense)
    assert all(math.isfinite(h.score) and h.score > 0 for h in fused)
    scores = [h.score for h in fused]
    assert scores == sorted(scores, reverse=True)
    assert len({h.doc_id for h in fused}) == len(fused)


def test_fusion_is_symmetric_in_its_inputs() -> None:
    a, b = hits("A", "B", "C"), hits("C", "D", "A")
    assert rrf_fuse(a, b) == rrf_fuse(b, a)


def test_duplicate_document_in_one_list_is_an_error() -> None:
    with pytest.raises(VerificationError, match="duplicate"):
        rrf_fuse(hits("A", "A"), [])


def test_hybrid_params_pin_k_and_depth() -> None:
    HybridParams(fusion="rrf", rrf_k=60, parent_depth=100)
    good: dict[str, Any] = {"fusion": "rrf", "rrf_k": 60, "parent_depth": 100}
    for bad in ({"fusion": "sum"}, {"rrf_k": 61}, {"parent_depth": 50}):
        with pytest.raises(ValueError):
            HybridParams(**{**good, **bad})


# ------------------------------------------------------------------ timing helpers


def test_phase_timer_collects_per_phase_samples() -> None:
    ticks = iter(range(100))
    timer = PhaseTimer("cpu", clock=lambda: float(next(ticks)))
    for _ in range(3):
        with timer.phase("a"):
            pass
    with timer.phase("b"):
        pass
    assert timer.samples_ms == {"a": [1000.0] * 3, "b": [1000.0]}
    assert timer.total_seconds() == {"a": 3.0, "b": 1.0}
    assert timer.overall_peak() is None  # CPU: no CUDA counters


def test_latency_warmup_is_untimed_and_percentiles_cover_timed_queries_only() -> None:
    queries = [Query(f"q{i}", "t") for i in range(5)]
    calls: list[str] = []

    def step(query: Query, timer: PhaseTimer) -> list[ScoredDoc]:
        calls.append(query.query_id)
        with timer.phase("work"):
            pass
        return [ScoredDoc(f"doc-{len(calls)}", 1.0)]

    ticks = iter(range(1000))
    stats, rankings = measure_pipeline_latency(
        step, queries, warmup=2, device="cpu", clock=lambda: float(next(ticks))
    )
    assert calls == ["q0", "q1"] + [q.query_id for q in queries]  # 2 warm-up + 5 timed
    assert stats["n"] == 5 and stats["warmup"] == 2
    assert stats["phases"]["work"]["n"] == 5  # warm-up phases are not in the timer
    assert [r[0].doc_id for r in rankings] == [f"doc-{i}" for i in range(3, 8)]
    assert {"p50_ms", "p90_ms", "p99_ms", "mean_ms"} <= stats.keys()
