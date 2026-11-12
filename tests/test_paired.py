"""Tests for the M4 paired statistics (synthetic data only)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from groundlab.evaluation import paired
from groundlab.evaluation.paired import (
    analyze,
    cluster_claims,
    holm_adjust,
    sign_flip_p,
    simulate,
)


def _rng() -> np.random.Generator:
    return np.random.Generator(np.random.PCG64([1]))


def _arms(deltas: dict[str, tuple[float, float, float]]) -> dict[str, dict[str, float]]:
    """Build arms from per-query (dense-bm25, hybrid-dense, rerank-hybrid) deltas."""
    out: dict[str, dict[str, float]] = {a: {} for a in paired.ARMS}
    for q, (d1, d2, d3) in deltas.items():
        out["bm25"][q] = 0.4
        out["dense"][q] = 0.4 + d1
        out["hybrid"][q] = 0.4 + d1 + d2
        out["rerank"][q] = 0.4 + d1 + d2 + d3
    return out


# ------------------------------------------------------------------ clusters


def test_cluster_transitive_and_smallest_id():
    qrels = {
        "q3": {"d1": 1},
        "q2": {"d1": 2, "d2": 1},
        "q1": {"d2": 1},
        "q4": {"d9": 1},
        "q5": {"d3": 0, "d9x": 1},
        "q6": {"d3": 0},  # shared grade-0 doc does not link
    }
    got = cluster_claims(qrels)
    assert got == {"q1": "q1", "q2": "q1", "q3": "q1", "q4": "q4", "q5": "q5", "q6": "q6"}
    assert list(got) == sorted(got)


def test_cluster_chain_is_order_independent():
    a = {"b": {"x": 1}, "a": {"y": 1}, "c": {"x": 1, "y": 1}, "d": {"z": 1}}
    b = dict(reversed(list(a.items())))
    assert cluster_claims(a) == cluster_claims(b)
    assert set(cluster_claims(a).values()) == {"a", "d"}


# ------------------------------------------------------------------ sign flip


def test_exact_sign_flip_two_clusters():
    # signs: totals +2, 0, 0, -2 -> |T| >= 2 in 2 of 4
    assert sign_flip_p(np.array([1.0, 1.0]), _rng()) == 0.5


def test_exact_sign_flip_zeros_are_ignored_and_ties_inclusive():
    assert sign_flip_p(np.array([1.0, 0.0, 0.0]), _rng()) == 1.0
    assert sign_flip_p(np.zeros(5), _rng()) == 1.0
    # equal magnitudes: obs 0 (1-1) -> every sign vector has |T| >= 0
    assert sign_flip_p(np.array([1.0, -1.0]), _rng()) == 1.0


def test_exact_all_same_sign_five_clusters():
    assert sign_flip_p(np.full(5, 0.3), _rng()) == pytest.approx(2 / 32)


def test_monte_carlo_path_and_floor():
    p = sign_flip_p(np.full(30, 1.0), _rng())
    assert 1e-5 <= p < 1e-3
    assert p == sign_flip_p(np.full(30, 1.0), _rng())  # seeded


def test_exact_threshold_is_nonzero_count():
    sums = np.concatenate([np.full(16, 0.5), np.zeros(100)])
    assert sign_flip_p(sums, _rng()) == pytest.approx(2 / 2**16)


# ------------------------------------------------------------------ holm


def test_holm_adjust():
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm_adjust([0.5, 0.9, 0.8]) == pytest.approx([1.0, 1.0, 1.0])
    assert holm_adjust([0.2, 0.2, 0.2]) == pytest.approx([0.6, 0.6, 0.6])
    assert holm_adjust([0.0, 0.0, 0.0]) == [0.0, 0.0, 0.0]


# ------------------------------------------------------------------ analyze


def _toy():
    qs = [f"q{i:02d}" for i in range(24)]
    deltas = {q: (0.05 if i % 3 else -0.01, 0.0, 0.1 * (i % 2)) for i, q in enumerate(qs)}
    clusters = {q: q for q in qs}
    clusters["q01"] = clusters["q00"] = "q00"
    return _arms(deltas), clusters


def test_analyze_shape_and_reproducible():
    arms, clusters = _toy()
    r1 = analyze(arms, clusters)
    r2 = analyze(arms, clusters)
    assert r1 == r2
    assert r1["n_queries"] == 24
    assert r1["n_clusters"] == 23
    assert [c["name"] for c in r1["contrasts"]] == [
        "dense_minus_bm25",
        "hybrid_minus_dense",
        "rerank_minus_hybrid",
    ]
    mid = r1["contrasts"][1]
    assert mid["p_raw"] == 1.0 and mid["p_holm"] == 1.0 and mid["delta"] == pytest.approx(0.0)
    for c in r1["contrasts"]:
        assert c["ci_low"] <= c["delta"] <= c["ci_high"]
        assert c["p_raw"] <= c["p_holm"] <= 1.0
        assert all(isinstance(v, float) for k, v in c.items() if k != "name")


def test_analyze_delta_is_claim_level_mean():
    arms, clusters = _toy()
    r = analyze(arms, clusters)
    exp = np.mean([arms["dense"][q] - arms["bm25"][q] for q in arms["dense"]])
    assert r["contrasts"][0]["delta"] == pytest.approx(exp)


def test_analyze_input_order_does_not_matter():
    arms, clusters = _toy()
    rev_arms = {a: dict(reversed(list(v.items()))) for a, v in reversed(list(arms.items()))}
    rev_clusters = dict(reversed(list(clusters.items())))
    assert analyze(rev_arms, rev_clusters) == analyze(arms, clusters)


def test_analyze_validation():
    arms, clusters = _toy()
    bad = {a: dict(v) for a, v in arms.items()}
    del bad["rerank"]
    with pytest.raises(ValueError):
        analyze(bad, clusters)
    bad = {a: dict(v) for a, v in arms.items()}
    del bad["dense"]["q03"]
    with pytest.raises(ValueError):
        analyze(bad, clusters)
    for value in (math.nan, math.inf, 1.5, -0.1, True):
        bad = {a: dict(v) for a, v in arms.items()}
        bad["hybrid"]["q03"] = value
        with pytest.raises(ValueError):
            analyze(bad, clusters)
    bad_clusters = dict(clusters)
    del bad_clusters["q05"]
    with pytest.raises(ValueError):
        analyze(arms, bad_clusters)
    with pytest.raises(ValueError):
        analyze({a: {} for a in paired.ARMS}, {})


# ------------------------------------------------------------------ simulate


def test_simulation_shape_and_determinism():
    sizes = paired._sim_cluster_sizes()
    assert sum(sizes) == 188 and len(sizes) == 164
    assert sizes.count(1) == 144 and sizes.count(2) == 16 and sizes.count(3) == 4
    r1 = simulate(reps=40, flips=199)
    assert r1 == simulate(reps=40, flips=199)
    assert (r1["n_claims"], r1["n_clusters"]) == (188, 164)
    names = {s["name"] for s in r1["scenarios"]}
    assert {"symmetric_null", "symmetric_power_at_0.02", "skew_pos_margin_boundary"} <= names
    assert all(0.0 <= s["rejection_rate"] <= 1.0 for s in r1["scenarios"])


def test_simulation_default_constants():
    assert (paired.PERM_DRAWS, paired.BOOT_DRAWS) == (99_999, 10_000)
    assert (paired.PERM_SEED, paired.BOOT_SEED, paired.SIM_SEED) == (
        20260411,
        20260412,
        20260413,
    )
