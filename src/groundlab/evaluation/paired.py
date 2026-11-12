"""Paired, cluster-aware statistics for the four-arm comparison (M4).

Design (all choices are fixed before any held-out scoring):

- Claims (queries) that share a positive evidence document are not independent.
  ``cluster_claims`` links them transitively; the inferential unit is the cluster.
- Three confirmatory contrasts, each a mean claim-level nDCG difference:
  ``dense_minus_bm25``, ``hybrid_minus_dense``, ``rerank_minus_hybrid``.
  The dense comparator for hybrid is fixed before held-out scoring.
- p-values: two-sided sign-flip (randomization) test on cluster sums of the
  per-claim deltas with independent +/-1 signs per cluster. Exact enumeration
  when at most ``EXACT_MAX_NONZERO`` clusters have a nonzero sum, otherwise
  ``PERM_DRAWS`` seeded Monte Carlo draws plus the identity (observed) sign
  vector, with inclusive tails: ``p = (1 + #{|T*| >= |T_obs|}) / (PERM_DRAWS + 1)``.
  An all-zero vector has p = 1.
- CI: percentile 95% paired cluster bootstrap. Clusters are resampled with
  replacement (the same resample is applied to all contrasts); the statistic is
  the ratio of the resampled delta sum to the resampled number of claims.
- Holm step-down adjustment over the three raw p-values.
- ``PRACTICAL_DELTA`` (+0.02 nDCG) is a descriptive practical threshold for
  reporting only. It is not a margin test and is not used by any p-value.

Determinism: queries and cluster IDs are sorted lexicographically; every random
stream is a NumPy ``Generator(PCG64(SeedSequence([seed, stream])))``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from groundlab.types import Qrels

ARMS: tuple[str, ...] = ("bm25", "dense", "hybrid", "rerank")
CONTRASTS: tuple[tuple[str, str, str], ...] = (
    ("dense_minus_bm25", "dense", "bm25"),
    ("hybrid_minus_dense", "hybrid", "dense"),
    ("rerank_minus_hybrid", "rerank", "hybrid"),
)

PERM_SEED = 20260411  # sign-flip streams: PCG64([PERM_SEED, contrast_index])
BOOT_SEED = 20260412  # bootstrap stream: PCG64([BOOT_SEED])
SIM_SEED = 20260413  # simulation streams: PCG64([SIM_SEED, stream])
PERM_DRAWS = 99_999
BOOT_DRAWS = 10_000
EXACT_MAX_NONZERO = 16
CI_LEVEL = 0.95
PRACTICAL_DELTA = 0.02  # descriptive only

_ZERO_TOL = 1e-12
_NDCG_SLACK = 1e-9
_PERM_CHUNK = 5_000
_BOOT_CHUNK = 2_000


# --------------------------------------------------------------------------- clusters


def cluster_claims(qrels: Qrels) -> dict[str, str]:
    """Map each query to its cluster ID.

    Queries are linked when they share a positive (grade >= 1) document; links
    are transitive (connected components). The cluster ID is the
    lexicographically smallest query ID in the component. Queries without
    positives are singletons. The result is sorted by query ID.
    """
    parent: dict[str, str] = {qid: qid for qid in qrels}

    def find(x: str) -> str:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        lo, hi = (ra, rb) if ra < rb else (rb, ra)
        parent[hi] = lo

    first_owner: dict[str, str] = {}
    for qid in sorted(qrels):
        for doc in sorted(d for d, g in qrels[qid].items() if g >= 1):
            owner = first_owner.setdefault(doc, qid)
            if owner != qid:
                union(owner, qid)
    return {qid: find(qid) for qid in sorted(qrels)}


# --------------------------------------------------------------------------- p-values


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm step-down adjusted p-values, returned in the input order.

    Ties are ordered by input position, so the result is deterministic.
    """
    m = len(p_values)
    order = sorted(range(m), key=lambda i: (p_values[i], i))
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p_values[i])
        adjusted[i] = min(1.0, running)
    return adjusted


def _exact_p(nonzero: np.ndarray, obs: float, tol: float) -> float:
    k = nonzero.size
    codes = np.arange(1 << k, dtype=np.int64)
    bits = (codes[:, None] >> np.arange(k, dtype=np.int64)) & 1
    totals = (1.0 - 2.0 * bits) @ nonzero
    return float(np.count_nonzero(np.abs(totals) >= obs - tol)) / float(1 << k)


def _mc_p(nonzero: np.ndarray, obs: float, tol: float, rng: np.random.Generator) -> float:
    k = nonzero.size
    count = 0
    remaining = PERM_DRAWS
    while remaining > 0:
        c = min(_PERM_CHUNK, remaining)
        signs = rng.integers(0, 2, size=(c, k), dtype=np.int8).astype(np.float64)
        signs = 2.0 * signs - 1.0
        count += int(np.count_nonzero(np.abs(signs @ nonzero) >= obs - tol))
        remaining -= c
    return (1.0 + count) / (PERM_DRAWS + 1.0)


def sign_flip_p(cluster_sums: np.ndarray, rng: np.random.Generator) -> float:
    """Two-sided sign-flip p-value for a vector of cluster delta sums."""
    sums = np.asarray(cluster_sums, dtype=np.float64)
    nonzero = sums[np.abs(sums) > _ZERO_TOL]
    if nonzero.size == 0:
        return 1.0
    obs = abs(float(nonzero.sum()))
    tol = _ZERO_TOL * float(np.abs(nonzero).sum())
    if nonzero.size <= EXACT_MAX_NONZERO:
        return _exact_p(nonzero, obs, tol)
    return _mc_p(nonzero, obs, tol, rng)


# --------------------------------------------------------------------------- analyze


def _validate(per_arm: Mapping[str, Mapping[str, float]], clusters: Mapping[str, str]) -> list[str]:
    if set(per_arm) != set(ARMS):
        raise ValueError(f"per_arm must have exactly the arms {list(ARMS)}; got {sorted(per_arm)}")
    queries = sorted(per_arm[ARMS[0]])
    if not queries:
        raise ValueError("no queries to analyze")
    qset = set(queries)
    for arm in ARMS:
        if set(per_arm[arm]) != qset:
            raise ValueError(f"arm {arm!r} does not have exactly the same query set as {ARMS[0]!r}")
        for qid in queries:
            v = per_arm[arm][qid]
            if isinstance(v, bool) or not isinstance(v, (int, float, np.integer, np.floating)):
                raise ValueError(f"nDCG for {arm}/{qid} is not a real number: {v!r}")
            if not math.isfinite(v) or v < -_NDCG_SLACK or v > 1.0 + _NDCG_SLACK:
                raise ValueError(f"nDCG for {arm}/{qid} must be finite in [0, 1]; got {v!r}")
    if set(clusters) != qset:
        raise ValueError("clusters must have exactly the same query set as the arms")
    for qid in queries:
        if not isinstance(clusters[qid], str):
            raise ValueError(f"cluster ID for {qid!r} must be a string")
    return queries


def analyze(per_arm: dict[str, dict[str, float]], clusters: dict[str, str]) -> dict[str, Any]:
    """Run the three confirmatory paired contrasts.

    Returns ``{"n_queries", "n_clusters", "contrasts": [...]}`` where each
    contrast has ``name, delta, ci_low, ci_high, p_raw, p_holm`` (plain floats).
    """
    queries = _validate(per_arm, clusters)
    cluster_ids = sorted({clusters[q] for q in queries})
    cindex = {c: i for i, c in enumerate(cluster_ids)}
    k = len(cluster_ids)
    qcluster = np.array([cindex[clusters[q]] for q in queries], dtype=np.int64)
    counts = np.bincount(qcluster, minlength=k).astype(np.float64)

    sums = np.empty((len(CONTRASTS), k), dtype=np.float64)
    deltas: list[float] = []
    for j, (_, a, b) in enumerate(CONTRASTS):
        d = np.array([float(per_arm[a][q]) - float(per_arm[b][q]) for q in queries])
        sums[j] = np.bincount(qcluster, weights=d, minlength=k)
        deltas.append(float(sums[j].sum() / len(queries)))

    p_raw = [
        sign_flip_p(sums[j], np.random.Generator(np.random.PCG64([PERM_SEED, j])))
        for j in range(len(CONTRASTS))
    ]
    p_holm = holm_adjust(p_raw)

    # Paired cluster bootstrap: one resample stream shared by all contrasts.
    boot: list[list[np.ndarray]] = [[] for _ in CONTRASTS]
    rng = np.random.Generator(np.random.PCG64([BOOT_SEED]))
    remaining = BOOT_DRAWS
    while remaining > 0:
        c = min(_BOOT_CHUNK, remaining)
        idx = rng.integers(0, k, size=(c, k))
        den = counts[idx].sum(axis=1)
        for j in range(len(CONTRASTS)):
            boot[j].append(sums[j][idx].sum(axis=1) / den)
        remaining -= c
    alpha = (1.0 - CI_LEVEL) / 2.0 * 100.0

    contrasts = []
    for j, (name, _, _) in enumerate(CONTRASTS):
        draws = np.concatenate(boot[j])
        lo, hi = np.percentile(draws, [alpha, 100.0 - alpha])
        contrasts.append(
            {
                "name": name,
                "delta": deltas[j],
                "ci_low": float(lo),
                "ci_high": float(hi),
                "p_raw": float(p_raw[j]),
                "p_holm": float(p_holm[j]),
            }
        )
    return {"n_queries": len(queries), "n_clusters": k, "contrasts": contrasts}


# --------------------------------------------------------------------------- simulation

_SIM_ALPHA = 0.05
_SIM_ZERO_PROB = 0.5  # share of claims whose delta is exactly 0 (ties)
_SIM_CLUSTER_SD = 0.10
_SIM_CLAIM_SD = 0.20


def _sim_cluster_sizes() -> list[int]:
    """188 claims in 164 clusters: 144 singletons, 16 pairs, 4 triples."""
    return [1] * 144 + [2] * 16 + [3] * 4


def _sim_noise(rng: np.random.Generator, kind: str, size: Any) -> np.ndarray:
    """Bounded, discrete, zero-mean, unit-variance noise."""
    if kind == "symmetric":
        return np.asarray(2.0 * rng.integers(0, 2, size=size) - 1.0)
    sign = 1.0 if kind == "skew_pos" else -1.0
    p = 0.15
    return np.asarray(sign * ((rng.random(size) < p) - p) / math.sqrt(p * (1.0 - p)))


def simulate(*, reps: int = 600, flips: int = 1999) -> dict[str, Any]:
    """Deterministic synthetic stress test of the sign-flip test.

    Synthetic claim-level deltas only; no real labels, data or results. Each
    delta is ``z * (mu / (1 - p0) + 0.10 * g_cluster + 0.20 * e_claim)`` with
    ``z ~ Bernoulli(1 - p0)`` (zero inflation, p0 = 0.5), so ``E[delta] = mu``.
    ``g`` and ``e`` are zero-mean unit-variance noise (Rademacher for the symmetric
    scenarios, centered Bernoulli for the skewed ones). The cluster shock is
    shared by all claims in a cluster. The test applied is the sign-flip test
    of ``mean(delta) = margin`` on cluster sums, using ``flips`` random sign
    vectors (one sign matrix reused for all replicates and scenarios) and the
    inclusive-tail p-value ``(1 + #{|T*| >= |T|}) / (flips + 1)``; rejection is
    ``p <= 0.05``.
    """
    if reps <= 0 or flips <= 0:
        raise ValueError("simulation requires positive reps and flips")
    sizes = _sim_cluster_sizes()
    n_clusters = len(sizes)
    n_claims = sum(sizes)
    assert (n_claims, n_clusters) == (188, 164)
    sizes_arr = np.array(sizes, dtype=np.int64)
    starts = np.concatenate(([0], np.cumsum(sizes_arr)[:-1]))
    claim_cluster = np.repeat(np.arange(n_clusters), sizes_arr)

    sign_rng = np.random.Generator(np.random.PCG64([SIM_SEED, 0]))
    signs = 2.0 * sign_rng.integers(0, 2, size=(flips, n_clusters), dtype=np.int8) - 1.0

    # (name, noise kind, true mean, tested margin, role)
    specs = [
        ("symmetric_null", "symmetric", 0.0, 0.0, "size"),
        ("symmetric_power_at_0.02", "symmetric", 0.02, 0.0, "power"),
    ]
    for kind in ("skew_pos", "skew_neg"):
        specs += [
            (f"{kind}_null", kind, 0.0, 0.0, "exploratory_size_skewed"),
            (f"{kind}_power_at_0.02", kind, 0.02, 0.0, "exploratory_power_skewed"),
            (f"{kind}_margin_boundary", kind, 0.02, 0.02, "exploratory_size_at_margin"),
        ]

    scenarios = []
    for stream, (name, kind, mu, margin, role) in enumerate(specs, start=1):
        rng = np.random.Generator(np.random.PCG64([SIM_SEED, stream]))
        shock = _sim_noise(rng, kind, (reps, n_clusters))[:, claim_cluster]
        claim = _sim_noise(rng, kind, (reps, n_claims))
        nonzero = rng.random((reps, n_claims)) >= _SIM_ZERO_PROB
        delta = nonzero * (
            mu / (1.0 - _SIM_ZERO_PROB) + _SIM_CLUSTER_SD * shock + _SIM_CLAIM_SD * claim
        )
        csum = np.add.reduceat(delta - margin, starts, axis=1)
        obs = np.abs(csum.sum(axis=1))
        tol = _ZERO_TOL * np.abs(csum).sum(axis=1)
        tail = np.abs(csum @ signs.T) >= (obs - tol)[:, None]
        p = (1.0 + tail.sum(axis=1)) / (flips + 1.0)
        rate = float(np.mean(p <= _SIM_ALPHA))
        scenarios.append(
            {
                "name": name,
                "role": role,
                "noise": kind,
                "true_mean": mu,
                "tested_margin": margin,
                "rejection_rate": rate,
                "mc_se": math.sqrt(rate * (1.0 - rate) / reps),
            }
        )

    return {
        "seed": SIM_SEED,
        "reps": reps,
        "flips": flips,
        "alpha": _SIM_ALPHA,
        "n_claims": n_claims,
        "n_clusters": n_clusters,
        "cluster_sizes": {"1": 144, "2": 16, "3": 4},
        "zero_inflation": _SIM_ZERO_PROB,
        "scenarios": scenarios,
        "construction": (
            "delta = z*(mu/(1-p0) + 0.10*cluster_shock + 0.20*claim_noise), z~Bernoulli(0.5); "
            "cluster shock shared within clusters; skewed scenarios use standardized Bernoulli "
            "noise (p=0.15, mirrored for negative skew) with the same variances and mean mu."
        ),
        "limitations": [
            "Synthetic deltas are bounded and discrete, but do not model actual relevance ranks.",
            "The shape (188/164) is representative, not derived from real data.",
            "Simulation uses 1999 flips and a shared sign matrix, not the 99,999-draw "
            "or exact-enumeration path of analyze(); bootstrap CIs are not simulated.",
            "Rates have Monte Carlo error (see mc_se); with 600 reps the SE near 0.05 is ~0.009.",
            "margin_boundary scenarios are exploratory: the sign-flip test assumes symmetry "
            "of cluster sums about the tested value, which skew violates; the +0.02 "
            "threshold is descriptive in analyze() and no margin test is performed there.",
        ],
    }
