"""Independent cross-check of native metrics against pytrec_eval (trec_eval).

trec_eval re-sorts by score and breaks ties by doc ID in its own order, so the
run is passed with synthetic strictly decreasing scores derived from our rank
order. MRR@k is computed as trec_eval `recip_rank` on the run truncated to k.

There is no fallback: if pytrec_eval is missing or disagrees, this raises.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pytrec_eval

from groundlab.errors import VerificationError
from groundlab.evaluation.metrics import Metric
from groundlab.types import Qrels

TOLERANCE = 1e-9


def _trec_measure(metric: Metric) -> tuple[str, str]:
    if metric.name == "ndcg":
        return "ndcg_cut", f"ndcg_cut_{metric.k}"
    if metric.name == "recall":
        return "recall", f"recall_{metric.k}"
    return "recip_rank", "recip_rank"


def trec_per_query(
    run: Mapping[str, Sequence[str]], qrels: Qrels, metric: Metric
) -> dict[str, float]:
    family, key = _trec_measure(metric)
    measure = family if family == "recip_rank" else f"{family}.{metric.k}"
    truncate = metric.k if metric.name == "mrr" else None
    trec_run = {
        qid: {d: float(len(ranking) - i) for i, d in enumerate(ranking[:truncate])}
        for qid, ranking in run.items()
        if ranking
    }
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {measure})
    results = evaluator.evaluate(trec_run)
    # trec_eval skips queries absent from the run; our convention scores them 0.
    return {qid: float(results[qid][key]) if qid in results else 0.0 for qid in qrels}


def cross_check(
    run: Mapping[str, Sequence[str]],
    qrels: Qrels,
    metrics: Sequence[Metric],
    native: Mapping[str, Mapping[str, float]],
) -> dict[str, float]:
    """Raise VerificationError on any per-query disagreement; return max |diff| per metric."""
    max_diff: dict[str, float] = {}
    for metric in metrics:
        trec = trec_per_query(run, qrels, metric)
        worst = 0.0
        for qid, value in trec.items():
            diff = abs(value - native[qid][metric.label])
            if diff > TOLERANCE:
                raise VerificationError(
                    f"{metric.label} mismatch on query {qid}: native "
                    f"{native[qid][metric.label]!r} vs pytrec_eval {value!r}"
                )
            worst = max(worst, diff)
        max_diff[metric.label] = worst
    return max_diff
