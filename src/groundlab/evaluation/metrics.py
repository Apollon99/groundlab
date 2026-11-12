"""Native ranking metrics.

Conventions (D-010):
- A run maps query ID -> ranked doc IDs (best first). Rank order is taken from
  the list, never re-sorted by score.
- Relevant means grade >= 1. nDCG uses linear gain = grade and discount
  1/log2(rank + 1), with the ideal ranking built from all judged grades >= 1.
- The evaluated query set is the qrels query set. A qrels query absent from the
  run scores 0 on every metric and counts in the mean.
- A run query absent from the qrels is an error (it would signal a split mix-up).
- A qrels query with no relevant document is an error (the loader never emits one).
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from groundlab.types import Qrels

_METRIC_RE = re.compile(r"^(ndcg|recall|mrr)@([1-9][0-9]*)$")


@dataclass(frozen=True)
class Metric:
    name: str
    k: int

    @property
    def label(self) -> str:
        return f"{self.name}@{self.k}"


def parse_metric(label: str) -> Metric:
    match = _METRIC_RE.match(label)
    if not match:
        raise ValueError(f"unsupported metric {label!r}; use ndcg@k, recall@k or mrr@k")
    return Metric(match.group(1), int(match.group(2)))


def _relevant(judgments: Mapping[str, int]) -> dict[str, int]:
    return {d: g for d, g in judgments.items() if g >= 1}


def ndcg_at_k(ranking: Sequence[str], judgments: Mapping[str, int], k: int) -> float:
    rel = _relevant(judgments)
    if not rel:
        raise ValueError("nDCG undefined for a query with no relevant documents")
    dcg = sum(rel.get(d, 0) / math.log2(i + 2) for i, d in enumerate(ranking[:k]))
    ideal = sorted(rel.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg


def recall_at_k(ranking: Sequence[str], judgments: Mapping[str, int], k: int) -> float:
    rel = _relevant(judgments)
    if not rel:
        raise ValueError("recall undefined for a query with no relevant documents")
    return len(set(ranking[:k]) & rel.keys()) / len(rel)


def mrr_at_k(ranking: Sequence[str], judgments: Mapping[str, int], k: int) -> float:
    rel = _relevant(judgments)
    if not rel:
        raise ValueError("MRR undefined for a query with no relevant documents")
    for i, d in enumerate(ranking[:k]):
        if d in rel:
            return 1.0 / (i + 1)
    return 0.0


_FUNCS = {"ndcg": ndcg_at_k, "recall": recall_at_k, "mrr": mrr_at_k}


def check_ranking(qid: str, ranking: Sequence[str]) -> None:
    if len(set(ranking)) != len(ranking):
        raise ValueError(f"query {qid}: duplicate doc IDs in ranking")


def per_query(
    run: Mapping[str, Sequence[str]], qrels: Qrels, metrics: Sequence[Metric]
) -> dict[str, dict[str, float]]:
    extra = sorted(run.keys() - qrels.keys())
    if extra:
        raise ValueError(f"run contains {len(extra)} queries not in qrels, e.g. {extra[:3]}")
    out: dict[str, dict[str, float]] = {}
    for qid in sorted(qrels):
        ranking = run.get(qid, [])
        check_ranking(qid, ranking)
        out[qid] = {m.label: _FUNCS[m.name](ranking, qrels[qid], m.k) for m in metrics}
    return out


def mean(per_q: Mapping[str, Mapping[str, float]], metrics: Sequence[Metric]) -> dict[str, float]:
    if not per_q:
        raise ValueError("no queries to average")
    return {m.label: math.fsum(v[m.label] for v in per_q.values()) / len(per_q) for m in metrics}
