"""Core data types shared across ingestion, retrieval and evaluation.

`Query` deliberately carries only an ID and text. Retrievers receive `Query`
objects and the corpus, never qrels or any dataset metadata.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

# qrels[query_id][doc_id] = graded relevance (int). Grades <= 0 are non-relevant.
Qrels = dict[str, dict[str, int]]


@dataclass(frozen=True, slots=True)
class Document:
    doc_id: str
    title: str
    text: str


@dataclass(frozen=True, slots=True)
class Query:
    query_id: str
    text: str


@dataclass(frozen=True, slots=True)
class ScoredDoc:
    doc_id: str
    score: float


# run[query_id] = ranked list, best first.
Run = dict[str, list[ScoredDoc]]


def rank_scores(scores: Mapping[str, float], depth: int) -> list[ScoredDoc]:
    """Rank by score descending, breaking ties by doc ID ascending (string order).

    This is the single tie-breaking rule used by every ranker in the project.
    """
    if depth <= 0:
        raise ValueError(f"depth must be positive, got {depth}")
    for doc_id, score in scores.items():
        if not math.isfinite(score):
            raise ValueError(f"non-finite score {score!r} for doc {doc_id!r}")
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return [ScoredDoc(doc_id, float(score)) for doc_id, score in ordered[:depth]]
