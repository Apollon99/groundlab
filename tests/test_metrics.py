"""Metric fixtures with hand-computed expected values (worked arithmetic in comments)."""

from __future__ import annotations

import math

import pytest

from groundlab.evaluation import metrics as M
from groundlab.evaluation.trec_check import cross_check
from groundlab.types import rank_scores

LOG2_3 = math.log2(3)  # 1.5849625007211562

# q1: two relevant docs, d2 more relevant (grade 2); retrieved order d3, d1, d2, d4.
# q2: judged but absent from the run.
QRELS = {"q1": {"d1": 1, "d2": 2}, "q2": {"d5": 1}}
RUN = {"q1": ["d3", "d1", "d2", "d4"]}
ALL = [
    M.parse_metric(x)
    for x in ("ndcg@1", "ndcg@3", "recall@1", "recall@2", "recall@3", "mrr@1", "mrr@10")
]


def test_ndcg_hand_computed() -> None:
    # DCG@3  = 0/log2(2) + 1/log2(3) + 2/log2(4) = 0.6309297535714575 + 1.0
    # IDCG@3 = 2/log2(2) + 1/log2(3)             = 2.0 + 0.6309297535714575
    # nDCG@3 = 1.6309297535714575 / 2.6309297535714575 = 0.6199062332840657
    assert M.ndcg_at_k(RUN["q1"], QRELS["q1"], 3) == pytest.approx(0.6199062332840657, abs=1e-12)
    assert M.ndcg_at_k(RUN["q1"], QRELS["q1"], 1) == 0.0


def test_ndcg_binary_single_relevant_at_rank_two() -> None:
    # One relevant doc at rank 2: DCG = 1/log2(3), IDCG = 1.
    assert M.ndcg_at_k(["x", "r"], {"r": 1}, 10) == pytest.approx(1 / LOG2_3, abs=1e-12)
    assert M.ndcg_at_k(["x", "r"], {"r": 1}, 10) == pytest.approx(0.6309297535714575, abs=1e-12)


def test_ndcg_graded_swap() -> None:
    # Grade-1 doc ranked above the grade-2 doc:
    # DCG = 1/1 + 2/log2(3) = 2.2618595071429148 ; IDCG = 2/1 + 1/log2(3) = 2.6309297535714575
    expected = (1 + 2 / LOG2_3) / (2 + 1 / LOG2_3)
    assert expected == pytest.approx(0.8597186998521192, abs=1e-12)
    assert M.ndcg_at_k(["b", "a"], {"a": 2, "b": 1}, 10) == pytest.approx(expected, abs=1e-12)


def test_recall_and_mrr_hand_computed() -> None:
    assert M.recall_at_k(RUN["q1"], QRELS["q1"], 1) == 0.0
    assert M.recall_at_k(RUN["q1"], QRELS["q1"], 2) == 0.5  # d1 of {d1, d2}
    assert M.recall_at_k(RUN["q1"], QRELS["q1"], 3) == 1.0
    assert M.mrr_at_k(RUN["q1"], QRELS["q1"], 1) == 0.0
    assert M.mrr_at_k(RUN["q1"], QRELS["q1"], 10) == 0.5  # first relevant at rank 2


def test_grade_zero_is_not_relevant() -> None:
    assert M.recall_at_k(["a", "b"], {"a": 0, "b": 1}, 1) == 0.0
    with pytest.raises(ValueError):
        M.ndcg_at_k(["a"], {"a": 0}, 10)


def test_missing_query_scores_zero_and_counts_in_mean() -> None:
    per_q = M.per_query(RUN, QRELS, ALL)
    assert set(per_q) == {"q1", "q2"}
    assert all(v == 0.0 for v in per_q["q2"].values())
    means = M.mean(per_q, ALL)
    # (0.6199062332840657 + 0) / 2
    assert means["ndcg@3"] == pytest.approx(0.30995311664203284, abs=1e-12)
    assert means["recall@3"] == 0.5
    assert means["mrr@10"] == 0.25


def test_run_query_not_in_qrels_is_an_error() -> None:
    with pytest.raises(ValueError, match="not in qrels"):
        M.per_query({"q1": ["d1"], "qX": ["d1"]}, QRELS, ALL)


def test_duplicate_doc_in_ranking_is_an_error() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        M.per_query({"q1": ["d1", "d1"]}, {"q1": {"d1": 1}}, ALL)


def test_tie_break_by_doc_id_ascending() -> None:
    ranked = rank_scores({"b": 1.0, "a": 1.0, "c": 2.0}, depth=3)
    assert [h.doc_id for h in ranked] == ["c", "a", "b"]
    # String order, not numeric: "10" < "9".
    assert [h.doc_id for h in rank_scores({"9": 0.5, "10": 0.5}, depth=2)] == ["10", "9"]
    # With b relevant, the tie-break decides MRR: b sits at rank 3 -> 1/3.
    ranking = [h.doc_id for h in ranked]
    assert M.mrr_at_k(ranking, {"b": 1}, 10) == pytest.approx(1 / 3, abs=1e-12)


def test_rank_scores_rejects_nan_and_bad_depth() -> None:
    with pytest.raises(ValueError):
        rank_scores({"a": float("nan")}, depth=1)
    with pytest.raises(ValueError):
        rank_scores({"a": 1.0}, depth=0)


def test_parse_metric() -> None:
    assert M.parse_metric("ndcg@10") == M.Metric("ndcg", 10)
    for bad in ("ndcg", "map@10", "ndcg@0", "recall@-1"):
        with pytest.raises(ValueError):
            M.parse_metric(bad)


def test_pytrec_eval_agrees_on_fixtures() -> None:
    qrels = {**QRELS, "q3": {"a": 2, "b": 1, "z": 1}}
    run = {**RUN, "q3": ["c", "a", "b"] + [f"n{i}" for i in range(20)]}
    per_q = M.per_query(run, qrels, ALL)
    diffs = cross_check(run, qrels, ALL, per_q)
    assert set(diffs) == {m.label for m in ALL}
    assert max(diffs.values()) <= 1e-9
