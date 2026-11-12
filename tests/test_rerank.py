"""Head-only reordering (pure) and the cross-encoder wrapper (tiny local model, CPU)."""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import replace
from pathlib import Path

import pytest

from groundlab.errors import VerificationError
from groundlab.models import ModelSpec
from groundlab.retrieval.rerank import CrossEncoder, RerankParams, rerank_candidates
from groundlab.types import ScoredDoc
from tests.dense_fixtures import TINY_CE_KEY, TINY_REVISION, make_tiny_cross_encoder


def fused_list(n: int) -> list[ScoredDoc]:
    return [ScoredDoc(f"d{i:02d}", 1.0 / (61 + i)) for i in range(n)]


class Scorer:
    """Stub reranker: records what it was asked to score."""

    def __init__(self, logits: Sequence[float]) -> None:
        self.logits = list(logits)
        self.calls: list[list[str]] = []

    def __call__(self, doc_ids: list[str]) -> list[float]:
        self.calls.append(doc_ids)
        return self.logits[: len(doc_ids)]


def ids(run: list[ScoredDoc]) -> list[str]:
    return [h.doc_id for h in run]


def test_only_the_head_is_scored_and_reordered_tail_is_untouched() -> None:
    fused = fused_list(10)
    scorer = Scorer([0.1, 0.9, 0.5, 0.7])
    out = rerank_candidates(fused, scorer, depth=4)
    assert scorer.calls == [["d00", "d01", "d02", "d03"]]  # once, head only, fused order
    assert ids(out)[:4] == ["d01", "d03", "d02", "d00"]
    assert ids(out)[4:] == ids(fused)[4:]  # tail: same members, same order
    assert set(ids(out)[:4]) == set(ids(fused)[:4])  # head membership unchanged
    assert len(out) == len(fused)


def test_reranker_ties_break_by_doc_id_string_order() -> None:
    fused = [ScoredDoc("b", 0.3), ScoredDoc("a", 0.2), ScoredDoc("9", 0.1), ScoredDoc("10", 0.05)]
    out = rerank_candidates(fused, Scorer([2.0, 2.0, 2.0, 2.0]), depth=4)
    assert ids(out) == ["10", "9", "a", "b"]
    # a tie only inside the head: a higher-scored member still wins
    out = rerank_candidates(fused, Scorer([2.0, 2.0, 3.0, 2.0]), depth=4)
    assert ids(out) == ["9", "10", "a", "b"]


def test_short_list_scores_everything() -> None:
    fused = fused_list(3)
    scorer = Scorer([0.0, 1.0, 2.0])
    out = rerank_candidates(fused, scorer, depth=30)
    assert scorer.calls == [["d00", "d01", "d02"]]
    assert ids(out) == ["d02", "d01", "d00"]


def test_list_exactly_at_depth_and_depth_one() -> None:
    assert ids(rerank_candidates(fused_list(30), Scorer(range(30)), depth=30))[0] == "d29"
    out = rerank_candidates(fused_list(5), Scorer([9.0]), depth=1)
    assert ids(out) == ids(fused_list(5))  # a single scored candidate cannot move


def test_empty_list_never_calls_the_model() -> None:
    scorer = Scorer([])
    assert rerank_candidates([], scorer, depth=30) == []
    assert scorer.calls == []


def test_final_scores_are_strictly_descending_ranks() -> None:
    out = rerank_candidates(fused_list(8), Scorer([1.0, 1.0, 1.0]), depth=3)
    scores = [h.score for h in out]
    assert scores == [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
    assert all(math.isfinite(s) for s in scores)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_non_finite_logits_are_rejected(bad: float) -> None:
    with pytest.raises(VerificationError, match="non-finite"):
        rerank_candidates(fused_list(3), Scorer([0.0, bad, 1.0]), depth=3)


def test_wrong_number_of_scores_is_rejected() -> None:
    with pytest.raises(VerificationError, match="scores for 3 candidates"):
        rerank_candidates(fused_list(3), lambda doc_ids: [1.0, 2.0], depth=3)


def test_depth_must_be_positive() -> None:
    with pytest.raises(ValueError):
        rerank_candidates(fused_list(3), Scorer([1.0]), depth=0)


def test_identical_inputs_give_identical_output() -> None:
    logits = [0.3, -1.0, 0.3, 2.5, 0.3]
    a = rerank_candidates(fused_list(7), Scorer(logits), depth=5)
    b = rerank_candidates(fused_list(7), Scorer(logits), depth=5)
    assert a == b
    assert ids(a)[:5] == ["d03", "d00", "d02", "d04", "d01"]


# ------------------------------------------------------------------ params


def tiny_params(**overrides: object) -> RerankParams:
    base = RerankParams(
        model=TINY_CE_KEY,
        revision=TINY_REVISION,
        max_length=24,
        rerank_depth=3,
        batch_size=2,
        attn_implementation="eager",
        dtype="float32",
        device="cuda",
    )
    return replace(base, **overrides)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "override",
    [
        {"attn_implementation": "sdpa"},
        {"dtype": "float16"},
        {"device": "cpu"},
        {"max_length": 513},
        {"max_length": 0},
        {"rerank_depth": 0},
        {"batch_size": 0},
    ],
)
def test_rerank_params_reject_unapproved_numerics(override: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        tiny_params(**override)


# ------------------------------------------------------------------ tiny model


@pytest.fixture(scope="module")
def tiny_ce(tmp_path_factory: pytest.TempPathFactory) -> tuple[ModelSpec, Path]:
    models_dir = tmp_path_factory.mktemp("ce-models")
    return make_tiny_cross_encoder(models_dir), models_dir


@pytest.fixture(scope="module")
def cross_encoder(tiny_ce: tuple[ModelSpec, Path]) -> Iterator[CrossEncoder]:
    spec, models_dir = tiny_ce
    yield CrossEncoder(spec, models_dir, tiny_params(), device="cpu")


DOCS = [
    "alpha\nbeta gamma",
    "delta epsilon zeta eta",
    "protein cell gene",
    "omega psi",
    "title\nbody text claim",
]


def test_scores_are_finite_raw_logits_one_per_document(cross_encoder: CrossEncoder) -> None:
    scores = cross_encoder.score("alpha beta", DOCS)
    assert len(scores) == len(DOCS)
    assert all(isinstance(s, float) and math.isfinite(s) for s in scores)
    assert cross_encoder.score("alpha beta", []) == []


def test_scoring_is_deterministic(cross_encoder: CrossEncoder) -> None:
    assert cross_encoder.score("alpha beta", DOCS) == cross_encoder.score("alpha beta", DOCS)


def test_chunking_does_not_change_scores_materially(
    tiny_ce: tuple[ModelSpec, Path], cross_encoder: CrossEncoder
) -> None:
    spec, models_dir = tiny_ce
    one = CrossEncoder(spec, models_dir, tiny_params(batch_size=1), device="cpu")
    whole = CrossEncoder(spec, models_dir, tiny_params(batch_size=64), device="cpu")
    a, b = one.score("gene", DOCS), whole.score("gene", DOCS)
    assert a == pytest.approx(b, abs=1e-5)
    assert cross_encoder.score("gene", DOCS) == pytest.approx(b, abs=1e-5)


def test_pair_stats_count_truncation_and_reset(tiny_ce: tuple[ModelSpec, Path]) -> None:
    spec, models_dir = tiny_ce
    ce = CrossEncoder(spec, models_dir, tiny_params(max_length=8), device="cpu")
    ce.score("alpha", ["beta", " ".join(["gamma"] * 20)])
    stats = ce.take_stats()
    assert stats == {"pairs_scored": 2, "pairs_document_truncated": 1, "max_pair_tokens": 8}
    assert set(ce.take_stats().values()) == {0}


def test_model_is_loaded_from_pinned_safetensors_only(tiny_ce: tuple[ModelSpec, Path]) -> None:
    spec, models_dir = tiny_ce
    assert "model.safetensors" in spec.files
    kept = {k: v for k, v in spec.files.items() if k != "model.safetensors"}
    no_weights = replace(spec, files=kept)
    with pytest.raises(VerificationError, match="not pinned"):
        CrossEncoder(no_weights, models_dir, tiny_params(), device="cpu")


def test_revision_mismatch_and_tampered_weights_are_refused(
    tiny_ce: tuple[ModelSpec, Path],
) -> None:
    spec, models_dir = tiny_ce
    with pytest.raises(VerificationError, match="revision"):
        CrossEncoder(spec, models_dir, tiny_params(revision="f" * 40), device="cpu")
    wrong = replace(spec, files={**spec.files, "model.safetensors": "0" * 64})
    with pytest.raises(VerificationError, match="SHA-256"):
        CrossEncoder(wrong, models_dir, tiny_params(), device="cpu")


def test_query_is_not_prefixed_and_documents_use_canonical_text(
    cross_encoder: CrossEncoder,
) -> None:
    from groundlab.text import encode_cross_encoder_pair

    enc = encode_cross_encoder_pair(cross_encoder.tokenizer, "alpha", "title\nbody", 24)
    decoded = cross_encoder.tokenizer.convert_ids_to_tokens(enc.input_ids)
    assert decoded == ["[CLS]", "alpha", "[SEP]", "title", "body", "[SEP]"]
