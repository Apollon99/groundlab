"""Dense retrieval: encoder semantics, embedding cache integrity, exact search, latency."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from groundlab import models
from groundlab.errors import VerificationError
from groundlab.models import ModelSpec
from groundlab.retrieval.dense import DenseIndex, DenseParams, Encoder, measure_latency
from groundlab.text import canonical_document_text
from groundlab.types import Document, Query
from tests.dense_fixtures import TINY_PARAMS, make_tiny_model, params

DOCS = [
    Document("d2", "Alpha", "beta gamma"),
    Document("d1", "Delta", "epsilon zeta eta"),
    Document("d3", "", "theta iota kappa"),
    Document("d0", "Omega", "psi"),
]


@pytest.fixture(scope="module")
def tiny(tmp_path_factory: pytest.TempPathFactory) -> tuple[ModelSpec, Path]:
    models_dir = tmp_path_factory.mktemp("models")
    return make_tiny_model(models_dir), models_dir


@pytest.fixture(scope="module")
def encoder(tiny: tuple[ModelSpec, Path]) -> Iterator[Encoder]:
    spec, models_dir = tiny
    yield Encoder(spec, models_dir, TINY_PARAMS, device="cpu")


def test_canonical_document_text() -> None:
    assert canonical_document_text(Document("x", "Title", "Body")) == "Title\nBody"
    assert canonical_document_text(Document("x", "", "Body")) == "Body"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("pooling", "mean"),
        ("normalize", False),
        ("truncation_side", "left"),
        ("document_instruction", "passage: "),
        ("document_separator", " "),
        ("attn_implementation", "sdpa"),
        ("dtype", "float16"),
        ("device", "cpu"),
        ("max_length", 513),
        ("batch_size", 0),
    ],
)
def test_dense_params_reject_non_approved_settings(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        params(**{field: value})


def test_encoder_uses_cls_pooling_and_l2_normalization(encoder: Encoder) -> None:
    import torch

    texts = ["alpha beta", "gamma delta epsilon"]
    emb = encoder.encode(texts)
    enc = encoder.tokenizer(texts, padding="longest", return_tensors="pt")
    with torch.inference_mode():
        cls = encoder.model(**enc).last_hidden_state[:, 0]
    manual = torch.nn.functional.normalize(cls, dim=-1).numpy()
    np.testing.assert_allclose(emb, manual, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-6)
    assert emb.dtype == np.float32


def test_encoder_truncates_from_the_end(tiny: tuple[ModelSpec, Path]) -> None:
    spec, models_dir = tiny
    enc = Encoder(spec, models_dir, params(max_length=6), device="cpu")
    # [CLS] + 4 tokens + [SEP] fit; everything after the 4th word is cut.
    a, b = enc.encode(["alpha beta gamma delta epsilon zeta", "alpha beta gamma delta omega psi"])
    np.testing.assert_allclose(a, b, atol=1e-6)
    c = enc.encode(["omega beta gamma delta epsilon zeta"])[0]
    assert not np.allclose(a, c, atol=1e-4)


def test_instruction_only_on_queries(encoder: Encoder) -> None:
    q = Query("q1", "alpha beta")
    np.testing.assert_allclose(
        encoder.encode_queries([q])[0],
        encoder.encode([TINY_PARAMS.query_instruction + "alpha beta"])[0],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        encoder.encode_documents([DOCS[0]])[0], encoder.encode(["Alpha\nbeta gamma"])[0], atol=1e-6
    )


def test_batch_size_does_not_change_embeddings_beyond_tolerance(encoder: Encoder) -> None:
    texts = [canonical_document_text(d) for d in DOCS]
    np.testing.assert_allclose(
        encoder.encode(texts, batch_size=1), encoder.encode(texts, batch_size=4), atol=1e-5
    )


def test_encoder_refuses_tampered_model(tmp_path: Path) -> None:
    spec = make_tiny_model(tmp_path)
    (tmp_path / spec.key / "vocab.txt").write_text("[PAD]\n", encoding="utf-8")
    with pytest.raises(VerificationError, match=r"vocab\.txt"):
        Encoder(spec, tmp_path, TINY_PARAMS, device="cpu")


def test_encoder_refuses_revision_mismatch(tiny: tuple[ModelSpec, Path]) -> None:
    spec, models_dir = tiny
    with pytest.raises(VerificationError, match="revision"):
        Encoder(spec, models_dir, params(revision="f" * 40), device="cpu")


# --------------------------------------------------------------------------- cache


def _built(encoder: Encoder, tmp_path: Path) -> Path:
    DenseIndex.build(encoder, DOCS, corpus_sha256="corpus").save(tmp_path)
    return tmp_path


def test_cache_roundtrip(encoder: Encoder, tmp_path: Path) -> None:
    index_dir = _built(encoder, tmp_path)
    loaded = DenseIndex.load(index_dir, TINY_PARAMS, "corpus", encoder.model_file_sha256)
    np.testing.assert_array_equal(loaded.embeddings, encoder.encode_documents(DOCS))
    manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["file_sha256"]) == {"embeddings.npy", "doc_ids.json"}
    assert manifest["document_tokens"]["n_truncated"] == 0
    assert manifest["encoder"]["numerics"]["cuda_matmul_allow_tf32"] is False


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda d: (d / "embeddings.npy").write_bytes(
                (d / "embeddings.npy").read_bytes()[:-4] + b"\0\0\0\0"
            ),
            "embeddings.npy",
        ),
        (lambda d: (d / "doc_ids.json").write_text('["d1", "d2", "d3", "d0"]'), "doc_ids.json"),
        (lambda d: (d / "extra.bin").write_bytes(b"x"), "unexpected files"),
        (lambda d: (d / "embeddings.npy").unlink(), "unexpected files"),
    ],
    ids=["embeddings-bytes", "doc-ids-reordered", "extra-file", "missing-file"],
)
def test_cache_refuses_tampering(encoder: Encoder, tmp_path: Path, mutate, message: str) -> None:  # type: ignore[no-untyped-def]
    index_dir = _built(encoder, tmp_path)
    mutate(index_dir)
    with pytest.raises(VerificationError, match=message):
        DenseIndex.load(index_dir, TINY_PARAMS, "corpus", encoder.model_file_sha256)


def test_cache_refuses_other_params_corpus_or_model(encoder: Encoder, tmp_path: Path) -> None:
    index_dir = _built(encoder, tmp_path)
    h = encoder.model_file_sha256
    with pytest.raises(VerificationError, match="params"):
        DenseIndex.load(index_dir, params(batch_size=8), "corpus", h)
    with pytest.raises(VerificationError, match="corpus"):
        DenseIndex.load(index_dir, TINY_PARAMS, "other", h)
    with pytest.raises(VerificationError, match="model files"):
        DenseIndex.load(index_dir, TINY_PARAMS, "corpus", {**h, "vocab.txt": "0" * 64})


def test_cache_refuses_non_unit_vectors_even_with_consistent_hashes(
    encoder: Encoder, tmp_path: Path
) -> None:
    index = DenseIndex.build(encoder, DOCS, "corpus")
    index.embeddings = index.embeddings * 2.0
    index.save(tmp_path)  # hashes recomputed, so only the norm check can catch it
    with pytest.raises(VerificationError, match="unit vectors"):
        DenseIndex.load(tmp_path, TINY_PARAMS, "corpus", encoder.model_file_sha256)


# --------------------------------------------------------------------------- search


def _index(emb: list[list[float]], ids: list[str]) -> DenseIndex:
    arr = np.asarray(emb, dtype=np.float32)
    return DenseIndex(arr, ids, {})


def test_exact_search_matches_brute_force(encoder: Encoder) -> None:
    index = DenseIndex.build(encoder, DOCS, "corpus")
    queries = [Query("q1", "alpha beta"), Query("q2", "theta kappa")]
    q_emb = encoder.encode_queries(queries)
    run = index.search_embeddings(queries, q_emb, depth=4)
    for i, q in enumerate(queries):
        brute = q_emb[i].astype(np.float64) @ index.embeddings.astype(np.float64).T
        expected = sorted(zip(index.doc_ids, brute, strict=True), key=lambda t: (-t[1], t[0]))
        assert [h.doc_id for h in run[q.query_id]] == [d for d, _ in expected]
        np.testing.assert_allclose([h.score for h in run[q.query_id]], [s for _, s in expected])


def test_dense_ties_break_by_doc_id_and_depth_truncates() -> None:
    s = 1 / np.sqrt(2)
    index = _index([[1, 0], [s, s], [1, 0], [0, 1]], ["b", "c", "a", "d"])
    run = index.search_embeddings([Query("q", "x")], np.asarray([[1.0, 0.0]], np.float32), 3)
    assert [h.doc_id for h in run["q"]] == ["a", "b", "c"]
    assert run["q"][0].score == run["q"][1].score == 1.0


def test_query_embedding_shape_checked() -> None:
    index = _index([[1, 0]], ["a"])
    with pytest.raises(VerificationError):
        index.search_embeddings([Query("q", "x")], np.zeros((1, 3), np.float32), 1)


def test_latency_protocol_with_fake_clock(encoder: Encoder) -> None:
    index = DenseIndex.build(encoder, DOCS, "corpus")
    queries = [Query(f"q{i}", "alpha") for i in range(4)]
    ticks = iter(range(0, 1000, 1))  # every clock() call advances 1 s
    result = measure_latency(encoder, index, queries, 2, warmup=2, clock=lambda: next(ticks))
    assert result["n"] == 4 and result["warmup"] == 2
    assert result["p50_ms"] == result["mean_ms"] == 1000.0
    assert result["single_query_embeddings"].shape == (4, encoder.dim)


@pytest.mark.cuda
def test_cuda_matches_cpu_within_tolerance(tiny: tuple[ModelSpec, Path]) -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    spec, models_dir = tiny
    texts = [canonical_document_text(d) for d in DOCS]
    cpu = Encoder(spec, models_dir, TINY_PARAMS, device="cpu").encode(texts)
    gpu = Encoder(spec, models_dir, TINY_PARAMS, device="cuda").encode(texts)
    np.testing.assert_allclose(cpu, gpu, atol=1e-5)


def test_models_registry_pins_approved_revisions() -> None:
    assert models.BGE_SMALL.revision == "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
    assert models.MINILM_CE.revision == "233902d25c440f23af6f7d6e94d2946bac0bee0a"
    assert "model.safetensors" in models.BGE_SMALL.files
    with pytest.raises(VerificationError, match="unknown model"):
        models.get_model_spec("nope")


def test_shipped_dense_configs_use_approved_settings() -> None:
    from groundlab.config import load_config

    repo = Path(__file__).resolve().parents[1]
    for name in ("dense_scifact_evidence", "dense_scifact_beir", "dense_nfcorpus"):
        cfg = load_config(repo / "configs" / f"{name}.toml")
        assert cfg.dense is not None
        d: DenseParams = cfg.dense
        assert d.query_instruction == "Represent this sentence for searching relevant passages: "
        assert (d.document_separator, d.max_length, d.truncation_side) == ("\n", 512, "right")
        assert (d.model, d.revision) == (models.BGE_SMALL.key, models.BGE_SMALL.revision)
