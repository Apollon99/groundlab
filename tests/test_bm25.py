from __future__ import annotations

import inspect
import json
import math
from pathlib import Path

import pytest

from groundlab.errors import VerificationError
from groundlab.retrieval.bm25 import BM25Index, BM25Params
from groundlab.retrieval.tokenize import tokenize
from groundlab.types import Document, Query

PARAMS = BM25Params(method="lucene", k1=1.2, b=0.75)


def _lucene(
    tf: int, df: int, n: int, dl: int, avgdl: float, k1: float = 1.2, b: float = 0.75
) -> float:
    idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
    return idf * tf / (tf + k1 * (1 - b + b * dl / avgdl))


def test_tokenizer() -> None:
    assert tokenize("Running_jumps") == ["run", "jump"]  # underscore splits, stemmed
    assert tokenize("the a of") == ["the", "a", "of"]  # no stopword removal
    assert tokenize("STRASSE") == tokenize("straße")  # Unicode casefold
    assert tokenize("p53-induced, (IL-6)") == ["p53", "induc", "il", "6"]
    assert tokenize("") == []


def test_lucene_scores_hand_computed() -> None:
    docs = [Document("d0", "", "a b"), Document("d1", "", "b c c")]
    index = BM25Index.build(docs, PARAMS, corpus_sha256="x")
    scores = index.scores(Query("q", "c b"))
    # N=2, avgdl=2.5. d0 (len 2): only b (df=2, tf=1).
    # d1 (len 3): c (df=1, tf=2) + b (df=2, tf=1).
    d0 = _lucene(1, 2, 2, 2, 2.5)
    d1 = _lucene(2, 1, 2, 3, 2.5) + _lucene(1, 2, 2, 3, 2.5)
    assert d0 == pytest.approx(0.0902582, abs=1e-6)
    assert d1 == pytest.approx(0.4867520, abs=1e-6)
    assert scores[0] == pytest.approx(d0, rel=1e-6)
    assert scores[1] == pytest.approx(d1, rel=1e-6)
    run = index.search([Query("q", "c b")], depth=10)
    assert [h.doc_id for h in run["q"]] == ["d1", "d0"]


def test_ties_break_by_doc_id_and_nonmatching_docs_excluded() -> None:
    docs = [Document("b", "", "x y"), Document("a", "", "x y"), Document("c", "", "z")]
    run = BM25Index.build(docs, PARAMS, "x").search([Query("q", "x")], depth=10)
    assert [h.doc_id for h in run["q"]] == ["a", "b"]
    assert run["q"][0].score == run["q"][1].score


def test_repeated_query_terms_count_per_occurrence() -> None:
    # Pinned behaviour (D-008): "c c" scores exactly twice "c", as with Lucene's
    # BooleanQuery of repeated SHOULD clauses.
    index = BM25Index.build([Document("d0", "", "a b"), Document("d1", "", "b c c")], PARAMS, "x")
    once = index.scores(Query("q", "c"))
    twice = index.scores(Query("q", "c c"))
    assert twice[1] == pytest.approx(2 * once[1], rel=1e-6)


def test_oov_query_returns_empty_ranking() -> None:
    index = BM25Index.build([Document("a", "", "x")], PARAMS, "x")
    assert index.search([Query("q", "unseen words")], depth=5) == {"q": []}


def test_depth_truncates() -> None:
    docs = [Document(f"d{i}", "", "x " * (i + 1)) for i in range(5)]
    run = BM25Index.build(docs, PARAMS, "x").search([Query("q", "x")], depth=2)
    assert len(run["q"]) == 2


def test_save_load_roundtrip_and_staleness(tmp_path: Path) -> None:
    docs = [Document("d0", "t", "a b"), Document("d1", "t", "b c c")]
    index = BM25Index.build(docs, PARAMS, corpus_sha256="abc")
    index.save(tmp_path)
    loaded = BM25Index.load(tmp_path, PARAMS, "abc")
    queries = [Query("q", "c b")]
    assert loaded.search(queries, 10) == index.search(queries, 10)
    with pytest.raises(VerificationError, match="different BM25 params"):
        BM25Index.load(tmp_path, BM25Params(k1=0.9), "abc")
    with pytest.raises(VerificationError, match="different corpus"):
        BM25Index.load(tmp_path, PARAMS, "other")


def _saved_index(tmp_path: Path) -> Path:
    docs = [Document("d0", "t", "a b"), Document("d1", "t", "b c c"), Document("d2", "t", "c d")]
    BM25Index.build(docs, PARAMS, corpus_sha256="abc").save(tmp_path)
    return tmp_path


def test_manifest_records_hash_of_every_index_file(tmp_path: Path) -> None:
    index_dir = _saved_index(tmp_path)
    manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
    hashes = manifest["file_sha256"]
    on_disk = {
        p.relative_to(index_dir).as_posix()
        for p in index_dir.rglob("*")
        if p.is_file() and p.name != "manifest.json"
    }
    assert set(hashes) == on_disk
    assert "doc_ids.json" in hashes and any(k.startswith("bm25s/") for k in hashes)


def test_reordered_doc_ids_same_length_rejected(tmp_path: Path) -> None:
    # Regression: a permuted doc_ids.json keeps the length check happy but would silently
    # attach scores to the wrong documents. The hash check must catch it.
    index_dir = _saved_index(tmp_path)
    path = index_dir / "doc_ids.json"
    ids = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps(list(reversed(ids))), encoding="utf-8")
    assert len(json.loads(path.read_text(encoding="utf-8"))) == len(ids)
    with pytest.raises(VerificationError, match=r"doc_ids\.json"):
        BM25Index.load(index_dir, PARAMS, "abc")


def test_tampered_missing_or_extra_index_file_rejected(tmp_path: Path) -> None:
    index_dir = _saved_index(tmp_path)
    model_file = next(p for p in sorted((index_dir / "bm25s").iterdir()) if p.is_file())
    original = model_file.read_bytes()
    model_file.write_bytes(original + b" ")
    with pytest.raises(VerificationError, match="differ from manifest"):
        BM25Index.load(index_dir, PARAMS, "abc")
    model_file.write_bytes(original)
    (index_dir / "bm25s" / "extra.npy").write_bytes(b"x")
    with pytest.raises(VerificationError, match=r"extra\.npy"):
        BM25Index.load(index_dir, PARAMS, "abc")
    (index_dir / "bm25s" / "extra.npy").unlink()
    model_file.unlink()
    with pytest.raises(VerificationError, match="differ from manifest"):
        BM25Index.load(index_dir, PARAMS, "abc")


def test_manifest_without_file_hashes_rejected(tmp_path: Path) -> None:
    index_dir = _saved_index(tmp_path)
    manifest_path = index_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["file_sha256"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(VerificationError, match="no file hashes"):
        BM25Index.load(index_dir, PARAMS, "abc")


def test_rebuild_into_same_dir_leaves_no_stale_files(tmp_path: Path) -> None:
    index_dir = _saved_index(tmp_path)
    (index_dir / "bm25s" / "stale.npy").write_bytes(b"old")
    BM25Index.build([Document("x", "", "y")], PARAMS, "def").save(index_dir)
    assert not (index_dir / "bm25s" / "stale.npy").exists()
    BM25Index.load(index_dir, PARAMS, "def")


def test_retriever_interface_takes_only_queries() -> None:
    # Rankers must not be able to receive qrels: search() takes queries and depth only.
    assert list(inspect.signature(BM25Index.search).parameters) == ["self", "queries", "depth"]
