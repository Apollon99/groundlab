"""Synthetic SciFact-shaped archives. No real data is used anywhere in the test suite."""

from __future__ import annotations

import io
import json
import tarfile
import zipfile
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from groundlab.data.specs import AI2_SCIFACT, BEIR_SCIFACT, SCIFACT, ArchiveSpec, DatasetSpec
from groundlab.io_utils import sha256_bytes


def _jsonl(rows: list[dict[str, Any]]) -> bytes:
    return "".join(json.dumps(r) + "\n" for r in rows).encode()


def _ev(*doc_ids: int) -> dict[str, list[dict[str, Any]]]:
    return {str(d): [{"sentences": [0], "label": "SUPPORT"}] for d in doc_ids}


CORPUS = [  # doc_id, title, abstract sentences
    (10, "Alpha", ["alpha beta."]),
    (20, "Delta", ["delta epsilon."]),
    (30, "Delta", ["delta zeta."]),
    (40, "Omega", ["omega", "psi."]),
]
CLAIMS = {
    "train": [
        {"id": 1, "claim": "alpha beta", "evidence": _ev(10), "cited_doc_ids": [10]},
        {"id": 2, "claim": "gamma", "evidence": {}, "cited_doc_ids": [20]},
        {"id": 3, "claim": "delta", "evidence": _ev(20, 30), "cited_doc_ids": [20, 30]},
    ],
    "dev": [
        {"id": 4, "claim": "omega psi", "evidence": _ev(40), "cited_doc_ids": [40]},
        {"id": 5, "claim": "zeta", "evidence": {}, "cited_doc_ids": [30]},
    ],
    "test": [{"id": 6, "claim": "hidden test claim"}],
}

# Hand-derived counts for the synthetic data above.
EXPECTED: dict[str, Any] = {
    "beir.corpus_rows": 4,
    "beir.queries_rows": 5,
    "beir.queries_with_nonempty_metadata": 3,  # claims 1, 3, 4 have evidence
    "beir.qrels_test_rows": 2,
    "beir.qrels_test_qids": 2,
    "beir.qrels_train_rows": 4,
    "beir.qrels_train_qids": 3,
    "ai2.corpus_rows": 4,
    "ai2.claims_train_rows": 3,
    "ai2.claims_dev_rows": 2,
    "ai2.claims_test_rows": 1,
    "map.beir_test_qids_equal_ai2_dev_ids": True,
    "map.beir_train_qids_equal_ai2_train_ids": True,
    "map.claim_text_exact_matches": 5,
    "map.ai2_test_ids_in_beir_queries": 0,
    "map.corpus_ids_equal": True,
    "map.title_exact_matches": 4,
    "map.text_space_join_matches": 3,
    "map.text_whitespace_only_differences": 1,  # doc 40 has a double space in BEIR
    "view.evidence.dev.queries": 2,
    "view.evidence.dev.pairs": 3,
    "view.evidence.test.queries": 1,
    "view.evidence.test.pairs": 1,
    "view.beir.dev.queries": 3,
    "view.beir.dev.pairs": 4,
    "view.beir.test.queries": 2,
    "view.beir.test.pairs": 2,
    "processed.queries_fields": ["query_id", "text"],
    "processed.split_query_overlap": 0,
}


def _zip(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return buf.getvalue()


def _targz(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def build_members() -> tuple[dict[str, bytes], dict[str, bytes]]:
    beir_corpus = []
    for doc_id, title, abstract in CORPUS:
        text = "omega  psi." if doc_id == 40 else " ".join(abstract)
        beir_corpus.append({"_id": str(doc_id), "title": title, "text": text, "metadata": {}})
    queries = [
        {"_id": str(c["id"]), "text": c["claim"], "metadata": c["evidence"]}
        for split in ("train", "dev")
        for c in CLAIMS[split]
    ]
    qrels_train = "query-id\tcorpus-id\tscore\n1\t10\t1\n2\t20\t1\n3\t20\t1\n3\t30\t1\n"
    qrels_test = "query-id\tcorpus-id\tscore\n4\t40\t1\n5\t30\t1\n"
    beir = {
        "scifact/corpus.jsonl": _jsonl(beir_corpus),
        "scifact/queries.jsonl": _jsonl(queries),
        "scifact/qrels/train.tsv": qrels_train.encode(),
        "scifact/qrels/test.tsv": qrels_test.encode(),
    }
    ai2_corpus = [
        {"doc_id": d, "title": t, "abstract": a, "structured": False} for d, t, a in CORPUS
    ]
    ai2 = {
        "data/corpus.jsonl": _jsonl(ai2_corpus),
        "data/claims_train.jsonl": _jsonl(CLAIMS["train"]),
        "data/claims_dev.jsonl": _jsonl(CLAIMS["dev"]),
        "data/claims_test.jsonl": _jsonl(CLAIMS["test"]),
        "._data": b"appledouble junk",
    }
    return beir, ai2


def _spec_for(base: ArchiveSpec, blob: bytes, members: dict[str, bytes]) -> ArchiveSpec:
    return replace(
        base,
        url="synthetic://" + base.filename,
        size=len(blob),
        sha256=sha256_bytes(blob),
        members={n: sha256_bytes(members[n]) for n in base.members},
    )


class SyntheticSciFact:
    def __init__(self, raw_dir: Path) -> None:
        beir, ai2 = build_members()
        self.blobs = {BEIR_SCIFACT.key: _zip(beir), AI2_SCIFACT.key: _targz(ai2)}
        self.archives = (
            _spec_for(BEIR_SCIFACT, self.blobs[BEIR_SCIFACT.key], beir),
            _spec_for(AI2_SCIFACT, self.blobs[AI2_SCIFACT.key], ai2),
        )
        self.raw_dir = raw_dir
        self.downloads: list[str] = []

    def spec(self, **expected_overrides: Any) -> DatasetSpec:
        return replace(SCIFACT, archives=self.archives, expected={**EXPECTED, **expected_overrides})

    def downloader(self, url: str, dest: Path) -> None:
        self.downloads.append(url)
        key = next(a.key for a in self.archives if a.url == url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(self.blobs[key])


@pytest.fixture
def synthetic(tmp_path: Path) -> SyntheticSciFact:
    return SyntheticSciFact(tmp_path / "data" / "raw")


@pytest.fixture
def prepared_data_dir(tmp_path: Path, synthetic: SyntheticSciFact) -> Path:
    from groundlab.data.prepare import prepare_dataset

    data_dir = tmp_path / "data"
    result = prepare_dataset(synthetic.spec(), data_dir, downloader=synthetic.downloader)
    assert result.passed, result.audit["differences"]
    return data_dir
