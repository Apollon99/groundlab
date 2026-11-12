"""Pinned source archives and expected counts.

Provenance of every pinned value:
- Archive SHA-256/size and extracted-file SHA-256: Milestone 0 data audit
  (docs/DATA_AUDIT_M0.md summary, 2026-11-12), which matched the planning document
  and the BEIR README MD5s.
- Counts marked "M0" were measured in that audit.
- Counts marked "M1" were first measured by `groundlab data prepare` in Milestone 1
  and are not independently audited. See docs/DECISIONS.md (D-012).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ArchiveSpec:
    key: str
    url: str
    filename: str
    size: int
    sha256: str
    kind: str  # "zip" or "tar.gz"
    # Archive member path -> expected SHA-256 of its bytes.
    members: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    archives: tuple[ArchiveSpec, ...]
    # Flat dict of audit key -> expected value. Every key must be measured; every
    # measured key must be expected. Any difference fails `data prepare`.
    expected: dict[str, Any]


BEIR_SCIFACT = ArchiveSpec(
    key="beir_scifact",
    url="https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip",
    filename="scifact.zip",
    size=2_816_079,
    sha256="536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165",
    kind="zip",
    members={
        "scifact/corpus.jsonl": "dec31c8182f3d744c7d2c09423756fd1d17cbef75808db13ba01cc0aab4d1ac6",
        "scifact/queries.jsonl": "8ff84a7c903f722981cd8d595c022660140c51867b27608a6d4910db86080313",
        "scifact/qrels/test.tsv": "0864bb985e0ca2367ba217977e72004d549054b2b06666ed9d4825ac7c21284c",
        "scifact/qrels/train.tsv": "a53f2114831916c096b6c37d9e54da68cef4efdcdbd5ed46533601af972acf1d",
    },
)

AI2_SCIFACT = ArchiveSpec(
    key="ai2_scifact",
    # `release/latest` can change without notice: the SHA-256 is the only pin.
    url="https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz",
    filename="scifact_ai2_data.tar.gz",
    size=3_115_079,
    sha256="11c621288d41ac144d29b13b0f8503b3820b7d6e8b1f6ff24dff335c196d76be",
    kind="tar.gz",
    members={
        "data/corpus.jsonl": "b8d6c89624cb2ed74dee8938effc4f5d8bd2086887880af8110d64be4ceade62",
        "data/claims_train.jsonl": "f4c8fa82d8bd0653a9cc8d61a6ea48c25eacea64e90af5dbf390ebb1b74372f0",
        "data/claims_dev.jsonl": "86f0435d08fdb65d1aa41d1472684f57e6e71930626497bdf4d7a9ec1a632217",
        "data/claims_test.jsonl": "558930d75215c73f84a28fe538307d6d397c9de1ec7239514cf45f80d75d2ca3",
    },
)

BEIR_NFCORPUS = ArchiveSpec(
    key="beir_nfcorpus",
    url="https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip",
    filename="nfcorpus.zip",
    size=2_448_432,
    sha256="efe5be03f8c5b86a5870102d0599d227c8c6e2484328e68c6522560385671b0b",
    kind="zip",
    members={
        "nfcorpus/corpus.jsonl": "10cc83ef1826b1425e6a87090b5140b39b27755d5a27e48215a88611c899991f",
        "nfcorpus/queries.jsonl": "d024e6621b84925d485ae473d316a0c3af31c62c8068a59fb29d22f7613aef2a",
        "nfcorpus/qrels/test.tsv": "f8fba6ef3d4dd9c3a242a8ba4ae38276fc3622fce7dcbae764766d564542fd2a",
        "nfcorpus/qrels/dev.tsv": "b1d38b5e8f78c4a5820bce2b7ec2db54911d7690dc601e76811846b211180bd8",
        "nfcorpus/qrels/train.tsv": "6336b80f9bffc4f063f3aa450047ad35c0b7c534efe4a6ba35e16dbace047f6a",
    },
)

SCIFACT = DatasetSpec(
    name="scifact",
    archives=(BEIR_SCIFACT, AI2_SCIFACT),
    expected={
        # M0
        "beir.corpus_rows": 5183,
        "beir.queries_rows": 1109,
        "beir.queries_with_nonempty_metadata": 693,
        "beir.qrels_test_rows": 339,
        "beir.qrels_test_qids": 300,
        "beir.qrels_train_rows": 919,
        "beir.qrels_train_qids": 809,
        "ai2.corpus_rows": 5183,
        "ai2.claims_train_rows": 809,
        "ai2.claims_dev_rows": 300,
        "ai2.claims_test_rows": 300,
        "map.beir_test_qids_equal_ai2_dev_ids": True,
        "map.beir_train_qids_equal_ai2_train_ids": True,
        "map.claim_text_exact_matches": 1109,
        "map.ai2_test_ids_in_beir_queries": 0,
        "map.corpus_ids_equal": True,
        "map.title_exact_matches": 5183,
        "map.text_space_join_matches": 4128,
        "map.text_whitespace_only_differences": 1055,
        "view.evidence.test.queries": 188,
        "view.evidence.test.pairs": 209,
        "view.beir.test.queries": 300,
        "view.beir.test.pairs": 339,
        "view.beir.dev.queries": 809,
        "view.beir.dev.pairs": 919,
        # M1: measured by this code and by an independent stdlib recount on
        # 2026-11-12 (both 505/564); not part of the M0 audit.
        "view.evidence.dev.queries": 505,
        "view.evidence.dev.pairs": 564,
        # Structural checks (must always hold)
        "processed.queries_fields": ["query_id", "text"],
        "processed.split_query_overlap": 0,
    },
)

NFCORPUS = DatasetSpec(
    name="nfcorpus",
    archives=(BEIR_NFCORPUS,),
    expected={
        # M0
        "beir.corpus_rows": 3633,
        "beir.queries_rows": 3237,
        "beir.qrels_test_rows": 12334,
        "beir.qrels_test_qids": 323,
        "beir.qrels_test_grades": {"1": 11758, "2": 576},
        "beir.qrels_dev_rows": 11385,
        "beir.qrels_dev_qids": 324,
        "beir.qrels_dev_grades": {"1": 10864, "2": 521},
        "beir.qrels_train_rows": 110575,
        "beir.qrels_train_qids": 2590,
        "view.beir.test.queries": 323,
        "view.beir.test.pairs": 12334,
        "view.beir.dev.queries": 324,
        "view.beir.dev.pairs": 11385,
        "processed.queries_fields": ["query_id", "text"],
        "processed.split_query_overlap": 0,
    },
)

DATASETS: dict[str, DatasetSpec] = {SCIFACT.name: SCIFACT, NFCORPUS.name: NFCORPUS}

# Which views each dataset exposes. The first one is the primary view.
VIEWS: dict[str, tuple[str, ...]] = {"scifact": ("evidence", "beir"), "nfcorpus": ("beir",)}
SPLITS: tuple[str, ...] = ("dev", "test")
