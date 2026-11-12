"""Turn verified source archives into the processed views used by every experiment.

Processed layout (under <data-dir>/processed/<dataset>/):
    corpus.jsonl                         {"doc_id", "title", "text"}  (BEIR text is canonical)
    views/<view>/<split>/queries.jsonl   {"query_id", "text"}         (no metadata)
    views/<view>/<split>/qrels.tsv       query_id, doc_id, grade
    manifest.json                        SHA-256 of every processed file
    audit.json                           measured vs expected counts

Split mapping:
    scifact  evidence  dev  = Ai2 claims_train, evidence-bearing claims only
    scifact  evidence  test = Ai2 claims_dev,   evidence-bearing claims only (primary, 188)
    scifact  beir      dev  = BEIR qrels/train.tsv
    scifact  beir      test = BEIR qrels/test.tsv
    nfcorpus beir      dev  = BEIR qrels/dev.tsv
    nfcorpus beir      test = BEIR qrels/test.tsv
"""

from __future__ import annotations

import io
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from groundlab import __version__
from groundlab.data.archives import Downloader, ensure_archive, http_download, read_members
from groundlab.data.specs import DatasetSpec
from groundlab.errors import VerificationError
from groundlab.io_utils import (
    atomic_write_bytes,
    dumps_json,
    dumps_jsonl,
    iter_jsonl_lines,
    sha256_bytes,
    sha256_file,
)

QrelRow = tuple[str, str, int]


@dataclass
class Prepared:
    files: dict[str, bytes] = field(default_factory=dict)
    measured: dict[str, Any] = field(default_factory=dict)


@dataclass
class PrepareResult:
    dataset: str
    processed_dir: Path
    audit: dict[str, Any]

    @property
    def passed(self) -> bool:
        return bool(self.audit["status"] == "pass")


# --------------------------------------------------------------------------- parsing


def _jsonl(data: bytes, source: str) -> list[dict[str, Any]]:
    return list(iter_jsonl_lines(io.StringIO(data.decode("utf-8")), source))


def parse_beir_qrels(data: bytes, source: str) -> list[QrelRow]:
    lines = data.decode("utf-8").splitlines()
    if not lines or lines[0].split("\t") != ["query-id", "corpus-id", "score"]:
        raise VerificationError(f"{source}: unexpected qrels header {lines[:1]!r}")
    rows: list[QrelRow] = []
    seen: set[tuple[str, str]] = set()
    for lineno, line in enumerate(lines[1:], start=2):
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) != 3:
            raise VerificationError(f"{source}:{lineno}: expected 3 tab-separated fields")
        qid, did, score = parts[0], parts[1], int(parts[2])
        if (qid, did) in seen:
            raise VerificationError(f"{source}:{lineno}: duplicate pair ({qid}, {did})")
        seen.add((qid, did))
        rows.append((qid, did, score))
    return rows


def _beir_corpus(data: bytes, source: str) -> list[dict[str, str]]:
    docs: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in _jsonl(data, source):
        doc_id = str(row["_id"])
        if doc_id in seen:
            raise VerificationError(f"{source}: duplicate doc id {doc_id}")
        seen.add(doc_id)
        docs.append({"doc_id": doc_id, "title": str(row["title"]), "text": str(row["text"])})
    return docs


def _beir_queries(data: bytes, source: str) -> tuple[list[dict[str, str]], int]:
    """Return (queries with ONLY id and text, number of rows with non-empty metadata).

    BEIR SciFact `metadata` is a copy of the Ai2 gold evidence map. It is dropped here
    so it can never reach a retriever.
    """
    queries: list[dict[str, str]] = []
    seen: set[str] = set()
    with_metadata = 0
    for row in _jsonl(data, source):
        qid = str(row["_id"])
        if qid in seen:
            raise VerificationError(f"{source}: duplicate query id {qid}")
        seen.add(qid)
        if row.get("metadata"):
            with_metadata += 1
        queries.append({"query_id": qid, "text": str(row["text"])})
    return queries, with_metadata


def _qrels_tsv(rows: list[QrelRow]) -> bytes:
    lines = ["query_id\tdoc_id\tgrade"]
    lines += [f"{q}\t{d}\t{g}" for q, d, g in sorted(rows, key=lambda r: (r[0], r[1]))]
    return ("\n".join(lines) + "\n").encode("utf-8")


def _add_view(
    out: Prepared,
    view: str,
    split: str,
    queries_by_id: dict[str, dict[str, str]],
    query_order: list[str],
    rows: list[QrelRow],
) -> None:
    qids_with_qrels = {q for q, _, g in rows if g > 0}
    missing = sorted(qids_with_qrels - queries_by_id.keys())
    if missing:
        raise VerificationError(f"{view}/{split}: qrels reference unknown queries {missing[:5]}")
    queries = [queries_by_id[q] for q in query_order if q in qids_with_qrels]
    base = f"views/{view}/{split}"
    out.files[f"{base}/queries.jsonl"] = dumps_jsonl(queries)
    out.files[f"{base}/qrels.tsv"] = _qrels_tsv(rows)
    out.measured[f"view.{view}.{split}.queries"] = len(queries)
    out.measured[f"view.{view}.{split}.pairs"] = len(rows)


def _structural_checks(out: Prepared, views: tuple[str, ...]) -> None:
    fields: set[str] = set()
    overlap = 0
    for view in views:
        split_ids: dict[str, set[str]] = {}
        for split in ("dev", "test"):
            rows = _jsonl(out.files[f"views/{view}/{split}/queries.jsonl"], "processed")
            for row in rows:
                fields.update(row.keys())
            split_ids[split] = {r["query_id"] for r in rows}
        overlap += len(split_ids["dev"] & split_ids["test"])
    out.measured["processed.queries_fields"] = sorted(fields)
    out.measured["processed.split_query_overlap"] = overlap


# --------------------------------------------------------------------------- datasets


def build_scifact(members: dict[str, dict[str, bytes]]) -> Prepared:
    beir = members["beir_scifact"]
    ai2 = members["ai2_scifact"]
    out = Prepared()
    m = out.measured

    corpus = _beir_corpus(beir["scifact/corpus.jsonl"], "beir corpus")
    queries, with_metadata = _beir_queries(beir["scifact/queries.jsonl"], "beir queries")
    qrels_test = parse_beir_qrels(beir["scifact/qrels/test.tsv"], "beir qrels/test.tsv")
    qrels_train = parse_beir_qrels(beir["scifact/qrels/train.tsv"], "beir qrels/train.tsv")
    m["beir.corpus_rows"] = len(corpus)
    m["beir.queries_rows"] = len(queries)
    m["beir.queries_with_nonempty_metadata"] = with_metadata
    m["beir.qrels_test_rows"] = len(qrels_test)
    m["beir.qrels_test_qids"] = len({r[0] for r in qrels_test})
    m["beir.qrels_train_rows"] = len(qrels_train)
    m["beir.qrels_train_qids"] = len({r[0] for r in qrels_train})

    ai2_corpus = _jsonl(ai2["data/corpus.jsonl"], "ai2 corpus")
    claims = {
        split: _jsonl(ai2[f"data/claims_{split}.jsonl"], f"ai2 claims_{split}")
        for split in ("train", "dev", "test")
    }
    m["ai2.corpus_rows"] = len(ai2_corpus)
    for split, rows in claims.items():
        m[f"ai2.claims_{split}_rows"] = len(rows)

    # Mapping checks between the two releases.
    queries_by_id = {q["query_id"]: q for q in queries}
    ai2_ids = {split: {str(c["id"]) for c in rows} for split, rows in claims.items()}
    m["map.beir_test_qids_equal_ai2_dev_ids"] = {r[0] for r in qrels_test} == ai2_ids["dev"]
    m["map.beir_train_qids_equal_ai2_train_ids"] = {r[0] for r in qrels_train} == ai2_ids["train"]
    m["map.claim_text_exact_matches"] = sum(
        1
        for split in ("train", "dev")
        for c in claims[split]
        if str(c["id"]) in queries_by_id and queries_by_id[str(c["id"])]["text"] == c["claim"]
    )
    m["map.ai2_test_ids_in_beir_queries"] = len(ai2_ids["test"] & queries_by_id.keys())
    ai2_docs = {str(d["doc_id"]): d for d in ai2_corpus}
    m["map.corpus_ids_equal"] = {d["doc_id"] for d in corpus} == ai2_docs.keys()
    titles = space_join = ws_only = 0
    for doc in corpus:
        other = ai2_docs.get(doc["doc_id"])
        if other is None:
            continue
        titles += doc["title"] == other["title"]
        joined = " ".join(other["abstract"])
        if doc["text"] == joined:
            space_join += 1
        elif "".join(doc["text"].split()) == "".join(joined.split()):
            ws_only += 1
    m["map.title_exact_matches"] = titles
    m["map.text_space_join_matches"] = space_join
    m["map.text_whitespace_only_differences"] = ws_only

    out.files["corpus.jsonl"] = dumps_jsonl(corpus)

    # Evidence view: a doc is relevant iff it has >= 1 non-empty evidence set.
    # Stance is ignored; claims without evidence are excluded.
    for split, ai2_split in (("dev", "train"), ("test", "dev")):
        ev_rows: list[QrelRow] = []
        order: list[str] = []
        for claim in claims[ai2_split]:
            qid = str(claim["id"])
            relevant = sorted(
                {
                    str(doc_id)
                    for doc_id, sets in claim.get("evidence", {}).items()
                    if any(s.get("sentences") for s in sets)
                }
            )
            if relevant:
                order.append(qid)
                ev_rows.extend((qid, doc_id, 1) for doc_id in relevant)
        _add_view(out, "evidence", split, queries_by_id, order, ev_rows)

    query_order = [q["query_id"] for q in queries]
    _add_view(out, "beir", "dev", queries_by_id, query_order, qrels_train)
    _add_view(out, "beir", "test", queries_by_id, query_order, qrels_test)
    _structural_checks(out, ("evidence", "beir"))
    return out


def build_nfcorpus(members: dict[str, dict[str, bytes]]) -> Prepared:
    beir = members["beir_nfcorpus"]
    out = Prepared()
    m = out.measured
    corpus = _beir_corpus(beir["nfcorpus/corpus.jsonl"], "nfcorpus corpus")
    queries, _ = _beir_queries(beir["nfcorpus/queries.jsonl"], "nfcorpus queries")
    m["beir.corpus_rows"] = len(corpus)
    m["beir.queries_rows"] = len(queries)
    qrels: dict[str, list[QrelRow]] = {}
    for split in ("train", "dev", "test"):
        rows = parse_beir_qrels(beir[f"nfcorpus/qrels/{split}.tsv"], f"nfcorpus qrels/{split}")
        qrels[split] = rows
        m[f"beir.qrels_{split}_rows"] = len(rows)
        m[f"beir.qrels_{split}_qids"] = len({r[0] for r in rows})
        if split != "train":
            m[f"beir.qrels_{split}_grades"] = {
                str(g): n for g, n in sorted(Counter(r[2] for r in rows).items())
            }
    out.files["corpus.jsonl"] = dumps_jsonl(corpus)
    queries_by_id = {q["query_id"]: q for q in queries}
    order = [q["query_id"] for q in queries]
    _add_view(out, "beir", "dev", queries_by_id, order, qrels["dev"])
    _add_view(out, "beir", "test", queries_by_id, order, qrels["test"])
    _structural_checks(out, ("beir",))
    return out


BUILDERS: dict[str, Callable[[dict[str, dict[str, bytes]]], Prepared]] = {
    "scifact": build_scifact,
    "nfcorpus": build_nfcorpus,
}


# --------------------------------------------------------------------------- audit + IO


def compare(expected: dict[str, Any], measured: dict[str, Any]) -> list[dict[str, Any]]:
    differences = []
    for key in sorted(expected.keys() | measured.keys()):
        exp = expected.get(key, "<not expected>")
        got = measured.get(key, "<not measured>")
        if exp != got:
            differences.append({"key": key, "expected": exp, "measured": got})
    return differences


def processed_dir_for(data_dir: Path, dataset: str) -> Path:
    return data_dir / "processed" / dataset


def prepare_dataset(
    spec: DatasetSpec,
    data_dir: Path,
    *,
    verify_only: bool = False,
    downloader: Downloader = http_download,
    builder: Callable[[dict[str, dict[str, bytes]]], Prepared] | None = None,
) -> PrepareResult:
    raw_dir = data_dir / "raw"
    members: dict[str, dict[str, bytes]] = {}
    archive_hashes: dict[str, str] = {}
    for archive in spec.archives:
        path = ensure_archive(
            archive, raw_dir, allow_download=not verify_only, downloader=downloader
        )
        members[archive.key] = read_members(archive, path)
        archive_hashes[archive.key] = archive.sha256

    build = builder or BUILDERS[spec.name]
    prepared = build(members)
    differences = compare(spec.expected, prepared.measured)

    out_dir = processed_dir_for(data_dir, spec.name)
    file_hashes = {name: sha256_bytes(data) for name, data in sorted(prepared.files.items())}
    if verify_only:
        for name, digest in file_hashes.items():
            path = out_dir / name
            on_disk = sha256_file(path) if path.exists() else "<missing>"
            if on_disk != digest:
                differences.append({"key": f"file:{name}", "expected": digest, "measured": on_disk})

    audit = {
        "dataset": spec.name,
        "groundlab_version": __version__,
        "mode": "verify-only" if verify_only else "prepare",
        "archives_sha256_verified": archive_hashes,
        "members_sha256_verified": sorted(f"{a.key}:{n}" for a in spec.archives for n in a.members),
        "expected": spec.expected,
        "measured": prepared.measured,
        "differences": differences,
        "status": "pass" if not differences else "fail",
    }
    if not verify_only:
        if not differences:
            for name, data in prepared.files.items():
                atomic_write_bytes(out_dir / name, data)
            manifest = {"dataset": spec.name, "files": file_hashes, "archives": archive_hashes}
            atomic_write_bytes(out_dir / "manifest.json", dumps_json(manifest))
        # The audit is always written; loaders refuse a dataset whose audit failed.
        atomic_write_bytes(out_dir / "audit.json", dumps_json(audit))
    return PrepareResult(spec.name, out_dir, audit)
