from __future__ import annotations

import json
from pathlib import Path

import pytest

from groundlab.data.loader import ProcessedDataset
from groundlab.data.prepare import prepare_dataset
from groundlab.errors import VerificationError
from tests.conftest import SyntheticSciFact


def test_prepare_builds_views_and_passes_audit(tmp_path: Path, synthetic: SyntheticSciFact) -> None:
    data_dir = tmp_path / "data"
    result = prepare_dataset(synthetic.spec(), data_dir, downloader=synthetic.downloader)
    assert result.passed, result.audit["differences"]
    assert len(synthetic.downloads) == 2
    ds = ProcessedDataset(data_dir, "scifact")
    assert [q.query_id for q in ds.queries("evidence", "dev")] == ["1", "3"]
    assert ds.qrels("evidence", "dev") == {"1": {"10": 1}, "3": {"20": 1, "30": 1}}
    # BEIR view keeps the no-evidence claim 2.
    assert [q.query_id for q in ds.queries("beir", "dev")] == ["1", "2", "3"]
    assert [d.doc_id for d in ds.corpus()] == ["10", "20", "30", "40"]
    # BEIR text is canonical (double space kept, not replaced by the Ai2 abstract).
    assert ds.corpus()[3].text == "omega  psi."


def test_gold_metadata_is_dropped(prepared_data_dir: Path) -> None:
    root = prepared_data_dir / "processed" / "scifact" / "views"
    for path in root.rglob("queries.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            assert set(row) == {"query_id", "text"}, path
        assert "SUPPORT" not in path.read_text(encoding="utf-8")


def test_second_prepare_does_not_redownload(
    prepared_data_dir: Path, synthetic: SyntheticSciFact
) -> None:
    n = len(synthetic.downloads)
    assert prepare_dataset(
        synthetic.spec(), prepared_data_dir, downloader=synthetic.downloader
    ).passed
    assert len(synthetic.downloads) == n


def test_count_mismatch_fails_and_blocks_loading(
    tmp_path: Path, synthetic: SyntheticSciFact
) -> None:
    data_dir = tmp_path / "data"
    spec = synthetic.spec(**{"view.evidence.test.queries": 188})
    result = prepare_dataset(spec, data_dir, downloader=synthetic.downloader)
    assert not result.passed
    assert result.audit["differences"] == [
        {"key": "view.evidence.test.queries", "expected": 188, "measured": 1}
    ]
    assert not (data_dir / "processed" / "scifact" / "corpus.jsonl").exists()
    with pytest.raises(VerificationError, match="audit status"):
        ProcessedDataset(data_dir, "scifact")


def test_unexpected_measured_key_fails(tmp_path: Path, synthetic: SyntheticSciFact) -> None:
    spec = synthetic.spec()
    del spec.expected["map.title_exact_matches"]
    result = prepare_dataset(spec, tmp_path / "d", downloader=synthetic.downloader)
    assert not result.passed


def test_tampered_archive_is_rejected(tmp_path: Path, synthetic: SyntheticSciFact) -> None:
    data_dir = tmp_path / "data"
    raw = data_dir / "raw"
    raw.mkdir(parents=True)
    blob = bytearray(synthetic.blobs["beir_scifact"])
    blob[-1] ^= 0xFF  # same size, different bytes
    (raw / "scifact.zip").write_bytes(bytes(blob))
    with pytest.raises(VerificationError, match="SHA-256"):
        prepare_dataset(synthetic.spec(), data_dir, downloader=synthetic.downloader)


def test_verify_only_never_downloads(tmp_path: Path, synthetic: SyntheticSciFact) -> None:
    with pytest.raises(VerificationError, match="archive missing"):
        prepare_dataset(
            synthetic.spec(), tmp_path / "d", verify_only=True, downloader=synthetic.downloader
        )
    assert synthetic.downloads == []


def test_verify_only_detects_modified_processed_file(
    prepared_data_dir: Path, synthetic: SyntheticSciFact
) -> None:
    ok = prepare_dataset(synthetic.spec(), prepared_data_dir, verify_only=True)
    assert ok.passed
    qrels = prepared_data_dir / "processed/scifact/views/evidence/dev/qrels.tsv"
    qrels.write_text(qrels.read_text(encoding="utf-8") + "1\t40\t1\n", encoding="utf-8")
    bad = prepare_dataset(synthetic.spec(), prepared_data_dir, verify_only=True)
    assert not bad.passed
    assert bad.audit["differences"][0]["key"] == "file:views/evidence/dev/qrels.tsv"
    # The loader's own manifest check also refuses the modified file.
    with pytest.raises(VerificationError, match="differs from manifest"):
        ProcessedDataset(prepared_data_dir, "scifact").qrels("evidence", "dev")
