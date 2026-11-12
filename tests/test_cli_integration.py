"""End-to-end CLI pipeline on the synthetic dataset: build -> run (dev) -> evaluate (dev)."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from groundlab import cli
from groundlab.data.loader import ProcessedDataset
from groundlab.errors import EXIT_REFUSED, EXIT_USAGE, EXIT_VERIFICATION_FAILED
from groundlab.io_utils import sha256_file

CONFIG = """\
[experiment]
name = "synthetic_bm25"
description = "integration test"

[dataset]
name = "scifact"
view = "evidence"

[retriever]
kind = "bm25"
method = "lucene"
k1 = 1.2
b = 0.75
tokenizer = "regex_casefold_snowball_en"

[run]
depth = 10

[evaluation]
metrics = ["ndcg@10", "recall@10", "mrr@10"]
"""


@pytest.fixture
def workdir(tmp_path: Path, prepared_data_dir: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "cfg.toml").write_text(CONFIG, encoding="utf-8")
    return tmp_path


def test_dev_pipeline_end_to_end(workdir: Path) -> None:
    common = ["--config", "cfg.toml", "--data-dir", "data"]
    assert cli.main(["build", *common]) == 0
    assert cli.main(["run", *common, "--split", "dev"]) == 0
    run_dir = workdir / "outputs" / "runs" / "synthetic_bm25" / "dev"
    lines = (run_dir / "run.trec").read_text(encoding="utf-8").splitlines()
    # Claim 3 "delta": docs 20 and 30 tie exactly -> doc ID ascending.
    q3 = [line.split() for line in lines if line.startswith("3 ")]
    assert [(p[2], p[3]) for p in q3] == [("20", "1"), ("30", "2")]
    assert q3[0][4] == q3[1][4]
    assert cli.main(["evaluate", *common, "--split", "dev"]) == 0
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    # Claim 1 retrieves doc 10 at rank 1; claim 3 retrieves both relevant docs at ranks 1-2.
    assert metrics["mean"] == {"ndcg@10": 1.0, "recall@10": 1.0, "mrr@10": 1.0}
    assert metrics["n_queries"] == 2
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["split"] == "dev" and manifest["environment"]["python"]["version"]


def test_test_split_commands_refused(workdir: Path) -> None:
    common = ["--config", "cfg.toml", "--data-dir", "data"]
    assert cli.main(["build", *common]) == 0
    assert cli.main(["run", *common, "--split", "test"]) == EXIT_REFUSED
    assert cli.main(["evaluate", *common, "--split", "test"]) == EXIT_REFUSED
    assert not (workdir / "outputs" / "runs" / "synthetic_bm25" / "test").exists()


def test_evaluate_rejects_run_from_other_config(workdir: Path) -> None:
    common = ["--config", "cfg.toml", "--data-dir", "data"]
    assert cli.main(["build", *common]) == 0
    assert cli.main(["run", *common, "--split", "dev"]) == 0
    (workdir / "cfg.toml").write_text(CONFIG.replace("depth = 10", "depth = 5"), encoding="utf-8")
    assert cli.main(["evaluate", *common, "--split", "dev"]) == EXIT_VERIFICATION_FAILED


def test_run_rejects_reordered_doc_ids(workdir: Path) -> None:
    # Regression: same-length permutation of doc_ids.json must make `run` fail.
    common = ["--config", "cfg.toml", "--data-dir", "data"]
    assert cli.main(["build", *common]) == 0
    path = workdir / "artifacts" / "indexes" / "synthetic_bm25" / "doc_ids.json"
    ids = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps(ids[1:] + ids[:1]), encoding="utf-8")
    assert cli.main(["run", *common, "--split", "dev"]) == EXIT_VERIFICATION_FAILED
    assert not (workdir / "outputs" / "runs" / "synthetic_bm25" / "dev" / "run.trec").exists()


def _rewrite_processed_file(data_dir: Path, rel: str, new_text: str) -> None:
    """Change a processed file AND regenerate its hash in the local data manifest,
    so the loader's own manifest check passes. Only the run manifest can catch this."""
    root = data_dir / "processed" / "scifact"
    (root / rel).write_bytes(new_text.encode("utf-8"))
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    manifest["files"][rel] = sha256_file(root / rel)
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


@pytest.mark.parametrize(
    ("rel", "edit"),
    [
        # Drop the judgment for doc 30 on claim 3.
        ("views/evidence/dev/qrels.tsv", lambda t: t.replace("3\t30\t1\n", "")),
        # Reword claim 1.
        ("views/evidence/dev/queries.jsonl", lambda t: t.replace("alpha beta", "alpha")),
    ],
    ids=["qrels", "queries"],
)
def test_evaluate_rejects_stale_run_after_input_and_manifest_update(
    workdir: Path, rel: str, edit: Callable[[str], str]
) -> None:
    common = ["--config", "cfg.toml", "--data-dir", "data"]
    assert cli.main(["build", *common]) == 0
    assert cli.main(["run", *common, "--split", "dev"]) == 0
    path = workdir / "data" / "processed" / "scifact" / rel
    original = path.read_text(encoding="utf-8")
    edited = edit(original)
    assert edited != original
    _rewrite_processed_file(workdir / "data", rel, edited)
    # Loader-level integrity still passes, so the old run is only caught by input hashes.
    ProcessedDataset(workdir / "data", "scifact").input_sha256("evidence", "dev")
    assert cli.main(["evaluate", *common, "--split", "dev"]) == EXIT_VERIFICATION_FAILED
    metrics = workdir / "outputs" / "runs" / "synthetic_bm25" / "dev" / "metrics.json"
    assert not metrics.exists()
    # A fresh run against the new inputs is accepted.
    assert cli.main(["run", *common, "--split", "dev"]) == 0
    assert cli.main(["evaluate", *common, "--split", "dev"]) == 0
    assert metrics.exists()


def test_evaluate_rejects_run_manifest_without_input_hashes(workdir: Path) -> None:
    common = ["--config", "cfg.toml", "--data-dir", "data"]
    assert cli.main(["build", *common]) == 0
    assert cli.main(["run", *common, "--split", "dev"]) == 0
    path = workdir / "outputs" / "runs" / "synthetic_bm25" / "dev" / "run_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    del manifest["input_sha256"]
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert cli.main(["evaluate", *common, "--split", "dev"]) == EXIT_VERIFICATION_FAILED


def test_bad_config_is_usage_error(workdir: Path) -> None:
    (workdir / "bad.toml").write_text(CONFIG + "\n[extra]\nx = 1\n", encoding="utf-8")
    assert cli.main(["build", "--config", "bad.toml", "--data-dir", "data"]) == EXIT_USAGE


def test_run_before_build_fails_cleanly(workdir: Path) -> None:
    rc = cli.main(["run", "--config", "cfg.toml", "--data-dir", "data", "--split", "dev"])
    assert rc == EXIT_VERIFICATION_FAILED
