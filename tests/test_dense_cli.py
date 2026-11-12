"""End-to-end dense CLI on synthetic data with the tiny local model (needs CUDA)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from groundlab import cli, models
from groundlab.errors import EXIT_CUDA_UNAVAILABLE, EXIT_REFUSED, EXIT_VERIFICATION_FAILED
from tests.dense_fixtures import TINY_KEY, TINY_REVISION, make_tiny_model

CONFIG = f"""\
[experiment]
name = "synthetic_dense"
description = "integration test"

[dataset]
name = "scifact"
view = "evidence"

[retriever]
kind = "dense"
model = "{TINY_KEY}"
revision = "{TINY_REVISION}"
pooling = "cls"
normalize = true
max_length = 32
truncation_side = "right"
query_instruction = "Represent this sentence for searching relevant passages: "
document_instruction = ""
document_separator = "\\n"
attn_implementation = "eager"
dtype = "float32"
device = "cuda"
batch_size = 2

[run]
depth = 3
latency_warmup = 1

[evaluation]
metrics = ["ndcg@10", "recall@10", "mrr@10"]
"""


@pytest.fixture
def workdir(tmp_path: Path, prepared_data_dir: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    spec = make_tiny_model(tmp_path / "models")
    monkeypatch.setitem(models.MODELS, TINY_KEY, spec)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "cfg.toml").write_text(CONFIG, encoding="utf-8")
    return tmp_path


COMMON = ["--config", "cfg.toml", "--data-dir", "data"]
MODELS_ARG = ["--models-dir", "models"]


def _cuda() -> bool:
    torch = pytest.importorskip("torch")
    return bool(torch.cuda.is_available())


@pytest.mark.cuda
def test_dense_dev_pipeline_end_to_end(workdir: Path) -> None:
    if not _cuda():
        pytest.skip("no CUDA device")
    assert cli.main(["build", *COMMON, *MODELS_ARG]) == 0
    assert cli.main(["run", *COMMON, *MODELS_ARG, "--split", "dev"]) == 0
    assert cli.main(["evaluate", *COMMON, "--split", "dev"]) == 0
    run_dir = workdir / "outputs" / "runs" / "synthetic_dense" / "dev"
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["device"] == "cuda"
    assert manifest["latency"]["n"] == 2 and manifest["latency"]["warmup"] == 1
    assert manifest["query_encoding"]["gpu_memory"]["peak_allocated_bytes"] > 0
    assert manifest["index"]["build_gpu_memory"]["peak_allocated_bytes"] > 0
    assert set(manifest["index"]["file_sha256"]) == {"embeddings.npy", "doc_ids.json"}
    assert manifest["encoder"]["numerics"]["cuda_matmul_allow_tf32"] is False
    # Every query gets exactly `depth` documents (dense scores every document).
    lines = (run_dir / "run.trec").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2 * 3
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["n_queries"] == 2
    assert all(v == 0.0 for v in metrics["pytrec_eval_cross_check_max_abs_diff"].values())


@pytest.mark.cuda
def test_dense_test_split_refused(workdir: Path) -> None:
    if not _cuda():
        pytest.skip("no CUDA device")
    assert cli.main(["build", *COMMON, *MODELS_ARG]) == 0
    assert cli.main(["run", *COMMON, *MODELS_ARG, "--split", "test"]) == EXIT_REFUSED
    assert cli.main(["evaluate", *COMMON, "--split", "test"]) == EXIT_REFUSED


@pytest.mark.cuda
def test_dense_run_refuses_tampered_cache(workdir: Path) -> None:
    if not _cuda():
        pytest.skip("no CUDA device")
    assert cli.main(["build", *COMMON, *MODELS_ARG]) == 0
    ids = workdir / "artifacts" / "indexes" / "synthetic_dense" / "doc_ids.json"
    values = json.loads(ids.read_text(encoding="utf-8"))
    ids.write_text(json.dumps(values[::-1]), encoding="utf-8")
    rc = cli.main(["run", *COMMON, *MODELS_ARG, "--split", "dev"])
    assert rc == EXIT_VERIFICATION_FAILED


def test_dense_without_cuda_fails_with_exit_5(
    workdir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert cli.main(["build", *COMMON, *MODELS_ARG]) == EXIT_CUDA_UNAVAILABLE
    assert cli.main(["run", *COMMON, *MODELS_ARG, "--split", "dev"]) == EXIT_CUDA_UNAVAILABLE
    assert not (workdir / "artifacts" / "indexes" / "synthetic_dense").exists()


def test_models_fetch_verify_only_detects_tampering(workdir: Path) -> None:
    args = ["models", "fetch", "--model", TINY_KEY, "--models-dir", "models", "--verify-only"]
    assert cli.main(args) == 0
    (workdir / "models" / TINY_KEY / "config.json").write_text("{}", encoding="utf-8")
    assert cli.main(args) == EXIT_VERIFICATION_FAILED
