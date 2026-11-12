"""Hybrid and hybrid+rerank through the CLI: configs, refusals, and an end-to-end synthetic run.

The end-to-end tests need CUDA (tiny random models, synthetic data) and are skipped, with a
reason, without it. Config, refusal and no-CUDA tests run everywhere.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from groundlab import cli, models
from groundlab.config import load_config
from groundlab.errors import (
    EXIT_CUDA_UNAVAILABLE,
    EXIT_REFUSED,
    EXIT_USAGE,
    EXIT_VERIFICATION_FAILED,
    ConfigError,
    SplitAccessError,
)
from groundlab.models import MINILM_CE
from groundlab.retrieval.hybrid import rrf_fuse
from groundlab.runio import read_run
from tests.dense_fixtures import (
    TINY_CE_KEY,
    TINY_KEY,
    TINY_REVISION,
    make_tiny_cross_encoder,
    make_tiny_model,
)
from tests.test_cli_integration import CONFIG as BM25_CONFIG
from tests.test_dense_cli import CONFIG as DENSE_CONFIG

REPO = Path(__file__).resolve().parents[1]
CONFIGS = REPO / "configs"
VIEWS = ("scifact_evidence", "scifact_beir", "nfcorpus")
FUSED_FILES = [f"{arm}_{v}.toml" for arm in ("hybrid", "rerank") for v in VIEWS]

HYBRID_CONFIG = """\
[experiment]
name = "synthetic_hybrid"
description = "integration test"

[dataset]
name = "scifact"
view = "evidence"

[retriever]
kind = "hybrid"
fusion = "rrf"
rrf_k = 60
parent_depth = 100
bm25_config = "bm25.toml"
dense_config = "dense.toml"

[run]
depth = 100
latency_warmup = 1

[evaluation]
metrics = ["ndcg@10", "recall@10", "mrr@10"]
"""

RERANK_CONFIG = (
    HYBRID_CONFIG.replace("synthetic_hybrid", "synthetic_rerank")
    .replace('kind = "hybrid"', 'kind = "rerank"')
    .replace(
        'dense_config = "dense.toml"\n',
        'dense_config = "dense.toml"\n'
        f'model = "{TINY_CE_KEY}"\n'
        f'revision = "{TINY_REVISION}"\n'
        "max_length = 24\nrerank_depth = 3\nbatch_size = 2\n"
        'attn_implementation = "eager"\ndtype = "float32"\ndevice = "cuda"\n',
    )
)

COMMON = ["--data-dir", "data"]
MODELS_ARG = ["--models-dir", "models"]


def _cuda() -> bool:
    torch = pytest.importorskip("torch")
    return bool(torch.cuda.is_available())


@pytest.fixture
def workdir(tmp_path: Path, prepared_data_dir: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setitem(models.MODELS, TINY_KEY, make_tiny_model(tmp_path / "models"))
    monkeypatch.setitem(models.MODELS, TINY_CE_KEY, make_tiny_cross_encoder(tmp_path / "models"))
    monkeypatch.chdir(tmp_path)
    files = {
        "bm25.toml": BM25_CONFIG.replace("depth = 10", "depth = 100"),
        "dense.toml": DENSE_CONFIG.replace("depth = 3", "depth = 100"),
        "hybrid.toml": HYBRID_CONFIG,
        "rerank.toml": RERANK_CONFIG,
    }
    for name, text in files.items():
        (tmp_path / name).write_text(text, encoding="utf-8")
    return tmp_path


def _cfg(name: str) -> list[str]:
    return ["--config", name]


def _run_dir(workdir: Path, name: str) -> Path:
    return workdir / "outputs" / "runs" / name / "dev"


def _manifest(workdir: Path, name: str) -> dict[str, Any]:
    text = (_run_dir(workdir, name) / "run_manifest.json").read_text(encoding="utf-8")
    return dict(json.loads(text))


def _ids(run: dict[str, list[Any]], qid: str) -> list[str]:
    return [h.doc_id for h in run[qid]]


# ------------------------------------------------------------------ shipped configs


@pytest.mark.parametrize("view", VIEWS)
def test_hybrid_and_rerank_share_fusion_and_parents(view: str) -> None:
    hybrid = load_config(CONFIGS / f"hybrid_{view}.toml")
    rerank = load_config(CONFIGS / f"rerank_{view}.toml")
    assert hybrid.kind == "hybrid" and rerank.kind == "rerank"
    assert hybrid.hybrid is not None and hybrid.hybrid == rerank.hybrid
    assert hybrid.hybrid.rrf_k == 60 and hybrid.hybrid.parent_depth == 100
    assert hybrid.parent_sha256 == rerank.parent_sha256
    assert {k: p.name for k, p in hybrid.parents.items()} == {
        "bm25": f"bm25_{view}",
        "dense": f"dense_{view}",
    }
    assert (hybrid.dataset, hybrid.view) == (rerank.dataset, rerank.view)
    assert hybrid.depth == rerank.depth == 100
    assert hybrid.latency_warmup == rerank.latency_warmup == 10
    assert hybrid.metrics == rerank.metrics


@pytest.mark.parametrize("view", VIEWS)
def test_shipped_rerank_settings(view: str) -> None:
    rr = load_config(CONFIGS / f"rerank_{view}.toml").rerank
    assert rr is not None
    assert rr.model == MINILM_CE.key and rr.revision == MINILM_CE.revision
    assert rr.revision == "233902d25c440f23af6f7d6e94d2946bac0bee0a"
    assert (rr.max_length, rr.rerank_depth) == (512, 30)
    assert (rr.dtype, rr.attn_implementation, rr.device) == ("float32", "eager", "cuda")
    assert MINILM_CE.files["model.safetensors"] == (
        "821d1aa69520101d6e0737f78a042ae25b19e5cb9160701909d10434f4aeb0ae"
    )


def _edited(tmp_path: Path, name: str, old: str, new: str) -> Path:
    for src in CONFIGS.glob("*.toml"):
        (tmp_path / src.name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    path = tmp_path / name
    text = path.read_text(encoding="utf-8")
    assert old in text
    path.write_text(text.replace(old, new), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("name", "old", "new"),
    [
        ("hybrid_nfcorpus.toml", "rrf_k = 60", "rrf_k = 61"),
        ("hybrid_nfcorpus.toml", "parent_depth = 100", "parent_depth = 50"),
        ("hybrid_nfcorpus.toml", "depth = 100\nlatency_warmup", "depth = 50\nlatency_warmup"),
        (
            "hybrid_nfcorpus.toml",
            'bm25_config = "bm25_nfcorpus.toml"',
            'bm25_config = "hybrid_nfcorpus.toml"',
        ),
        ("hybrid_nfcorpus.toml", 'fusion = "rrf"', 'fusion = "borda"'),
        ("hybrid_nfcorpus.toml", 'bm25_config = "bm25_nfcorpus.toml"', 'bm25_config = "x/y.toml"'),
        (
            "hybrid_nfcorpus.toml",
            'dense_config = "dense_nfcorpus.toml"',
            'dense_config = "dense_scifact_beir.toml"',  # other dataset
        ),
        (
            "hybrid_nfcorpus.toml",
            'bm25_config = "bm25_nfcorpus.toml"',
            'bm25_config = "dense_nfcorpus.toml"',  # wrong parent kind
        ),
        ("rerank_nfcorpus.toml", "rerank_depth = 30", "rerank_depth = 20"),
        ("rerank_nfcorpus.toml", "max_length = 512", "max_length = 256"),
        ("rerank_nfcorpus.toml", 'dtype = "float32"', 'dtype = "float16"'),
        ("rerank_nfcorpus.toml", 'device = "cuda"', 'device = "cpu"'),
        ("rerank_nfcorpus.toml", "batch_size = 32", "batch_size = 32\nextra = 1"),
    ],
)
def test_bad_fused_configs_are_config_errors(tmp_path: Path, name: str, old: str, new: str) -> None:
    with pytest.raises(ConfigError):
        load_config(_edited(tmp_path, name, old, new))


def test_parent_depth_must_match_the_parent_config(tmp_path: Path) -> None:
    _edited(tmp_path, "bm25_nfcorpus.toml", "depth = 100", "depth = 10")
    with pytest.raises(ConfigError, match="parent_depth"):
        load_config(tmp_path / "hybrid_nfcorpus.toml")


@pytest.mark.parametrize("command", ["run", "evaluate"])
def test_fused_split_guard_digest_changes_when_a_parent_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, command: str
) -> None:
    child = _edited(tmp_path, "hybrid_nfcorpus.toml", "latency_warmup = 10", "latency_warmup = 1")
    original = load_config(child)
    parent = tmp_path / "bm25_nfcorpus.toml"
    leaf = load_config(parent)
    assert leaf.freeze_sha256 == leaf.sha256
    parent.write_text(parent.read_text(encoding="utf-8") + "\n# Changed parent\n", encoding="utf-8")
    changed = load_config(child)
    assert changed.sha256 == original.sha256
    assert changed.freeze_sha256 != original.freeze_sha256

    def check_freeze(split: str, experiment: str, digest: str, root: Path) -> None:
        assert split == "test" and experiment == changed.name
        assert digest == changed.freeze_sha256
        raise SplitAccessError("test refused")

    monkeypatch.setattr(cli, "check_split_access", check_freeze)
    assert cli.main([command, "--config", str(child), "--split", "test"]) == EXIT_REFUSED


@pytest.mark.parametrize("name", FUSED_FILES)
@pytest.mark.parametrize("command", ["run", "evaluate"])
def test_shipped_fused_configs_refuse_the_test_split_before_any_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str, command: str
) -> None:
    monkeypatch.setattr(cli, "REPO_ROOT", tmp_path)
    args = [
        command,
        "--config",
        str(CONFIGS / name),
        "--split",
        "test",
        "--data-dir",
        str(tmp_path / "does-not-exist"),
    ]
    if command == "run":
        args += ["--models-dir", str(tmp_path / "no-models"), "--out-dir", str(tmp_path / "out")]
    assert cli.main(args) == EXIT_REFUSED
    assert not (tmp_path / "out").exists()


# ------------------------------------------------------------------ without CUDA


@pytest.mark.parametrize("config", ["hybrid.toml", "rerank.toml"])
def test_fused_arms_without_cuda_exit_5(
    workdir: Path, monkeypatch: pytest.MonkeyPatch, config: str
) -> None:
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert cli.main(["build", *_cfg(config), *COMMON, *MODELS_ARG]) == EXIT_CUDA_UNAVAILABLE
    run = ["run", *_cfg(config), *COMMON, *MODELS_ARG, "--split", "dev"]
    assert cli.main(run) == EXIT_CUDA_UNAVAILABLE
    assert not (workdir / "outputs").exists()


@pytest.mark.parametrize("config", ["hybrid.toml", "rerank.toml"])
def test_fused_test_split_refused_with_synthetic_data(workdir: Path, config: str) -> None:
    run = ["run", *_cfg(config), *COMMON, *MODELS_ARG, "--split", "test"]
    assert cli.main(run) == EXIT_REFUSED
    assert cli.main(["evaluate", *_cfg(config), *COMMON, "--split", "test"]) == EXIT_REFUSED
    assert not (workdir / "outputs").exists()


def test_index_dir_and_index_root_are_not_interchangeable(workdir: Path) -> None:
    fused = ["run", *_cfg("hybrid.toml"), *COMMON, *MODELS_ARG, "--split", "dev"]
    assert cli.main([*fused, "--index-dir", "somewhere"]) == EXIT_USAGE
    plain = ["run", *_cfg("bm25.toml"), *COMMON, "--split", "dev", "--index-root", "somewhere"]
    assert cli.main(plain) == EXIT_USAGE


@pytest.mark.parametrize("config", ["hybrid.toml", "rerank.toml"])
def test_build_refuses_missing_parent_indexes(workdir: Path, config: str) -> None:
    if not _cuda():
        pytest.skip("no CUDA device")
    assert cli.main(["build", *_cfg(config), *COMMON, *MODELS_ARG]) == EXIT_VERIFICATION_FAILED


# ------------------------------------------------------------------ end to end (CUDA)


@pytest.mark.cuda
def test_hybrid_and_rerank_end_to_end(workdir: Path) -> None:
    if not _cuda():
        pytest.skip("no CUDA device")
    # Parents through their normal commands.
    for parent in ("bm25.toml", "dense.toml"):
        assert cli.main(["build", *_cfg(parent), *COMMON, *MODELS_ARG]) == 0
        assert cli.main(["run", *_cfg(parent), *COMMON, *MODELS_ARG, "--split", "dev"]) == 0
    for arm in ("hybrid.toml", "rerank.toml"):
        assert cli.main(["build", *_cfg(arm), *COMMON, *MODELS_ARG]) == 0
        assert cli.main(["run", *_cfg(arm), *COMMON, *MODELS_ARG, "--split", "dev"]) == 0
        assert cli.main(["evaluate", *_cfg(arm), *COMMON, "--split", "dev"]) == 0

    bm25_run = read_run(_run_dir(workdir, "synthetic_bm25") / "run.trec")
    dense_run = read_run(_run_dir(workdir, "synthetic_dense") / "run.trec")
    hybrid_run = read_run(_run_dir(workdir, "synthetic_hybrid") / "run.trec")
    rerank_run = read_run(_run_dir(workdir, "synthetic_rerank") / "run.trec")
    assert hybrid_run.keys() == rerank_run.keys() == bm25_run.keys() | dense_run.keys()
    for qid in hybrid_run:
        # Hybrid equals RRF of the parents' own run files: parents are reused unchanged.
        expected = rrf_fuse(bm25_run.get(qid, []), dense_run.get(qid, []))
        assert _ids(hybrid_run, qid) == [h.doc_id for h in expected]
        # Rerank: the head (3) is a permutation of the hybrid head; the tail is identical.
        h, r = _ids(hybrid_run, qid), _ids(rerank_run, qid)
        assert len(h) == len(r) == 4
        assert set(r[:3]) == set(h[:3]) and r[3:] == h[3:]
        scores = [x.score for x in rerank_run[qid]]
        assert scores == sorted(set(scores), reverse=True)  # strictly descending

    hm, rm = _manifest(workdir, "synthetic_hybrid"), _manifest(workdir, "synthetic_rerank")
    # Both arms report the same fused candidate ordering and the same parents.
    assert hm["fused_candidates_sha256"] == rm["fused_candidates_sha256"]
    assert hm["parents"] == rm["parents"] and hm["fusion"] == rm["fusion"]
    assert hm["parents_config_sha256"] == rm["parents_config_sha256"]
    assert hm["fusion"]["rrf_k"] == 60 and hm["fusion"]["parent_depth"] == 100
    assert set(hm["index"]) == {"bm25", "dense"} and hm["index"]["bm25"]["file_sha256"]
    assert hm["index"]["dense"]["encoder"]["model_file_sha256"]
    assert "reranker" not in hm
    rr = rm["reranker"]
    assert rr["revision"] == TINY_REVISION and rr["model_file_sha256"]["model.safetensors"]
    assert rr["loading"] == {"local_files_only": True, "use_safetensors": True}
    assert rr["numerics"]["cuda_matmul_allow_tf32"] is False
    assert rr["throughput_pairs"]["pairs_scored"] == 2 * 3  # 2 queries x head of 3
    phases = ["bm25_search", "dense_query_encoding", "dense_search", "fusion"]
    for manifest, expected_phases in ((hm, phases), (rm, [*phases, "rerank"])):
        lat = manifest["latency"]
        assert lat["n"] == 2 and lat["warmup"] == 1
        assert {"p50_ms", "p90_ms", "p99_ms"} <= lat.keys()
        assert list(lat["phases"]) == expected_phases
        assert all(lat["phases"][p]["n"] == 2 for p in expected_phases)
        assert lat["gpu_memory"]["peak_allocated_bytes"] > 0
        thr = manifest["throughput"]
        assert list(thr["phase_seconds"]) == expected_phases
        assert list(thr["phase_gpu_memory"]) == expected_phases
        assert thr["gpu_memory"]["peak_allocated_bytes"] == max(
            m["peak_allocated_bytes"] for m in thr["phase_gpu_memory"].values()
        )
        assert manifest["batch_vs_single_query"]["n_queries"] == 2
        assert manifest["device"] == "cuda"
    metrics_path = _run_dir(workdir, "synthetic_rerank") / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert all(v == 0.0 for v in metrics["pytrec_eval_cross_check_max_abs_diff"].values())


@pytest.mark.cuda
def test_evaluate_refuses_changed_parent_config(workdir: Path) -> None:
    if not _cuda():
        pytest.skip("no CUDA device")
    for parent in ("bm25.toml", "dense.toml"):
        assert cli.main(["build", *_cfg(parent), *COMMON, *MODELS_ARG]) == 0
    arm = ["--config", "hybrid.toml"]
    assert cli.main(["run", *arm, *COMMON, *MODELS_ARG, "--split", "dev"]) == 0
    assert cli.main(["evaluate", *arm, *COMMON, "--split", "dev"]) == 0
    bm25 = workdir / "bm25.toml"
    bm25.write_text(bm25.read_text(encoding="utf-8") + "\n# edited\n", encoding="utf-8")
    assert cli.main(["evaluate", *arm, *COMMON, "--split", "dev"]) == EXIT_VERIFICATION_FAILED


@pytest.mark.cuda
def test_run_refuses_a_tampered_parent_index(workdir: Path) -> None:
    if not _cuda():
        pytest.skip("no CUDA device")
    for parent in ("bm25.toml", "dense.toml"):
        assert cli.main(["build", *_cfg(parent), *COMMON, *MODELS_ARG]) == 0
    ids = workdir / "artifacts" / "indexes" / "synthetic_dense" / "doc_ids.json"
    values = json.loads(ids.read_text(encoding="utf-8"))
    ids.write_text(json.dumps(values[::-1]), encoding="utf-8")
    run = ["run", *_cfg("rerank.toml"), *COMMON, *MODELS_ARG, "--split", "dev"]
    assert cli.main(run) == EXIT_VERIFICATION_FAILED
