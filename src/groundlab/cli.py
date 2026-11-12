"""Command-line interface.

    groundlab env check [--out PATH] [--require-cuda]
    groundlab data prepare --dataset {scifact,nfcorpus} --data-dir DIR [--verify-only]
    groundlab models fetch --model KEY [--models-dir DIR] [--verify-only]
    groundlab build --config CFG [--data-dir DIR] [--index-dir DIR] [--index-root DIR]
                    [--models-dir DIR]
    groundlab run --config CFG --split {dev,test} [--data-dir DIR] [--index-dir DIR]
                  [--index-root DIR] [--models-dir DIR] [--out-dir DIR]
    groundlab evaluate --config CFG --split {dev,test} [--data-dir DIR] [--run PATH] [--out PATH]
    groundlab freeze [--data-dir DIR]
    groundlab report [--runs-dir DIR] [--out-dir DIR]
    groundlab simulate [--out PATH]

Hybrid and rerank configs name their BM25 and dense parent configs. Their `build` only
verifies the parents' indexes (and the reranker files); they take `--index-root` (default
artifacts/indexes, parent indexes live in <root>/<parent name>) instead of `--index-dir`.

Exit codes: 0 ok, 1 error, 2 usage/config, 3 verification failed, 4 refused
(test split without freeze), 5 CUDA required but unavailable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from groundlab import __version__, envinfo
from groundlab.config import FUSED_KINDS, ExperimentConfig, load_config
from groundlab.data.loader import ProcessedDataset
from groundlab.data.prepare import prepare_dataset
from groundlab.data.specs import DATASETS, SPLITS
from groundlab.errors import (
    EXIT_OK,
    ConfigError,
    CudaUnavailableError,
    GroundLabError,
    VerificationError,
)
from groundlab.evaluation import metrics as M
from groundlab.evaluation.trec_check import cross_check
from groundlab.freeze import check_split_access
from groundlab.io_utils import loads_strict, sha256_file, write_json
from groundlab.models import MODELS, fetch_model, get_model_spec, model_dir, verify_model
from groundlab.retrieval.bm25 import BM25Index
from groundlab.retrieval.dense import (
    DenseIndex,
    Encoder,
    GpuMemoryProbe,
    measure_latency,
    sync_device,
)
from groundlab.retrieval.hybrid import HybridCandidates, PhaseTimer, measure_pipeline_latency
from groundlab.retrieval.rerank import CrossEncoder, rerank_query
from groundlab.runio import rankings, read_run, write_run
from groundlab.text import canonical_document_text
from groundlab.types import Query, Run, ScoredDoc

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODELS_DIR = str(Path("artifacts") / "models")


def _print(obj: Any) -> None:
    print(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False))


def _default_index_dir(cfg: ExperimentConfig) -> Path:
    return Path("artifacts") / "indexes" / cfg.name


def _default_run_dir(cfg: ExperimentConfig, split: str) -> Path:
    return Path("outputs") / "runs" / cfg.name / split


# --------------------------------------------------------------------------- commands


def cmd_env_check(args: argparse.Namespace) -> int:
    info = envinfo.collect(REPO_ROOT)
    _print(info)
    if args.out:
        write_json(Path(args.out), info)
    if args.require_cuda and not info["gpu"]["cuda_available"]:
        raise CudaUnavailableError("CUDA is required (--require-cuda) but torch reports none")
    return EXIT_OK


def cmd_data_prepare(args: argparse.Namespace) -> int:
    result = prepare_dataset(
        DATASETS[args.dataset], Path(args.data_dir), verify_only=args.verify_only
    )
    _print(
        {
            "dataset": result.dataset,
            "mode": result.audit["mode"],
            "status": result.audit["status"],
            "processed_dir": str(result.processed_dir),
            "differences": result.audit["differences"],
        }
    )
    if not result.passed:
        raise VerificationError(f"{result.dataset}: audit failed, see differences above")
    return EXIT_OK


def cmd_models_fetch(args: argparse.Namespace) -> int:
    spec = get_model_spec(args.model)
    models_dir = Path(args.models_dir)
    # --verify-only never touches the network.
    verify_or_fetch = verify_model if args.verify_only else fetch_model
    hashes = verify_or_fetch(spec, models_dir)
    _print(
        {
            "model": spec.key,
            "repo_id": spec.repo_id,
            "revision": spec.revision,
            "license": spec.license,
            "dir": str(model_dir(spec, models_dir)),
            "files_verified": sorted(hashes),
            "status": "pass",
        }
    )
    return EXIT_OK


def _require_cuda() -> None:
    import torch

    if not torch.cuda.is_available():
        raise CudaUnavailableError("dense retrieval requires CUDA; no CPU fallback exists")


def _dense_encoder(cfg: ExperimentConfig, models_dir: Path) -> Encoder:
    assert cfg.dense is not None
    _require_cuda()
    return Encoder(get_model_spec(cfg.dense.model), models_dir, cfg.dense, device=cfg.dense.device)


def _parent_index_root(args: argparse.Namespace, cfg: ExperimentConfig) -> Path:
    """Where the parents' indexes live. Fused configs use --index-root, others --index-dir."""
    if cfg.kind in FUSED_KINDS:
        if args.index_dir:
            raise ConfigError("hybrid and rerank configs take --index-root, not --index-dir")
        return Path(args.index_root) if args.index_root else Path("artifacts") / "indexes"
    if args.index_root:
        raise ConfigError("--index-root is only for hybrid and rerank configs")
    return Path()


def _load_parent_indexes(
    cfg: ExperimentConfig, models_dir: Path, index_root: Path, corpus_sha256: str
) -> tuple[BM25Index, DenseIndex]:
    """Load the existing, hash-verified BM25 index and dense embedding cache."""
    bm25_cfg, dense_cfg = cfg.parents["bm25"], cfg.parents["dense"]
    assert bm25_cfg.bm25 is not None and dense_cfg.dense is not None
    bm25 = BM25Index.load(index_root / bm25_cfg.name, bm25_cfg.bm25, corpus_sha256)
    model_hashes = verify_model(get_model_spec(dense_cfg.dense.model), models_dir)
    dense_dir = index_root / dense_cfg.name
    dense = DenseIndex.load(dense_dir, dense_cfg.dense, corpus_sha256, model_hashes)
    return bm25, dense


def _build_fused(cfg: ExperimentConfig, args: argparse.Namespace, ds: ProcessedDataset) -> int:
    """Hybrid and rerank own no index: verify what they will read, build nothing."""
    models_dir = Path(args.models_dir)
    index_root = _parent_index_root(args, cfg)
    bm25, dense = _load_parent_indexes(cfg, models_dir, index_root, ds.corpus_sha256)
    report: dict[str, Any] = {
        "kind": cfg.kind,
        "status": "parent indexes verified; nothing was built",
        "index_root": str(index_root),
        "parents": {
            "bm25": {"name": cfg.parents["bm25"].name, "n_docs": len(bm25.doc_ids)},
            "dense": {"name": cfg.parents["dense"].name, "n_docs": len(dense.doc_ids)},
        },
    }
    if cfg.rerank is not None:
        spec = get_model_spec(cfg.rerank.model)
        report["reranker_files_verified"] = sorted(verify_model(spec, models_dir))
    _print(report)
    return EXIT_OK


def cmd_build(args: argparse.Namespace) -> int:
    cfg = load_config(Path(args.config))
    if cfg.kind in FUSED_KINDS:
        _require_cuda()  # both fused arms run the CUDA dense encoder; checked before data access
    ds = ProcessedDataset(Path(args.data_dir), cfg.dataset)
    if cfg.kind in FUSED_KINDS:
        return _build_fused(cfg, args, ds)
    _parent_index_root(args, cfg)
    index_dir = Path(args.index_dir) if args.index_dir else _default_index_dir(cfg)
    if cfg.kind == "dense":
        encoder = _dense_encoder(cfg, Path(args.models_dir))
        dense_index = DenseIndex.build(encoder, ds.corpus(), ds.corpus_sha256)
        dense_index.save(index_dir)
        m = dense_index.manifest
        _print(
            {
                "index_dir": str(index_dir),
                "n_docs": m["n_docs"],
                "encode_seconds": m["encode_seconds"],
                "gpu_memory": m["gpu_memory"],
                "document_tokens": m["document_tokens"],
            }
        )
        return EXIT_OK
    assert cfg.bm25 is not None
    start = time.perf_counter()
    index = BM25Index.build(ds.corpus(), cfg.bm25, ds.corpus_sha256)
    seconds = time.perf_counter() - start
    index.save(index_dir)
    _print({"index_dir": str(index_dir), "n_docs": len(index.doc_ids), "build_seconds": seconds})
    return EXIT_OK


def _index_bytes(path: Path) -> int:
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def _run_bm25(
    cfg: ExperimentConfig, index_dir: Path, queries: list[Query], corpus_sha256: str
) -> tuple[Run, dict[str, Any]]:
    assert cfg.bm25 is not None
    index = BM25Index.load(index_dir, cfg.bm25, corpus_sha256)
    start = time.perf_counter()
    run = index.search(queries, cfg.depth)
    seconds = time.perf_counter() - start

    def single(query: Query, timer: PhaseTimer) -> list[ScoredDoc]:
        return index.search([query], cfg.depth)[query.query_id]

    latency, _ = measure_pipeline_latency(single, queries, 10, "cpu")
    return run, {
        "index": index.manifest(),
        "index_bytes": _index_bytes(index_dir),
        "latency": latency | {"gpu_memory": None},
        "search_seconds_total": seconds,
        "search_ms_per_query": 1000 * seconds / len(queries),
        "device": "cpu",
    }


def _run_dense(
    cfg: ExperimentConfig,
    index_dir: Path,
    models_dir: Path,
    queries: list[Query],
    corpus_sha256: str,
) -> tuple[Run, dict[str, Any]]:
    encoder = _dense_encoder(cfg, models_dir)
    index = DenseIndex.load(index_dir, encoder.params, corpus_sha256, encoder.model_file_sha256)
    # Batched pass: produces the run that is evaluated.
    with GpuMemoryProbe(encoder.device) as probe:
        start = time.perf_counter()
        query_emb = encoder.encode_queries(queries)
        encode_seconds = time.perf_counter() - start
    # Snapshot now: the latency probe below resets the CUDA peak counters.
    query_memory = probe.result()
    start = time.perf_counter()
    run = index.search_embeddings(queries, query_emb, cfg.depth)
    search_seconds = time.perf_counter() - start
    # Latency pass: one query at a time, end to end. Not used for the run itself.
    with GpuMemoryProbe(encoder.device) as latency_probe:
        latency = measure_latency(encoder, index, queries, cfg.depth, cfg.latency_warmup)
        latency_memory = latency_probe.result()
    single = latency.pop("single_query_embeddings")
    # Doc-ID order of each timed single-query rank call (not recomputed with batched search).
    single_rankings = latency.pop("single_query_rankings")
    same_ranking = sum(
        1
        for q, ids in zip(queries, single_rankings, strict=True)
        if ids == [h.doc_id for h in run[q.query_id]]
    )
    return run, {
        "index": {k: v for k, v in index.manifest.items() if k != "gpu_memory"}
        | {"build_gpu_memory": index.manifest.get("gpu_memory")},
        "index_bytes": _index_bytes(index_dir),
        "encoder": encoder.describe(),
        "query_encoding": {
            "batch_size": encoder.params.batch_size,
            "seconds_total": encode_seconds,
            "gpu_memory": query_memory,
        },
        "search_seconds_total": search_seconds,
        "search": "exact dot product, float64, CPU, all documents",
        "latency": latency | {"gpu_memory": latency_memory},
        "batch_vs_single_query": {
            "max_abs_embedding_diff": float(np.max(np.abs(single - query_emb))),
            "queries_with_identical_ranking": same_ranking,
            "n_queries": len(queries),
        },
        "device": encoder.device,
    }


def _candidate_hash(run: Run) -> str:
    """SHA-256 of the ordered doc IDs per query (scores ignored)."""
    digest = hashlib.sha256()
    for qid in sorted(run):
        digest.update(f"{qid}\t{' '.join(h.doc_id for h in run[qid])}\n".encode())
    return digest.hexdigest()


def _run_fused(
    cfg: ExperimentConfig,
    ds: ProcessedDataset,
    queries: list[Query],
    models_dir: Path,
    index_root: Path,
) -> tuple[Run, dict[str, Any]]:
    """Hybrid (RRF) and hybrid+rerank. Both go through the same `HybridCandidates`."""
    assert cfg.hybrid is not None
    bm25, dense_index = _load_parent_indexes(cfg, models_dir, index_root, ds.corpus_sha256)
    encoder = _dense_encoder(cfg.parents["dense"], models_dir)
    cross_encoder: CrossEncoder | None = None
    doc_texts: dict[str, str] = {}
    if cfg.rerank is not None:
        spec = get_model_spec(cfg.rerank.model)
        cross_encoder = CrossEncoder(spec, models_dir, cfg.rerank, cfg.rerank.device)
        doc_texts = {d.doc_id: canonical_document_text(d) for d in ds.corpus()}
    candidates = HybridCandidates(bm25, dense_index, encoder, cfg.hybrid, cfg.depth)
    device = encoder.device

    def rerank(query: Query, fused: list[ScoredDoc], timer: PhaseTimer) -> list[ScoredDoc]:
        assert cfg.rerank is not None and cross_encoder is not None
        with timer.phase("rerank"):
            ranked = rerank_query(cross_encoder, query, fused, doc_texts, cfg.rerank.rerank_depth)
        return ranked

    # Throughput pass: batched query encoding and search; this is the run that is evaluated.
    # Every phase reads its own CUDA peak counters before the next phase can reset them.
    timer = PhaseTimer(device, track_memory=True)
    start = time.perf_counter()
    fused, parent_stats = candidates.batch(queries, timer)
    run = fused
    if cross_encoder is not None:
        run = {q.query_id: rerank(q, fused[q.query_id], timer) for q in queries}
    sync_device(device)
    throughput_seconds = time.perf_counter() - start
    pair_stats = cross_encoder.take_stats() if cross_encoder is not None else None

    # Latency pass: one query at a time through the whole pipeline. Not used for the run.
    def step(query: Query, step_timer: PhaseTimer) -> list[ScoredDoc]:
        ranked = candidates.single(query, step_timer)
        return ranked if cross_encoder is None else rerank(query, ranked, step_timer)

    with GpuMemoryProbe(device) as latency_probe:
        latency, single_rankings = measure_pipeline_latency(
            step, queries, cfg.latency_warmup, device
        )
        latency_memory = latency_probe.result()
    same_ranking = sum(
        1
        for q, ranked in zip(queries, single_rankings, strict=True)
        if [h.doc_id for h in ranked] == [h.doc_id for h in run[q.query_id]]
    )
    parents = {
        key: {
            "name": parent.name,
            "config_path": parent.path.as_posix(),
            "config_sha256": parent.sha256,
        }
        for key, parent in cfg.parents.items()
    }
    details: dict[str, Any] = {
        "parents": parents,
        "parents_config_sha256": cfg.parent_sha256,
        "index_root": index_root.as_posix(),
        "index": {
            "bm25": bm25.manifest(),
            "dense": {k: v for k, v in dense_index.manifest.items() if k != "gpu_memory"}
            | {"build_gpu_memory": dense_index.manifest.get("gpu_memory")},
        },
        "index_bytes": sum(_index_bytes(index_root / p.name) for p in cfg.parents.values()),
        "fusion": asdict(cfg.hybrid)
        | {
            "output_depth": cfg.depth,
            "rank_base": 1,
            "arithmetic": "exact rational sum, stored score is the rounded float",
            "ties": "score descending, then doc ID ascending (string order)",
            "padding": "none; an absent parent hit contributes 0",
        },
        "parent_list_stats": parent_stats,
        "fused_candidates_sha256": _candidate_hash(fused),
        "encoder": encoder.describe(),
        "throughput": {
            "protocol": "all queries in one pass: BM25 loop, batched dense query encoding, "
            "batched exact search, fusion, then (rerank arm) one rerank call per query; "
            "CUDA synchronized at every phase boundary",
            "n_queries": len(queries),
            "seconds_total": throughput_seconds,
            "phase_seconds": timer.total_seconds(),
            "phase_gpu_memory": timer.peaks,
            "gpu_memory": timer.overall_peak(),
            "gpu_memory_note": "PyTorch peak allocated/reserved, includes resident model "
            "weights, excludes the CUDA context",
        },
        "latency": latency | {"gpu_memory": latency_memory},
        "batch_vs_single_query": {
            "queries_with_identical_ranking": same_ranking,
            "n_queries": len(queries),
        },
        "device": device,
    }
    if cross_encoder is not None:
        details["reranker"] = cross_encoder.describe() | {"throughput_pairs": pair_stats}
    return run, details


def cmd_run(args: argparse.Namespace) -> int:
    cfg = load_config(Path(args.config))
    # Refuse before any data is read.
    freeze = check_split_access(args.split, cfg.name, cfg.freeze_sha256, REPO_ROOT)
    index_root = _parent_index_root(args, cfg)
    if cfg.kind != "bm25":
        _require_cuda()
    ds = ProcessedDataset(Path(args.data_dir), cfg.dataset, freeze)
    inputs = ds.input_sha256(cfg.view, args.split, freeze)
    queries = ds.queries(cfg.view, args.split, freeze)
    index_dir = Path(args.index_dir) if args.index_dir else _default_index_dir(cfg)
    if cfg.kind in FUSED_KINDS:
        run, details = _run_fused(cfg, ds, queries, Path(args.models_dir), index_root)
    elif cfg.kind == "dense":
        run, details = _run_dense(cfg, index_dir, Path(args.models_dir), queries, ds.corpus_sha256)
    else:
        run, details = _run_bm25(cfg, index_dir, queries, ds.corpus_sha256)
    out_dir = Path(args.out_dir) if args.out_dir else _default_run_dir(cfg, args.split)
    run_path = out_dir / "run.trec"
    write_run(run_path, run, tag=cfg.name)
    manifest = {
        "experiment": cfg.name,
        "config_path": cfg.path.as_posix(),
        "config_sha256": cfg.sha256,
        "config": cfg.raw,
        "dataset": cfg.dataset,
        "view": cfg.view,
        "split": args.split,
        "dataset_files": ds.manifest["files"],
        "input_sha256": inputs,
        "freeze": asdict(freeze) if freeze else None,
        "n_queries": len(queries),
        "n_queries_empty_ranking": sum(1 for q in queries if not run[q.query_id]),
        **details,
        "run_sha256": sha256_file(run_path),
        "environment": envinfo.collect(REPO_ROOT),
    }
    write_json(out_dir / "run_manifest.json", manifest)
    summary = {k: manifest[k] for k in ("experiment", "split", "n_queries", "device")}
    if "latency" in manifest:
        summary["latency_p50_ms"] = manifest["latency"]["p50_ms"]
    _print(summary | {"run": str(run_path)})
    return EXIT_OK


def cmd_evaluate(args: argparse.Namespace) -> int:
    cfg = load_config(Path(args.config))
    freeze = check_split_access(args.split, cfg.name, cfg.freeze_sha256, REPO_ROOT)
    ds = ProcessedDataset(Path(args.data_dir), cfg.dataset, freeze)
    run_path = Path(args.run) if args.run else _default_run_dir(cfg, args.split) / "run.trec"
    run_manifest_path = run_path.with_name("run_manifest.json")
    if not run_manifest_path.exists():
        raise VerificationError(f"{run_manifest_path} missing; runs must come from `groundlab run`")
    run_manifest = loads_strict(run_manifest_path.read_text(encoding="utf-8"))
    run_sha = sha256_file(run_path)
    for key, expected in (
        ("config_sha256", cfg.sha256),
        ("split", args.split),
        ("view", cfg.view),
        ("run_sha256", run_sha),
    ):
        if run_manifest.get(key) != expected:
            raise VerificationError(
                f"run manifest {key} = {run_manifest.get(key)!r}, expected {expected!r}"
            )
    if freeze and run_manifest.get("freeze") != asdict(freeze):
        raise VerificationError("run was not produced under the current freeze")
    if cfg.parents and run_manifest.get("parents_config_sha256") != cfg.parent_sha256:
        raise VerificationError("parent configs changed since the run was produced")
    # The corpus, queries and qrels on disk now must be exactly the ones the run saw.
    current_inputs = ds.input_sha256(cfg.view, args.split, freeze)
    recorded_inputs = run_manifest.get("input_sha256")
    if not isinstance(recorded_inputs, dict):
        raise VerificationError("run manifest has no input_sha256; re-run `groundlab run`")
    if recorded_inputs != current_inputs:
        changed = sorted(
            rel
            for rel in recorded_inputs.keys() | current_inputs.keys()
            if recorded_inputs.get(rel) != current_inputs.get(rel)
        )
        raise VerificationError(f"inputs changed since the run was produced: {changed}")
    qrels = ds.qrels(cfg.view, args.split, freeze)
    ranked = rankings(read_run(run_path))
    per_q = M.per_query(ranked, qrels, cfg.metrics)
    means = M.mean(per_q, cfg.metrics)
    max_diff = cross_check(ranked, qrels, cfg.metrics, per_q)
    results = {
        "experiment": cfg.name,
        "config_sha256": cfg.sha256,
        "dataset": cfg.dataset,
        "view": cfg.view,
        "split": args.split,
        "run_sha256": run_sha,
        "input_sha256": current_inputs,
        "freeze": asdict(freeze) if freeze else None,
        "n_queries": len(per_q),
        "n_queries_missing_from_run": len(qrels.keys() - ranked.keys()),
        "mean": means,
        "pytrec_eval_cross_check_max_abs_diff": max_diff,
        "per_query": per_q,
        "groundlab": __version__,
    }
    if cfg.dataset == "scifact" and cfg.view == "evidence":
        from groundlab.evaluation.paired import cluster_claims

        clusters = cluster_claims(qrels)
        if freeze:
            beir = ds.qrels("beir", args.split, freeze)
            cited_clusters = cluster_claims({qid: beir[qid] for qid in qrels})
            if (
                clusters != cited_clusters
                or len(clusters) != 188
                or len(set(clusters.values())) != 164
            ):
                raise VerificationError("SciFact cluster partition differs from the frozen audit")
        results["clusters"] = clusters
    out = Path(args.out) if args.out else run_path.with_name("metrics.json")
    write_json(out, results)
    _print(
        {k: results[k] for k in ("experiment", "split", "n_queries", "mean")} | {"out": str(out)}
    )
    return EXIT_OK


def cmd_freeze(args: argparse.Namespace) -> int:
    from groundlab.study import create_freeze

    _print(create_freeze(REPO_ROOT, Path(args.data_dir)))
    return EXIT_OK


def cmd_report(args: argparse.Namespace) -> int:
    from groundlab.study import report

    report(REPO_ROOT, Path(args.runs_dir), Path(args.out_dir))
    _print({"report": str(Path(args.out_dir) / "REPORT.md")})
    return EXIT_OK


def cmd_simulate(args: argparse.Namespace) -> int:
    from groundlab.evaluation.paired import simulate

    results = simulate()
    if args.out:
        write_json(Path(args.out), results)
    _print(results)
    return EXIT_OK


# --------------------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="groundlab", description=__doc__.split("\n")[0])
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    env = sub.add_parser("env", help="environment commands")
    env_sub = env.add_subparsers(dest="env_command", required=True)
    check = env_sub.add_parser("check", help="record OS, hardware, CUDA and package versions")
    check.add_argument("--out", help="also write the JSON record to this path")
    check.add_argument("--require-cuda", action="store_true", help="exit 5 if CUDA is unavailable")
    check.set_defaults(func=cmd_env_check)

    data = sub.add_parser("data", help="dataset commands")
    data_sub = data.add_subparsers(dest="data_command", required=True)
    prep = data_sub.add_parser("prepare", help="download, hash-verify, build views, audit")
    prep.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    prep.add_argument("--data-dir", required=True)
    prep.add_argument(
        "--verify-only",
        action="store_true",
        help="no download or writes; re-verify archives and processed files",
    )
    prep.set_defaults(func=cmd_data_prepare)

    models = sub.add_parser("models", help="pinned pretrained models")
    models_sub = models.add_subparsers(dest="models_command", required=True)
    fetch = models_sub.add_parser("fetch", help="download pinned files and verify SHA-256")
    fetch.add_argument("--model", required=True, choices=sorted(MODELS))
    fetch.add_argument("--models-dir", default=DEFAULT_MODELS_DIR)
    fetch.add_argument("--verify-only", action="store_true", help="no download; verify only")
    fetch.set_defaults(func=cmd_models_fetch)

    def experiment_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--config", required=True)
        p.add_argument("--data-dir", default="data")

    build = sub.add_parser("build", help="build the BM25 index or dense embedding cache")
    experiment_args(build)
    build.add_argument("--index-dir")
    build.add_argument("--index-root", help="hybrid/rerank: directory holding the parent indexes")
    build.add_argument("--models-dir", default=DEFAULT_MODELS_DIR)
    build.set_defaults(func=cmd_build)

    run = sub.add_parser("run", help="retrieve for one split (test refused unless frozen)")
    experiment_args(run)
    run.add_argument("--split", required=True, choices=SPLITS)
    run.add_argument("--index-dir")
    run.add_argument("--index-root", help="hybrid/rerank: directory holding the parent indexes")
    run.add_argument("--models-dir", default=DEFAULT_MODELS_DIR)
    run.add_argument("--out-dir")
    run.set_defaults(func=cmd_run)

    ev = sub.add_parser("evaluate", help="score a run (test refused unless frozen)")
    experiment_args(ev)
    ev.add_argument("--split", required=True, choices=SPLITS)
    ev.add_argument("--run")
    ev.add_argument("--out")
    ev.set_defaults(func=cmd_evaluate)
    freeze = sub.add_parser("freeze", help="write manifests for the complete study")
    freeze.add_argument("--data-dir", default="data")
    freeze.set_defaults(func=cmd_freeze)

    report = sub.add_parser("report", help="render the held-out report from saved artifacts")
    report.add_argument("--runs-dir", default="outputs/runs")
    report.add_argument("--out-dir", default="outputs/report")
    report.set_defaults(func=cmd_report)

    simulation = sub.add_parser("simulate", help="run the pre-freeze synthetic stress tests")
    simulation.add_argument("--out")
    simulation.set_defaults(func=cmd_simulate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except GroundLabError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return exc.exit_code


if __name__ == "__main__":
    sys.exit(main())
