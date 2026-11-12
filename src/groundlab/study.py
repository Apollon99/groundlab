"""Study freeze and offline report generation from validated run artifacts."""

from __future__ import annotations

import csv
import io
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

from groundlab.config import ExperimentConfig, load_config
from groundlab.data.loader import ProcessedDataset
from groundlab.data.prepare import prepare_dataset
from groundlab.data.specs import DATASETS
from groundlab.errors import VerificationError
from groundlab.evaluation.paired import PRACTICAL_DELTA, analyze
from groundlab.freeze import FROZEN_PATHS, PROTOCOL_PATH, FreezeManifest, _git, require_frozen
from groundlab.io_utils import atomic_write_bytes, dumps_json, loads_strict, sha256_file, write_json
from groundlab.retrieval.hybrid import rrf_fuse
from groundlab.runio import read_run

VIEWS = {
    "scifact_evidence": ("scifact", "evidence", 188),
    "scifact_beir": ("scifact", "beir", 300),
    "nfcorpus": ("nfcorpus", "beir", 323),
}
ARMS = ("bm25", "dense", "hybrid", "rerank")
METRICS = ("ndcg@10", "recall@10", "recall@100", "mrr@10")


def _configs(repo: Path) -> dict[tuple[str, str], ExperimentConfig]:
    configs = {}
    for view, (dataset, judgment, _) in VIEWS.items():
        for arm in ARMS:
            name = f"{arm}_{view}"
            cfg = load_config(repo / "configs" / f"{name}.toml")
            if (cfg.name, cfg.kind, cfg.dataset, cfg.view) != (name, arm, dataset, judgment):
                raise VerificationError(f"unexpected experiment identity: {cfg.path}")
            if cfg.depth != 100 or tuple(m.label for m in cfg.metrics) != METRICS:
                raise VerificationError(f"unexpected depth or metrics: {cfg.path}")
            if arm != "bm25" and cfg.latency_warmup != 10:
                raise VerificationError(f"unexpected warm-up count: {cfg.path}")
            configs[view, arm] = cfg
        for arm in ("hybrid", "rerank"):
            parents = configs[view, arm].parents
            if any(parents[parent].sha256 != configs[view, parent].sha256 for parent in parents):
                raise VerificationError(f"{arm}_{view} must use the same parent configs")
    return configs


def create_freeze(repo: Path, data_dir: Path) -> list[str]:
    """Verify the entire study and write its manifests. Git commit is a separate step."""
    if _git(repo, "status", "--porcelain", "--", *FROZEN_PATHS):
        raise VerificationError("commit changes to the code, configs and protocol before freezing")
    code_tree = _git(repo, "rev-parse", "HEAD:src/groundlab")
    protocol = sha256_file(repo / PROTOCOL_PATH)
    lock = sha256_file(repo / "uv.lock")
    configs = _configs(repo)
    manifest_digests = {}
    for dataset, spec in DATASETS.items():
        verified = prepare_dataset(spec, data_dir, verify_only=True)
        if not verified.passed:
            raise VerificationError(f"prepared {dataset} files do not match the pinned archives")
        ds = ProcessedDataset(data_dir, dataset)
        if not isinstance(ds.manifest.get("files"), dict):
            raise VerificationError(f"invalid {dataset} prepared manifest")
        # All prepared files have just been checked against regenerated source bytes.
        # Independently ensure the manifest also describes exactly those prepared files.
        actual = {
            p.relative_to(ds.root).as_posix(): sha256_file(p)
            for p in ds.root.rglob("*")
            if p.is_file() and p.name not in ("manifest.json", "audit.json")
        }
        if ds.manifest["files"] != actual:
            raise VerificationError(f"prepared {dataset} manifest differs from its files")
        manifest_digests[dataset] = sha256_file(ds.root / "manifest.json")

    outputs = {}
    for cfg in configs.values():
        freeze = FreezeManifest(
            cfg.name, cfg.freeze_sha256, code_tree, lock, manifest_digests[cfg.dataset], protocol
        )
        path = repo / "freeze" / f"{cfg.name}.json"
        encoded = dumps_json(asdict(freeze))
        if path.exists() and path.read_bytes() != encoded:
            raise VerificationError(f"existing freeze differs: {path.name}")
        outputs[path] = encoded
    for path, encoded in outputs.items():
        if not path.exists():
            atomic_write_bytes(path, encoded)
    return [path.name for path in outputs]


def _json(path: Path) -> dict[str, Any]:
    try:
        result = loads_strict(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise VerificationError(f"cannot read {path}: {exc}") from exc
    if not isinstance(result, dict):
        raise VerificationError(f"{path} is not a JSON object")
    return result


def _load_run(
    repo: Path, root: Path, cfg: ExperimentConfig, expected_count: int
) -> tuple[dict[str, Any], dict[str, Any], Path, dict[str, str]]:
    freeze = require_frozen(cfg.name, cfg.freeze_sha256, repo)
    if not freeze.data_manifest_sha256 or not freeze.protocol_sha256:
        raise VerificationError(f"{cfg.name}: incomplete study freeze")
    directory = root / cfg.name / "test"
    run_path = directory / "run.trec"
    manifest_path, metrics_path = directory / "run_manifest.json", directory / "metrics.json"
    manifest, metrics = _json(manifest_path), _json(metrics_path)
    if not run_path.is_file():
        raise VerificationError(f"missing ranking: {run_path}")
    hashes = {path.name: sha256_file(path) for path in (run_path, manifest_path, metrics_path)}
    common = {
        "experiment": cfg.name,
        "config_sha256": cfg.sha256,
        "freeze": asdict(freeze),
        "split": "test",
        "dataset": cfg.dataset,
        "view": cfg.view,
        "run_sha256": hashes["run.trec"],
        "n_queries": expected_count,
    }
    if any(any(record.get(k) != v for k, v in common.items()) for record in (manifest, metrics)):
        raise VerificationError(f"{cfg.name}: metrics or run manifest differs from frozen run")
    if manifest.get("input_sha256") != metrics.get("input_sha256"):
        raise VerificationError(f"{cfg.name}: evaluation input hashes differ from run")
    if cfg.parents and manifest.get("parents_config_sha256") != cfg.parent_sha256:
        raise VerificationError(f"{cfg.name}: parent configs changed")
    if manifest.get("environment", {}).get("git", {}).get("dirty") is not False:
        raise VerificationError(f"{cfg.name}: report requires runs from a clean checkout")
    per_query, means = metrics["per_query"], metrics["mean"]
    checks = metrics["pytrec_eval_cross_check_max_abs_diff"]
    if (
        len(per_query) != expected_count
        or set(means) != set(METRICS)
        or set(checks) != set(METRICS)
    ):
        raise VerificationError(f"{cfg.name}: missing evaluation metrics")
    for metric in METRICS:
        values = [q[metric] for q in per_query.values()]
        if (
            not all(0 <= v <= 1 for v in values)
            or not math.isclose(means[metric], math.fsum(values) / len(values), abs_tol=1e-12)
            or not 0 <= checks[metric] <= 1e-9
        ):
            raise VerificationError(f"{cfg.name}: invalid or inconsistent {metric}")
    return manifest, metrics, run_path, hashes


def _candidates(paths: dict[str, Path], queries: set[str]) -> dict[str, int]:
    bm25, dense, hybrid, rerank = (read_run(paths[arm]) for arm in ARMS)
    if any(run.keys() != queries for run in (dense, hybrid, rerank)) or not bm25.keys() <= queries:
        raise VerificationError("candidate query sets differ from evaluation cohort")
    changed = 0
    for qid in sorted(queries):
        expected = rrf_fuse(bm25.get(qid, []), dense[qid])
        fused = hybrid[qid]
        final = [doc.doc_id for doc in rerank[qid]]
        before = [doc.doc_id for doc in fused]
        if fused != expected or set(final[:30]) != set(before[:30]) or final[30:] != before[30:]:
            raise VerificationError(f"candidate membership or order changed for {qid}")
        if len(set(final)) != len(final) or len(final) != len(before):
            raise VerificationError(f"duplicate or missing reranked candidate for {qid}")
        changed += final[:30] != before[:30]
    return {
        "queries": len(queries),
        "bm25_empty": len(queries - bm25.keys()),
        "head_reordered": changed,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    buf = io.StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_bytes(path, buf.getvalue().encode("utf-8"))


def report(repo: Path, runs_dir: Path, out: Path) -> None:
    """Validate all saved test runs and render stable Markdown, CSV and JSON outputs."""
    configs = _configs(repo)
    rows: list[dict[str, Any]] = []
    hashes: dict[str, Any] = {}
    diagnostics = {}
    primary: dict[str, dict[str, float]] = {}
    primary_clusters: dict[str, str] | None = None
    for view, (_, _, count) in VIEWS.items():
        paths = {}
        cohort: set[str] | None = None
        for arm in ARMS:
            cfg = configs[view, arm]
            run, metrics, path, inputs = _load_run(repo, runs_dir, cfg, count)
            paths[arm] = path
            hashes[cfg.name] = {
                "directory": path.parent.relative_to(runs_dir).as_posix(),
                "files": inputs,
                "git_commit": run["environment"]["git"]["commit"],
                "frozen": run["freeze"],
            }
            ids = set(metrics["per_query"])
            if cohort is not None and ids != cohort:
                raise VerificationError(f"{view}: retrieval arms used different queries")
            cohort = ids
            if view == "scifact_evidence":
                clusters = metrics.get("clusters")
                if not isinstance(clusters, dict) or (
                    primary_clusters is not None and clusters != primary_clusters
                ):
                    raise VerificationError("primary evaluation files disagree on claim clusters")
                primary_clusters = clusters
                primary[arm] = {q: m["ndcg@10"] for q, m in metrics["per_query"].items()}
            latency = run["latency"]
            peak = latency.get("gpu_memory") or {}
            rows.append(
                {
                    "view": view,
                    "arm": arm,
                    "queries": count,
                    **{m: metrics["mean"][m] for m in METRICS},
                    "latency_p50_ms": latency["p50_ms"],
                    "latency_p90_ms": latency["p90_ms"],
                    "latency_p99_ms": latency["p99_ms"],
                    "gpu_allocated_bytes": peak.get("peak_allocated_bytes"),
                    "gpu_reserved_bytes": peak.get("peak_reserved_bytes"),
                    "index_bytes": run["index_bytes"],
                }
            )
        assert cohort is not None
        diagnostics[view] = _candidates(paths, cohort)
    if primary_clusters is None or len(set(primary_clusters.values())) != 164:
        raise VerificationError("expected 164 primary claim clusters")
    statistics = analyze(primary, primary_clusters)
    for contrast in statistics["contrasts"]:
        contrast["observed_gain_at_least_0.02"] = contrast["delta"] >= PRACTICAL_DELTA
    write_json(
        out / "study.json",
        {
            "input_artifacts": hashes,
            "metrics": rows,
            "paired": statistics,
            "candidates": diagnostics,
        },
    )
    _write_csv(out / "metrics.csv", rows)
    _write_csv(out / "contrasts.csv", statistics["contrasts"])
    atomic_write_bytes(out / "REPORT.md", _markdown(rows, statistics, diagnostics, hashes))


def _markdown(
    rows: list[dict[str, Any]],
    statistics: dict[str, Any],
    diagnostics: dict[str, dict[str, int]],
    hashes: dict[str, Any],
) -> bytes:
    lines = [
        "# GroundLab held-out retrieval study",
        "",
        "Four retrieval methods were evaluated against the frozen SciFact and NFCorpus views.",
        "SciFact evidence is the primary view. The other views and secondary metrics "
        "are exploratory.",
        "",
        "## Primary paired comparisons",
        "",
        f"{statistics['n_queries']} claims in {statistics['n_clusters']} evidence clusters. "
        "Effects are mean claim-level differences in nDCG@10. The 95% cluster-bootstrap "
        "intervals are marginal. Holm-adjusted p-values are from two-sided cluster sign flips.",
        "",
        "| Contrast | Δ nDCG@10 | 95% CI | Raw p | Holm p |",
        "|---|---:|---:|---:|---:|",
    ]
    for c in statistics["contrasts"]:
        label = c["name"].replace("_minus_", " \u2212 ")
        lines.append(
            f"| {label} | {c['delta']:+.4f} | [{c['ci_low']:+.4f}, {c['ci_high']:+.4f}] | "
            f"{c['p_raw']:.5g} | {c['p_holm']:.5g} |"
        )
    lines.extend(
        [
            "",
            "A positive contrast with Holm p below 0.05 supports a gain under the "
            "sign-flip assumptions. "
            "An observed gain of 0.02 is the descriptive practical threshold. "
            "These tests compare against zero.",
            "",
            "## Retrieval quality",
            "",
            "| View | Arm | Queries | nDCG@10 | Recall@10 | Recall@100 | MRR@10 |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for r in rows:
        lines.append(
            f"| {r['view']} | {r['arm']} | {r['queries']} | "
            + " | ".join(f"{r[m]:.4f}" for m in METRICS)
            + " |"
        )
    lines.extend(
        [
            "",
            "## Latency and resources",
            "",
            "Timings cover warmed, single-query retrieval through ranking on this machine. "
            "They exclude loading and use one observation per query. GPU memory is PyTorch peak "
            "allocated memory, including resident weights. Index size excludes model files.",
            "",
            "| View | Arm | p50 ms | p90 ms | p99 ms | GPU peak MiB | Index MiB |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for r in rows:
        memory = r["gpu_allocated_bytes"]
        gpu = "—" if memory is None else f"{memory / 2**20:.1f}"
        lines.append(
            f"| {r['view']} | {r['arm']} | {r['latency_p50_ms']:.2f} | "
            f"{r['latency_p90_ms']:.2f} | {r['latency_p99_ms']:.2f} | "
            f"{gpu} | {r['index_bytes'] / 2**20:.2f} |"
        )
    lines.extend(["", "## Candidate integrity", ""])
    for view, diag in diagnostics.items():
        lines.append(
            f"- **{view}**. The top-30 membership and tail were preserved for all "
            f"{diag['queries']} queries. The head was reordered for {diag['head_reordered']}; "
            f"BM25 returned no hits for {diag['bm25_empty']}."
        )
    commits = ", ".join(sorted({h["git_commit"] for h in hashes.values()}))
    lines.extend(
        [
            "",
            "## Scope and reproducibility",
            "",
            "Evidence judgments are incomplete. Frozen pretrained models can have seen benchmark "
            "material. The timing results characterize the recorded hardware and workload. The "
            "statistical test assumes independent clusters with a joint sign-symmetric null.",
            "",
            f"Run commit(s): {commits}. See study.json for artifact hashes and freeze records, "
            "metrics.csv for all metric rows, and contrasts.csv for all primary contrasts.",
            "",
        ]
    )
    return "\n".join(lines).encode("utf-8")
