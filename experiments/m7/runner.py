"""Separate, resumable downstream verification study. Run from the repository root."""

from __future__ import annotations

import argparse
import platform
import subprocess
import time
from collections.abc import Sequence
from itertools import combinations
from pathlib import Path
from typing import Any

from runtime import (
    CONTEXT,
    DOC_LIMIT,
    LABELS,
    OUTPUT_LIMIT,
    SCHEMA,
    SEED,
    SYSTEM,
    THREADS,
    TIMEOUT,
    Server,
    open_server,
)

from groundlab.config import load_config
from groundlab.data.archives import ensure_archive, read_members
from groundlab.data.loader import ProcessedDataset
from groundlab.data.specs import AI2_SCIFACT
from groundlab.freeze import require_frozen
from groundlab.io_utils import (
    atomic_write_bytes,
    dumps_json,
    dumps_jsonl,
    loads_strict,
    sha256_bytes,
    sha256_file,
    write_json,
)
from groundlab.runio import rankings, read_run

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "outputs/m7"
FREEZE = HERE / "freeze.json"
ARMS = ("closed_book", "bm25", "rerank", "oracle")
GOLD = LABELS[:2]
RUNS = {arm: f"{arm}_scifact_evidence" for arm in ("bm25", "rerank")}
SECONDS_BUDGET = 2 * 60 * 60


def read_json(path: Path) -> Any:
    return loads_strict(path.read_text(encoding="utf-8"))


def pins() -> dict[str, Any]:
    return read_json(HERE / "pins.json")


def verify_runtime(config: dict[str, Any]) -> tuple[Path, Path]:
    model = ROOT / config["model"]["path"]
    exe = ROOT / config["runtime"]["path"]
    for path, digest in (
        (model, config["model"]["sha256"]),
        (exe, config["runtime"]["sha256"]),
        (ROOT / config["runtime"]["archive"], config["runtime"]["archive_sha256"]),
    ):
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"missing or changed pinned artifact: {path}")
    return exe, model


def source() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, list[str]]]:
    archive = ensure_archive(AI2_SCIFACT, ROOT / "data/raw", allow_download=False)
    files = read_members(AI2_SCIFACT, archive)

    def rows(name: str) -> list[dict[str, Any]]:
        return [
            loads_strict(line) for line in files[f"data/{name}.jsonl"].decode("utf-8").splitlines()
        ]

    return (
        rows("claims_train"),
        rows("claims_dev"),
        {str(doc["doc_id"]): doc["abstract"] for doc in rows("corpus")},
    )


def gold_evidence(claim: dict[str, Any]) -> tuple[str, dict[str, list[list[int]]]]:
    annotations = {}
    labels = set()
    for doc_id, entries in claim["evidence"].items():
        sets = []
        for evidence in entries:
            sentences = evidence["sentences"]
            if sentences:
                labels.add(evidence["label"])
                sets.append(sentences)
        if sets:
            annotations[doc_id] = sets
    if len(labels) != 1 or not labels <= set(GOLD):
        raise ValueError(f"claim must have a single annotated stance: {claim['id']}")
    return next(iter(labels)), annotations


def sample(claims: Sequence[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    evidence_claims = [claim for claim in claims if any(claim["evidence"].values())]
    if len({c["id"] for c in evidence_claims}) != len(evidence_claims):
        raise ValueError("duplicate evidence claim")
    order = sorted(
        evidence_claims,
        key=lambda claim: (
            sha256_bytes(f"groundlab-v1|{claim['id']}".encode()),
            str(claim["id"]),
        ),
    )
    result = []
    for stance in GOLD:
        group = [claim for claim in order if gold_evidence(claim)[0] == stance]
        if len(group) < n:
            raise ValueError(f"not enough {stance} claims")
        result.extend(group[:n])
    return result


def sample_ids(claims: Sequence[dict[str, Any]]) -> list[str]:
    return [str(claim["id"]) for claim in sample(claims, 30)]


def inputs() -> tuple[ProcessedDataset, dict[str, Any]]:
    cfg = load_config(ROOT / "configs/bm25_scifact_evidence.toml")
    freeze = require_frozen(cfg.name, cfg.freeze_sha256, ROOT)
    dataset = ProcessedDataset(ROOT / "data", "scifact", freeze)
    return dataset, {"token": freeze, "study": read_json(ROOT / "outputs/report/study.json")}


def frozen_record() -> dict[str, Any]:
    config = pins()
    exe, _ = verify_runtime(config)
    dataset, bound = inputs()
    _, dev, _ = source()
    queries = {q.query_id: q.text for q in dataset.queries("evidence", "test", bound["token"])}
    qrels = dataset.qrels("evidence", "test", bound["token"])
    evidence = [claim for claim in dev if any(claim["evidence"].values())]
    if len(queries) != 188 or set(queries) != {str(c["id"]) for c in evidence}:
        raise ValueError("evidence cohort differs from the frozen M6 view")
    for claim in evidence:
        qid = str(claim["id"])
        if queries[qid] != claim["claim"] or set(qrels[qid]) != set(gold_evidence(claim)[1]):
            raise ValueError(f"ground truth does not match the prepared cohort: {qid}")

    runtime_hashes = {
        p.name: sha256_file(p)
        for p in sorted(exe.parent.iterdir())
        if p.is_file() and p.suffix.lower() in (".dll", ".exe")
    }
    return {
        "pins": config,
        "runtime_files": runtime_hashes,
        "runner_sha256": {
            name: sha256_file(HERE / name) for name in ("runner.py", "runtime.py", "PROTOCOL.md")
        },
        "dependency_lock_sha256": sha256_file(ROOT / "uv.lock"),
        "ai2_archive_sha256": AI2_SCIFACT.sha256,
        "dataset_manifest_sha256": bound["token"].data_manifest_sha256,
        "study_report_sha256": sha256_file(ROOT / "outputs/report/study.json"),
        "rankings_sha256": {
            arm: bound["study"]["input_artifacts"][name]["files"]["run.trec"]
            for arm, name in RUNS.items()
        },
        "generator": {
            "system_prompt": SYSTEM,
            "schema": SCHEMA,
            "context_tokens": CONTEXT,
            "output_tokens": OUTPUT_LIMIT,
            "document_tokens": DOC_LIMIT,
            "threads": THREADS,
            "seed": SEED,
            "temperature": 0,
            "request_timeout_seconds": TIMEOUT,
        },
        "sample_size": 60,
        "sample_sha256": sha256_bytes(dumps_json(sample_ids(evidence))),
        "arms": list(ARMS),
    }


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def verified_freeze() -> dict[str, Any]:
    saved = read_json(FREEZE)
    if git("status", "--porcelain", "--", "experiments/m7"):
        raise ValueError("commit the M7 code and freeze before running")
    git("ls-files", "--error-unmatch", "experiments/m7/freeze.json")
    if saved != frozen_record():
        raise ValueError("M7 inputs, runtime, protocol or code differ from freeze")
    return saved


def contexts(record: dict[str, Any], cohort: list[str]) -> dict[str, dict[str, list[str]]]:
    output = {}
    for arm, name in RUNS.items():
        path = ROOT / "outputs/runs" / name / "test/run.trec"
        if sha256_file(path) != record["rankings_sha256"][arm]:
            raise ValueError(f"retrieval artifact changed: {name}")
        ranked = rankings(read_run(path))
        if len(ranked) != 188 or not set(cohort) <= ranked.keys():
            raise ValueError(f"incomplete retrieval ranking: {name}")
        output[arm] = ranked
    return output


def assess(raw: str, shown: set[str], gold: set[str], gold_label: str) -> dict[str, Any]:
    try:
        answer = loads_strict(raw)
    except ValueError:
        answer = None
    if not isinstance(answer, dict):
        answer = {}
    label = answer.get("label")
    values = answer.get("citations")
    attempts = values if isinstance(values, list) else []
    strings = [value for value in attempts if isinstance(value, str)]
    valid = (
        set(answer) == {"label", "citations"}
        and label in LABELS
        and isinstance(values, list)
        and len(strings) == len(attempts) == len(set(strings)) <= 3
        and set(strings) <= shown
        and bool(strings) == (bool(shown) and label in GOLD)
    )
    correct = valid and label == gold_label
    return {
        "status": "ok" if valid else "invalid",
        "label": label,
        "citations": attempts,
        "correct": correct,
        "abstained": valid and label == "INSUFFICIENT",
        "attempted": len(attempts),
        "correct_citations": sum(c in shown and c in gold for c in strings),
        "grounded": correct and bool(strings) and set(strings) <= gold,
    }


def rationale_visible(
    displayed: dict[str, Any],
    annotated: dict[str, list[list[int]]],
    sentences: dict[str, list[str]],
) -> bool:
    doc_id = displayed["id"]
    body = " ".join(displayed["body"].split())
    return any(
        all(" ".join(sentences[doc_id][i].split()) in body for i in indices)
        for indices in annotated.get(doc_id, [])
    )


def trial(
    server: Server,
    claim: dict[str, Any],
    arm: str,
    ids: list[str],
    docs: dict[str, Any],
    sentences: dict[str, list[str]],
    cached: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    start = time.perf_counter()
    label, evidence = gold_evidence(claim)
    if len(ids) > 3 or len(set(ids)) != len(ids):
        raise ValueError("invalid context membership")
    displayed = []
    for doc_id in ids:
        if doc_id not in cached:
            doc = docs[doc_id]
            cached[doc_id] = server.document(doc_id, doc.title, doc.text)
        displayed.append(cached[doc_id])
    prompt_tokens, response = server.complete(claim["claim"], displayed)
    raw = "" if response is None else response["content"]
    evaluation = assess(raw, set(ids), set(evidence), label)
    if response is None:
        evaluation["status"] = "timeout"
    else:
        if response.get("tokens_evaluated") != prompt_tokens or response.get("truncated"):
            raise ValueError("server token accounting differs from the pinned prompt")
        if response.get("stop_type") == "limit":
            evaluation.update(status="invalid", correct=False, abstained=False, grounded=False)
    return {
        "qid": str(claim["id"]),
        "arm": arm,
        "gold_label": label,
        "presented": ids,
        "documents": [
            {k: doc[k] for k in ("id", "tokens", "untruncated_tokens")} for doc in displayed
        ],
        "evidence_present": bool(set(ids) & evidence.keys()),
        "rationale_visible": any(rationale_visible(d, evidence, sentences) for d in displayed),
        "prompt_tokens": prompt_tokens,
        "input_tokens": None if response is None else response.get("tokens_evaluated"),
        "output_tokens": None if response is None else response.get("tokens_predicted"),
        "stop_type": None if response is None else response.get("stop_type"),
        "seconds": time.perf_counter() - start,
        "raw_output": raw,
        **evaluation,
    }


def smoke() -> None:
    exe, model = verify_runtime(pins())
    train, _, sentences = source()
    docs = {doc.doc_id: doc for doc in ProcessedDataset(ROOT / "data", "scifact").corpus()}
    results = []
    with open_server(exe, model, OUT / "smoke-server.log") as server:
        for claim in sample(train, 1):
            for arm, ids in (("closed_book", []), ("oracle", sorted(gold_evidence(claim)[1])[:3])):
                result = trial(server, claim, arm, ids, docs, sentences, {})
                results.append(result)
                print(arm, result["status"], f"{result['seconds']:.2f}s", flush=True)
    estimate = max(row["seconds"] for row in results) * 240
    write_json(
        OUT / "smoke.json",
        {
            "estimated_worst_observed_seconds_times_240": estimate,
            "budget_seconds": SECONDS_BUDGET,
            "trials": results,
        },
    )
    if estimate > SECONDS_BUDGET or not any(row["status"] == "ok" for row in results):
        raise ValueError("CPU smoke did not meet the M7 gate; see outputs/m7/smoke.json")


def trial_path(qid: str, arm: str) -> Path:
    if not qid.isascii() or not qid.isdigit() or arm not in ARMS:
        raise ValueError("invalid trial key")
    return OUT / "trials" / f"{qid}-{arm}.json"


def read_trial(qid: str, arm: str, freeze_sha: str) -> dict[str, Any]:
    wrapped = read_json(trial_path(qid, arm))
    row = wrapped["record"]
    if (
        wrapped["freeze_sha256"] != freeze_sha
        or row["qid"] != qid
        or row["arm"] != arm
        or sha256_bytes(dumps_json(row)) != wrapped["record_sha256"]
    ):
        raise ValueError(f"cached trial failed verification: {qid}/{arm}")
    return row


def run() -> None:
    record = verified_freeze()
    freeze_sha = sha256_file(FREEZE)
    exe, model = verify_runtime(record["pins"])
    dataset, _ = inputs()
    _, dev, sentences = source()
    cohort = sample_ids(dev)
    if (
        len(cohort) != record["sample_size"]
        or sha256_bytes(dumps_json(cohort)) != record["sample_sha256"]
    ):
        raise ValueError("M7 cohort differs from freeze")
    ranked = contexts(record, cohort)
    by_claim = {str(claim["id"]): claim for claim in dev}
    write_json(OUT / "cohort.json", cohort)
    docs = {doc.doc_id: doc for doc in dataset.corpus()}
    expected = [(qid, arm) for qid in cohort for arm in ARMS]
    saved = {
        (qid, arm): read_trial(qid, arm, freeze_sha)
        for qid, arm in expected
        if trial_path(qid, arm).exists()
    }
    elapsed = sum(row["seconds"] for row in saved.values())
    started = time.perf_counter()
    cached: dict[str, dict[str, Any]] = {}
    write_json(
        OUT / "manifest.json",
        {
            "freeze_sha256": freeze_sha,
            "git_commit": git("rev-parse", "HEAD"),
            "hardware": {
                "system": platform.platform(),
                "machine": platform.machine(),
                "cpu": platform.processor(),
                "python": platform.python_version(),
            },
            "status": "incomplete",
        },
    )
    if len(saved) < len(expected):
        with open_server(exe, model, OUT / "server.log") as server:
            for qid, arm in expected:
                if (qid, arm) in saved:
                    continue
                if elapsed + time.perf_counter() - started >= SECONDS_BUDGET:
                    raise ValueError("M7 CPU budget exhausted; saved trials remain resumable")
                claim = by_claim[qid]
                if arm == "closed_book":
                    ids: list[str] = []
                elif arm == "oracle":
                    ids = sorted(gold_evidence(claim)[1])[:3]
                else:
                    ids = ranked[arm][qid][:3]
                row = trial(server, claim, arm, ids, docs, sentences, cached)
                write_json(
                    trial_path(qid, arm),
                    {
                        "freeze_sha256": freeze_sha,
                        "record_sha256": sha256_bytes(dumps_json(row)),
                        "record": row,
                    },
                )
                saved[qid, arm] = row
                if len(saved) % 12 == 0 or len(saved) == len(expected):
                    print(f"M7 {len(saved)}/{len(expected)} trials", flush=True)
    output = OUT / "trials.jsonl"
    atomic_write_bytes(output, dumps_jsonl(saved[key] for key in expected))
    manifest = read_json(OUT / "manifest.json")
    manifest.update(
        status="complete",
        n_trials=len(saved),
        trials_sha256=sha256_file(output),
        trial_seconds=sum(row["seconds"] for row in saved.values()),
        last_session_seconds=time.perf_counter() - started,
    )
    write_json(OUT / "manifest.json", manifest)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) != 240 or len({(r["qid"], r["arm"]) for r in rows}) != len(rows):
        raise ValueError("report requires exactly 240 unique trials")
    summary = {}
    by_arm = {arm: {r["qid"]: r for r in rows if r["arm"] == arm} for arm in ARMS}
    ids = by_arm["closed_book"].keys()
    if len(ids) != 60 or any(arm.keys() != ids for arm in by_arm.values()):
        raise ValueError("all four arms must cover the same 60 claims")
    labels = [row["gold_label"] for row in by_arm["closed_book"].values()]
    if any(labels.count(stance) != 30 for stance in GOLD):
        raise ValueError("cohort must include 30 claims per stance")
    for arm, trials in by_arm.items():
        values = list(trials.values())
        attempted = sum(r["attempted"] for r in values)
        summary[arm] = {
            "correct": sum(r["correct"] for r in values),
            "abstained": sum(r["abstained"] for r in values),
            "invalid": sum(r["status"] == "invalid" for r in values),
            "timeout": sum(r["status"] == "timeout" for r in values),
            "evidence": None
            if arm == "closed_book"
            else sum(r["evidence_present"] for r in values),
            "rationale": None
            if arm == "closed_book"
            else sum(r["rationale_visible"] for r in values),
            "grounded": None if arm == "closed_book" else sum(r["grounded"] for r in values),
            "citation_correct": sum(r["correct_citations"] for r in values),
            "citation_attempts": attempted,
            "documents_truncated": sum(
                d["untruncated_tokens"] > d["tokens"] for r in values for d in r["documents"]
            ),
        }
    paired = []
    for metric, arms in (("correct", ARMS), ("grounded", ARMS[1:])):
        for first, second in combinations(arms, 2):
            pairs = [(by_arm[first][qid][metric], by_arm[second][qid][metric]) for qid in ids]
            paired.append(
                {
                    "metric": metric,
                    "first": first,
                    "second": second,
                    "first_only": sum(a and not b for a, b in pairs),
                    "second_only": sum(b and not a for a, b in pairs),
                    "both": sum(a and b for a, b in pairs),
                    "neither": sum(not a and not b for a, b in pairs),
                }
            )
    return {"n": 60, "arms": summary, "paired": paired}


def report() -> None:
    manifest = read_json(OUT / "manifest.json")
    path = OUT / "trials.jsonl"
    if (
        manifest["status"] != "complete"
        or sha256_file(path) != manifest["trials_sha256"]
        or sha256_file(FREEZE) != manifest["freeze_sha256"]
    ):
        raise ValueError("M7 report requires a complete, verified trial artifact")
    rows = [loads_strict(line) for line in path.read_text(encoding="utf-8").splitlines()]
    data = summarize(rows)
    write_json(OUT / "summary.json", data)

    def rate(value: int | None) -> str:
        return "N/A" if value is None else f"{value}/60 ({value / 60:.1%})"

    lines = [
        "# GroundLab M7 — Scientific claim verification",
        "",
        "Local Qwen2.5-1.5B-Instruct Q4_K_M on CPU. The 60-claim SciFact evidence sample "
        "contains 30 SUPPORT and 30 CONTRADICT claims. The four arms use the same claims.",
        "",
        "| Arm | Stance accuracy | Grounded accuracy | Evidence present | Full rationale visible | "
        "Citation precision | Abstention | Invalid | Timeout |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        r = data["arms"][arm]
        attempted = r["citation_attempts"]
        precision = (
            "N/A"
            if not attempted
            else (f"{r['citation_correct']}/{attempted} ({r['citation_correct'] / attempted:.1%})")
        )
        lines.append(
            f"| {arm} | {rate(r['correct'])} | {rate(r['grounded'])} | {rate(r['evidence'])} | "
            f"{rate(r['rationale'])} | {precision} | {rate(r['abstained'])} | "
            f"{rate(r['invalid'])} | {rate(r['timeout'])} |"
        )
    lines.extend(
        [
            "",
            "## Paired outcomes",
            "",
            "| Metric | First | Second | First only | Second only | Both | Neither |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in data["paired"]:
        lines.append(
            f"| {row['metric']} | {row['first']} | {row['second']} | {row['first_only']} | "
            f"{row['second_only']} | {row['both']} | {row['neither']} |"
        )
    lines.extend(
        [
            "",
            "## Context and scope",
            "",
            "Document-level grounding requires a correct stance, at least one valid citation, and "
            "only presented documents annotated for the correct stance. "
            "Citation precision includes "
            "all attempted citation occurrences from parsed outputs, including invalid responses. "
            "A rationale is visible when all sentences in an annotated set are present "
            "after truncation.",
            "",
        ]
    )
    for arm in ARMS[1:]:
        lines.append(
            f"- {arm}: {data['arms'][arm]['documents_truncated']} truncated document occurrences."
        )
    lines.extend(
        [
            "",
            "The comparisons are descriptive. The balanced cohort represents evidence-bearing "
            "claims only. Labels can omit relevant documents. These outcomes are separate from "
            "the M6 retrieval inference.",
            "",
            f"Total recorded trial time: {manifest['trial_seconds']:.1f} seconds. "
            f"Run commit: {manifest['git_commit']}. Freeze SHA-256: {manifest['freeze_sha256']}.",
            "",
        ]
    )
    atomic_write_bytes(OUT / "REPORT.md", "\n".join(lines).encode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("smoke", "freeze", "run", "report"))
    command = parser.parse_args().command
    if command == "smoke":
        smoke()
    elif command == "freeze":
        smoke_result = read_json(OUT / "smoke.json")
        if smoke_result["estimated_worst_observed_seconds_times_240"] > SECONDS_BUDGET or not any(
            t["status"] == "ok" for t in smoke_result["trials"]
        ):
            raise ValueError("train-only smoke did not meet the CPU budget and validity gate")
        if FREEZE.exists():
            raise ValueError("M7 freeze already exists")
        write_json(FREEZE, frozen_record())
        print("Frozen 60 claim IDs and retrieval artifact hashes")
    elif command == "run":
        run()
    else:
        report()


if __name__ == "__main__":
    main()
