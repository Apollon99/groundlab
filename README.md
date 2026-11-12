# GroundLab

GroundLab is a reproducible benchmark that asks one question: **does more sophisticated
retrieval produce measurable gains, and are those gains worth the extra latency and compute?**

It compares four retrieval arms on the same data, split and metrics:

| Arm | Description | Status |
|---|---|---|
| Sparse | BM25 (BM25S, Lucene variant, k1=1.2, b=0.75) | implemented (M1) |
| Dense | `BAAI/bge-small-en-v1.5` (pinned), CUDA FP32, exact dot product | implemented (M2) |
| Hybrid | Reciprocal rank fusion (k=60) of sparse + dense top-100 | implemented and dev-verified (M3) |
| Hybrid + rerank | `cross-encoder/ms-marco-MiniLM-L6-v2` over the hybrid top-30 | implemented and dev-verified (M3) |

The separate [M7 claim verification study](docs/M7_RESULTS.md) evaluates a pinned local generator on 60 balanced SciFact claims. Its [protocol](experiments/m7/PROTOCOL.md) and freeze are independent of retrieval.

**Primary dataset:** SciFact evidence view (Ai2 dev = BEIR test, 188 evidence-bearing claims,
209 relevant pairs). **Secondary:** SciFact with BEIR qrels, and NFCorpus.
See [docs/PROJECT_SPEC.md](docs/PROJECT_SPEC.md) and [docs/EXPERIMENT_PLAN.md](docs/EXPERIMENT_PLAN.md).

> **Held-out results.** See [docs/RESULTS.md](docs/RESULTS.md). The experiment uses committed
> freeze manifests. The report is regenerated offline from saved local run artifacts, which are not part of this
> public snapshot.

## What this public snapshot contains

This repository is a sanitized public snapshot that starts with a single clean commit. It
includes the code, configs, frozen protocols and checksums (`freeze/`, `experiments/m7/`),
`uv.lock`, tests on synthetic fixtures, and the aggregate reports
([docs/RESULTS.md](docs/RESULTS.md), [docs/M7_RESULTS.md](docs/M7_RESULTS.md)).

It does **not** include datasets, model weights, local reports, manifests or runs
(`data/`, `artifacts/`, `outputs/`), or per-query records. The commands below therefore
recreate the retrieval workflow locally by downloading the data and models from their published
sources at pinned hashes; they cannot be run from the snapshot alone. Replaying the original M7
artifacts is a separate matter with stricter requirements, described in "M7 report and
reproduction" below.

Commit hashes printed in the reports and in [docs/HANDOFF.md](docs/HANDOFF.md) are identifiers
from the original local experiment history. They are kept as provenance references and cannot
be browsed in this repository.

The published aggregate reports are readable as they are. Re-running the retrieval study from a
fresh checkout needs the prepared data and pinned models, and (as noted in the M7 section below)
may produce different timing and provenance bytes than the archived results.

## Requirements

- Windows 10/11 x64 (native, no WSL2) or Linux x64
- [uv](https://docs.astral.sh/uv/) 0.12 or newer (installs CPython 3.11 for you)
- For the dense and rerank arms: an NVIDIA GPU with a CUDA 12.8-capable driver. BM25,
  fusion, metrics and statistics run on CPU. There is no silent CPU fallback for GPU stages.
- The CUDA path was verified on Windows only. Linux x64 is a supported platform target, but
  the CUDA stages were not verified there.

## Commands

All commands are PowerShell, run from the repository root.

```powershell
# 1. Environment setup (creates .venv from uv.lock, records environment metadata,
#    downloads the pinned model files and verifies their SHA-256)
.\scripts\setup.ps1
#    Re-verify the models without network access:
.\.venv\Scripts\groundlab.exe models fetch --model bge-small-en-v1.5 --verify-only

# 2. Data preparation (downloads, verifies pinned SHA-256, builds views, audits counts)
.\.venv\Scripts\groundlab.exe data prepare --dataset scifact --data-dir data
.\.venv\Scripts\groundlab.exe data prepare --dataset nfcorpus --data-dir data
#    Re-verify later without downloading or writing anything:
.\.venv\Scripts\groundlab.exe data prepare --dataset scifact --data-dir data --verify-only

# 3. Index construction (BM25 index on CPU; dense embedding cache on CUDA)
.\.venv\Scripts\groundlab.exe build --config configs\bm25_scifact_evidence.toml
.\.venv\Scripts\groundlab.exe build --config configs\dense_scifact_evidence.toml

# 4. Baseline execution (dev split only until the freeze)
.\.venv\Scripts\groundlab.exe run --config configs\bm25_scifact_evidence.toml --split dev
.\.venv\Scripts\groundlab.exe evaluate --config configs\bm25_scifact_evidence.toml --split dev

#    Dense: same commands with configs\dense_*.toml (records latency and GPU memory)

# 4b. Hybrid and hybrid+rerank (CUDA). They reuse the BM25 and dense indexes of step 3, so build
#     the two parents of a view first. `build` of a hybrid/rerank config builds nothing: it
#     verifies the parent indexes (and the reranker files). Parent indexes are read from
#     artifacts\indexes (override with --index-root).
.\.venv\Scripts\groundlab.exe models fetch --model ms-marco-MiniLM-L6-v2 --verify-only
.\.venv\Scripts\groundlab.exe build --config configs\hybrid_scifact_evidence.toml
.\.venv\Scripts\groundlab.exe run --config configs\hybrid_scifact_evidence.toml --split dev
.\.venv\Scripts\groundlab.exe evaluate --config configs\hybrid_scifact_evidence.toml --split dev
.\.venv\Scripts\groundlab.exe build --config configs\rerank_scifact_evidence.toml
.\.venv\Scripts\groundlab.exe run --config configs\rerank_scifact_evidence.toml --split dev
.\.venv\Scripts\groundlab.exe evaluate --config configs\rerank_scifact_evidence.toml --split dev
#    Other views: hybrid_/rerank_ + scifact_beir or nfcorpus. Outputs: outputs\runs\<name>\dev\.

# 5. Validate the statistical method on synthetic data
.\.venv\Scripts\groundlab.exe simulate --out outputs\m4\simulation.json

# 6. Held-out study (requires committed freeze manifests and prepared inputs).
#    To create a freeze for this study, run groundlab freeze and commit freeze/*.json first.
.\scripts\held_out.ps1
#    Regenerate the report from saved artifacts without running retrieval:
.\.venv\Scripts\groundlab.exe report

# 7. Tests, lint, types
.\scripts\test.ps1
```

Tests use synthetic fixtures only. Tests marked `pinned_models` need the real pinned model
files and skip with an explicit reason if they are absent; tests marked `cuda` need a CUDA
GPU. Run the setup step first to enable the model-dependent tests, and read the skip
report (`-rs`) to see what was not run. A passing run without the models or a GPU does not
cover those paths.

`scripts\dev_smoke.ps1` runs steps 2 to 4b for all twelve configs (three each of BM25, dense,
hybrid, rerank) on the dev split.

Hybrid and rerank design in short (details: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md),
D-027 to D-030): RRF with k=60 over up to 100 real hits per parent, exact arithmetic, ties by doc
ID; the cross-encoder scores only the first 30 fused candidates and reorders only them. The
rerank run file carries rank-derived scores (strictly descending); evaluation reads the rank
column. Limits: the reranker sees at most 512 tokens per pair (long abstracts are truncated),
latency is one timing per query on one occasion.

Exit codes: `0` ok, `1` error, `2` usage/config error, `3` verification failed (hash,
count or metric cross-check), `4` refused (test split without a freeze), `5` CUDA required
but unavailable.

## Repository layout

```
configs/      experiment configs (TOML, strict schema)
docs/         spec, architecture, plan, decisions, status, handoff, M0 data audit
scripts/      PowerShell entry points
src/groundlab/
  cli.py        command-line interface
  config.py     strict config loader
  envinfo.py    environment metadata
  freeze.py     test-split guard
  data/         pinned specs, archive verification, view building, loader
  retrieval/    tokenizer, BM25, dense, hybrid (RRF), rerank
  evaluation/   native metrics, pytrec_eval cross-check
tests/        unit and integration tests on synthetic fixtures only
data/ artifacts/ outputs/   created locally, never committed
```

## Data and licenses

GroundLab's own source code and documentation are released under the [MIT License](LICENSE).
The MIT License does **not** cover third-party datasets (SciFact, BEIR, NFCorpus) or
pretrained model weights; those keep their own licenses and terms. They are downloaded from
their original sources at pinned hashes and are never committed. See [NOTICES.md](NOTICES.md).

## Windows note: non-ASCII repository paths

Python 3.11 decodes `.pth` files with the locale codec, so a standard editable install
silently fails to import when the repository path contains non-ASCII characters (for example
a Windows folder name with accented letters). The project therefore uses hatchling's `dev-mode-exact`
editable mode, which needs the `editables` dev dependency (D-007). Always install with
`uv sync --locked` (dev dependencies included); `uv sync --no-dev` would break imports.

## M7 report and reproduction

The public M7 aggregate report, [docs/M7_RESULTS.md](docs/M7_RESULTS.md), can be read directly.

The command below regenerates it, and is meant for users who hold the saved local M7 trials
and manifest in `outputs/m7/` (not included in this snapshot):

```powershell
.\.venv\Scripts\python.exe experiments\m7\runner.py report
```

The exact original M7 run or replay needs the private original M6 `study.json` and exact
rankings matching the frozen hashes, plus the local runtime and model weights. Rerunning M6
from a fresh checkout can produce different timing and provenance bytes, so it will not
necessarily satisfy those hashes.

To rerun or resume the frozen inference under those conditions, use `runner.py run` with the
original M6 artifacts, the pinned model file and the verified extracted llama.cpp
runtime. `experiments/m7/pins.json` records the exact source versions and hashes.
The model is available at its pinned Hugging Face revision. Extract the matching
llama.cpp Windows CPU archive to its pinned runtime directory after checking
the archive SHA-256. The runner verifies the executable, DLLs, dataset, prompt,
and retrieval artifacts before processing any trial.
