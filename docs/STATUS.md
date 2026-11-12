# Status

Last updated: 2026-11-12 (M7 complete).

| Milestone | Scope | State |
|---|---|---|
| M0 | Read-only data audit (hashes, mappings, counts, overlap, licenses) | Done (docs/DATA_AUDIT_M0.md) |
| M1 | Repo, docs, locked env, env check, data prepare + audit, BM25 build/run/evaluate on dev, metrics with pytrec_eval cross-check, split guard, tests | **Accepted (PASS) by the project owner on `4e667f0`, 2026-11-12**, after an independent review (PASS WITH FIXES) and two review rounds |
| M2 | Dense arm: CUDA BGE embeddings, verified cache, exact search and profiling | Independent PASS at `2d65d5c`; the project owner authorized M3 |
| M3 | Hybrid RRF and CUDA cross-encoder reranking | Independent PASS at `e3b5be4`; included in the completed frozen study |
| M4 | Cluster-aware paired analysis, Holm and simulation | Complete before freeze. 219 tests passed. See the M4 handoff and the frozen experiment plan |
| M5 | Study freeze and held-out evaluation | Complete. Twelve runs at `825704a`, with four arms across three views |
| M6 | Report generation | Complete. Markdown, CSV and JSON reproduced byte for byte offline |
| M7 | Separate local claim verification | Complete. 240 paired trials, [M7 report](M7_RESULTS.md) |

## Current state

- The report is [RESULTS.md](RESULTS.md). The analysis protocol and all twelve experiment
  manifests were committed before held-out scoring.
- The held-out report is reproducible from the saved runs under `outputs/runs/`.
  Local reports and per-query artifacts remain in the ignored outputs directory.
- M7 used a separate freeze and CPU generation. See the M7 report and protocol.
