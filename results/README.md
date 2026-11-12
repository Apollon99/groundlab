# Aggregate results

Small, publishable summaries of the completed studies. They hold aggregate numbers only:
no local paths, raw data, model weights, query IDs or per-query records.

| File | Contents | Source |
|---|---|---|
| `retrieval_metrics.csv` | nDCG@10, Recall@10/100, MRR@10, latency p50/p90/p99 (ms), peak GPU bytes and index bytes for 4 arms × 3 views | `outputs/report/metrics.csv` |
| `retrieval_contrasts.csv` | The three primary paired nDCG@10 contrasts on SciFact evidence: estimate, 95% CI, raw and Holm p | `outputs/report/contrasts.csv` |
| `m7_arm_counts.csv` | M7 counts per arm over 60 claims (stance correct, grounded, evidence present, rationale visible, citations, abstention, invalid, timeout, truncated documents). Empty cells are not applicable to the closed-book arm | `outputs/m7/summary.json` |

## Provenance

The `outputs/...` sources and the commit `825704a` belong to the original local experiment
history. They are not part of this public snapshot, so the hashes below are provenance records
that can only be checked on the machine that holds those outputs.

- The two retrieval CSVs are byte-for-byte copies of the files written by `groundlab report`
  from the held-out runs at commit `825704a`. Regenerating the report from the saved runs
  produced identical files. SHA-256: `retrieval_metrics.csv`
  `5c556c0c4237e1866ff97e75bbc4956dc88f16cc18dcbb735171f7ae26d452ae`,
  `retrieval_contrasts.csv`
  `b99ea7928769e7fca4c050ce9d4b1d115e5277c20f026c3f5ad2ccc514b43ae6`. The rounded tables in
  [docs/RESULTS.md](../docs/RESULTS.md) come from the same data.
- `m7_arm_counts.csv` re-expresses the `arms` counts of the M7 `summary.json` (SHA-256
  `13c1c3079cc30b7313296093f39fcec8697266b6b5763c7d2adc4c8c6a7ab350`) as CSV. Rates in
  [docs/M7_RESULTS.md](../docs/M7_RESULTS.md) are these counts divided by 60, or by
  `citation_attempts` for citation precision.

Latency and GPU memory describe one machine and workload; see the main README.
