# GroundLab held-out retrieval study

Four retrieval methods were evaluated against the frozen SciFact and NFCorpus views.
SciFact evidence is the primary view. The other views and secondary metrics are exploratory.

## Primary paired comparisons

188 claims in 164 evidence clusters. Effects are mean claim-level differences in nDCG@10. The 95% cluster-bootstrap intervals are marginal. Holm-adjusted p-values are from two-sided cluster sign flips.

| Contrast | Δ nDCG@10 | 95% CI | Raw p | Holm p |
|---|---:|---:|---:|---:|
| dense − bm25 | +0.0268 | [-0.0098, +0.0651] | 0.17008 | 0.51024 |
| hybrid − dense | +0.0079 | [-0.0180, +0.0348] | 0.56057 | 1 |
| rerank − hybrid | +0.0031 | [-0.0284, +0.0338] | 0.84558 | 1 |

A positive contrast with Holm p below 0.05 supports a gain under the sign-flip assumptions. An observed gain of 0.02 is the descriptive practical threshold. These tests compare against zero.

## Retrieval quality

| View | Arm | Queries | nDCG@10 | Recall@10 | Recall@100 | MRR@10 |
|---|---|---:|---:|---:|---:|---:|
| scifact_evidence | bm25 | 188 | 0.8370 | 0.9384 | 0.9739 | 0.8079 |
| scifact_evidence | dense | 188 | 0.8638 | 0.9569 | 0.9947 | 0.8400 |
| scifact_evidence | hybrid | 188 | 0.8717 | 0.9649 | 0.9947 | 0.8475 |
| scifact_evidence | rerank | 188 | 0.8749 | 0.9578 | 0.9947 | 0.8545 |
| scifact_beir | bm25 | 300 | 0.6849 | 0.8082 | 0.9213 | 0.6518 |
| scifact_beir | dense | 300 | 0.7127 | 0.8362 | 0.9417 | 0.6822 |
| scifact_beir | hybrid | 300 | 0.7280 | 0.8546 | 0.9650 | 0.6943 |
| scifact_beir | rerank | 300 | 0.7007 | 0.8372 | 0.9650 | 0.6670 |
| nfcorpus | bm25 | 323 | 0.3224 | 0.1550 | 0.2475 | 0.5230 |
| nfcorpus | dense | 323 | 0.3438 | 0.1620 | 0.3110 | 0.5302 |
| nfcorpus | hybrid | 323 | 0.3640 | 0.1735 | 0.3119 | 0.5688 |
| nfcorpus | rerank | 323 | 0.3587 | 0.1668 | 0.3119 | 0.5868 |

## Latency and resources

Timings cover warmed, single-query retrieval through ranking on this machine. They exclude loading and use one observation per query. GPU memory is PyTorch peak allocated memory, including resident weights. Index size excludes model files.

| View | Arm | p50 ms | p90 ms | p99 ms | GPU peak MiB | Index MiB |
|---|---|---:|---:|---:|---:|---:|
| scifact_evidence | bm25 | 0.36 | 0.44 | 0.57 | — | 5.26 |
| scifact_evidence | dense | 6.76 | 9.79 | 11.24 | 137.8 | 7.66 |
| scifact_evidence | hybrid | 8.12 | 11.49 | 12.59 | 137.8 | 12.92 |
| scifact_evidence | rerank | 122.08 | 132.14 | 139.58 | 1085.9 | 12.92 |
| scifact_beir | bm25 | 0.35 | 0.40 | 0.51 | — | 5.26 |
| scifact_beir | dense | 6.48 | 7.65 | 10.41 | 137.8 | 7.66 |
| scifact_beir | hybrid | 8.26 | 9.27 | 11.33 | 137.8 | 12.92 |
| scifact_beir | rerank | 121.77 | 133.09 | 143.15 | 1085.9 | 12.92 |
| nfcorpus | bm25 | 0.10 | 0.29 | 0.34 | — | 3.97 |
| nfcorpus | dense | 6.83 | 8.20 | 11.75 | 137.0 | 5.37 |
| nfcorpus | hybrid | 7.57 | 8.90 | 10.05 | 137.0 | 9.35 |
| nfcorpus | rerank | 123.27 | 130.94 | 138.27 | 1085.9 | 9.35 |

## Candidate integrity

- **scifact_evidence**. The top-30 membership and tail were preserved for all 188 queries. The head was reordered for 188; BM25 returned no hits for 0.
- **scifact_beir**. The top-30 membership and tail were preserved for all 300 queries. The head was reordered for 300; BM25 returned no hits for 0.
- **nfcorpus**. The top-30 membership and tail were preserved for all 323 queries. The head was reordered for 323; BM25 returned no hits for 15.

## Scope and reproducibility

Evidence judgments are incomplete. Frozen pretrained models can have seen benchmark material. The timing results characterize the recorded hardware and workload. The statistical test assumes independent clusters with a joint sign-symmetric null.

Run commit(s): 825704a104a722ccca396294706cac0924420559. See study.json for artifact hashes and freeze records, metrics.csv for all metric rows, and contrasts.csv for all primary contrasts.
