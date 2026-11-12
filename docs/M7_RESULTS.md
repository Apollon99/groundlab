# GroundLab M7 — Scientific claim verification

Local Qwen2.5-1.5B-Instruct Q4_K_M on CPU. The 60-claim SciFact evidence sample contains 30 SUPPORT and 30 CONTRADICT claims. The four arms use the same claims.

| Arm | Stance accuracy | Grounded accuracy | Evidence present | Full rationale visible | Citation precision | Abstention | Invalid | Timeout |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| closed_book | 5/60 (8.3%) | N/A | N/A | N/A | 0/4 (0.0%) | 47/60 (78.3%) | 4/60 (6.7%) | 0/60 (0.0%) |
| bm25 | 37/60 (61.7%) | 18/60 (30.0%) | 51/60 (85.0%) | 51/60 (85.0%) | 46/81 (56.8%) | 0/60 (0.0%) | 0/60 (0.0%) | 0/60 (0.0%) |
| rerank | 34/60 (56.7%) | 21/60 (35.0%) | 55/60 (91.7%) | 55/60 (91.7%) | 51/81 (63.0%) | 0/60 (0.0%) | 0/60 (0.0%) | 0/60 (0.0%) |
| oracle | 31/60 (51.7%) | 31/60 (51.7%) | 60/60 (100.0%) | 60/60 (100.0%) | 64/64 (100.0%) | 0/60 (0.0%) | 0/60 (0.0%) | 0/60 (0.0%) |

## Paired outcomes

| Metric | First | Second | First only | Second only | Both | Neither |
|---|---|---|---:|---:|---:|---:|
| correct | closed_book | bm25 | 0 | 32 | 5 | 23 |
| correct | closed_book | rerank | 0 | 29 | 5 | 26 |
| correct | closed_book | oracle | 1 | 27 | 4 | 28 |
| correct | bm25 | rerank | 4 | 1 | 33 | 22 |
| correct | bm25 | oracle | 8 | 2 | 29 | 21 |
| correct | rerank | oracle | 4 | 1 | 30 | 25 |
| grounded | bm25 | rerank | 3 | 6 | 15 | 36 |
| grounded | bm25 | oracle | 2 | 15 | 16 | 27 |
| grounded | rerank | oracle | 2 | 12 | 19 | 27 |

## Context and scope

Document-level grounding requires a correct stance, at least one valid citation, and only presented documents annotated for the correct stance. Citation precision includes all attempted citation occurrences from parsed outputs, including invalid responses. A rationale is visible when all sentences in an annotated set are present after truncation.

- bm25: 5 truncated document occurrences.
- rerank: 6 truncated document occurrences.
- oracle: 4 truncated document occurrences.

The comparisons are descriptive. The balanced cohort represents evidence-bearing claims only. Labels can omit relevant documents. These outcomes are separate from the M6 retrieval inference.

Total recorded trial time: 909.8 seconds. Run commit: 2e85799af6bbf2749350657588ac2b6036026f96. Freeze SHA-256: 332a4d875e5ad440574969769b3c3b8714ad60fbe0bed5e27c742e69fd34b7a0.
