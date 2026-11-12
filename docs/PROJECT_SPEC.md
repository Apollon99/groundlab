# Project specification

## Goal

Measure whether increasingly sophisticated retrieval (sparse, dense, hybrid, hybrid + rerank)
produces measurable gains on scientific claim retrieval, and whether those gains justify their
extra latency and resource use. The result must be reproducible from a clean checkout and simple
enough to defend in a technical interview.

## Research questions

1. Does dense retrieval beat BM25 on the primary view?
2. Does hybrid fusion beat the fixed dense comparator?
3. Does cross-encoder reranking of the hybrid list beat hybrid alone?
4. What does each step cost in latency (per query) and resources (GPU memory, index size)?

The three contrasts above (dense vs sparse, hybrid vs dense, rerank vs hybrid) are the
confirmatory family (Holm-corrected). Everything else is exploratory. See EXPERIMENT_PLAN.md.

## Arms (fixed, not tuned)

| Arm | Settings |
|---|---|
| BM25 | BM25S `method="lucene"`, k1=1.2, b=0.75; tokenizer `[^\W_]+`, Unicode casefold, English Snowball stemmer (PyStemmer), no stopwords |
| Dense | `BAAI/bge-small-en-v1.5` rev `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`, FP32, exact dot product, CUDA |
| Hybrid | RRF, k=60, over BM25 top-100 and dense top-100 |
| Rerank | `cross-encoder/ms-marco-MiniLM-L6-v2` rev `233902d25c440f23af6f7d6e94d2946bac0bee0a` over hybrid top-30, CUDA |

All rankers sort by score descending, then doc ID ascending (string order).
CUDA for embedding and reranking; CPU for BM25, fusion, metrics and statistics. No silent CPU fallback.

## Datasets and views

| Dataset / view | Role | Dev split | Test split (confirmatory) |
|---|---|---|---|
| SciFact evidence | primary | Ai2 train, evidence-bearing (505 claims, 564 pairs) | Ai2 dev = BEIR test, evidence-bearing (188 claims, 209 pairs) |
| SciFact BEIR qrels | secondary (sensitivity) | BEIR train (809 queries, 919 pairs) | BEIR test (300 queries, 339 pairs) |
| NFCorpus | secondary | BEIR dev (324 queries) | BEIR test (323 queries) |

- Relevance in the evidence view: a doc is relevant iff the claim has at least one non-empty
  evidence set for it. Stance is ignored. Unjudged docs are non-relevant.
- BEIR `corpus.jsonl` title and text are canonical for every arm.
- BEIR `queries.jsonl` `metadata` (a copy of the gold evidence) is dropped at load time.
- Ai2 `claims_test` has no public labels and is not used.

## Metrics

nDCG@10 (primary), Recall@10, Recall@100, MRR@10, per query, plus latency per query.
Conventions: see ARCHITECTURE.md "Metric conventions".

## Non-goals

No multi-agent frameworks, Kubernetes, distributed databases, complex frontends,
authentication, cloud deployment, hosted LLM dependencies, larger models, or mandatory
answer generation. Grounded generation, if added, is a separate optional milestone evaluated
independently from retrieval.
