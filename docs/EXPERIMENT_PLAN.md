# Frozen experiment plan

## Confirmatory analysis

SciFact evidence is the primary view. The metric is per-claim nDCG@10 for the 188
evidence-bearing claims. The effect estimate is the arithmetic mean of paired
claim differences. Evidence-sharing claims form connected components, including
transitive links. The M0 audit showed that evidence-only and BEIR citation-based
linkage give the same 164 clusters (144 singletons). The evaluation verifies this
partition against both judgment views.

The three fixed contrasts are dense − BM25, hybrid − dense, and rerank − hybrid.
Dense is the fixed semantic comparator for hybrid, selected before held-out
scoring. Dev summaries were already available. The comparator is fixed by design
and is never selected from test rankings. The result addresses hybrid versus this
comparator. Hybrid versus BM25 can be read descriptively from the metric table.

Test each contrast against zero with a two-sided sign flip of independent cluster
sums and inclusive tails. This relies on invariance to flipping entire clusters
under the null. Use exact enumeration for up to 16 nonzero cluster sums and
otherwise 99,999 draws with the observed assignment included as a plus-one
correction. Apply Holm adjustment across the three tests at familywise α = 0.05.
Report marginal 95% percentile intervals from 10,000 paired cluster bootstrap
resamples, weighting each resample by its number of claims. Random seeds and
procedures live in the pinned analysis code.

Report directions, estimates, intervals, and adjusted p-values regardless of
outcome. An observed gain of 0.02 nDCG@10 marks practical interest. It is a
descriptive threshold. The shifted margin sign-flip simulation is exploratory
and does not promote a margin hypothesis into the confirmatory family.

## Pre-freeze checks and exploratory views

Validate the statistical code on synthetic fixtures and simulate size, power at
+0.02, and sensitivity to asymmetric differences. Dev data can be used for
integration, repeatability, and resource profiling. Retrieval settings stay fixed.
The SciFact BEIR and NFCorpus views, Recall@10/100, and MRR@10 are exploratory.
Check fused candidate integrity in every view. Use a warmed single-query timing
pass with 10 warm-up queries and one observation per query for each arm. Record
p50/p90/p99, peak PyTorch allocated/reserved GPU memory, and parent index sizes.
Pin OMP/MKL/OpenBLAS threads to four in the local held-out runner. The latency
measurements characterize one machine and workload.

## Freeze and output handling

1. Commit tested code, configs and this protocol before held-out scoring.
2. Run the freeze command from verified offline inputs. Commit the 12 manifests.
   Each binds the analysis code tree, effective arm configuration and parents,
   dependency lock, prepared dataset manifest, and this protocol.
3. Run the frozen held-out workflow. Keep raw rankings, item-level metrics,
   manifests and report artifacts under ignored local outputs. Preserve the
   run and evaluation inputs so the report can be generated offline.
4. Record any protocol amendment and its timing. The frozen guard rejects changes
   to its bound inputs. Report cross-machine variability without bit-identity claims.

Methods: [Phipson & Smyth, permutation p-values](https://doi.org/10.2202/1544-6115.1585)
and [Holm, sequentially rejective multiple testing](https://www.jstor.org/stable/4615733).
