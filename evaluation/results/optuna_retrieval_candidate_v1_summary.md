# Optuna Retrieval Candidate v1

Frozen on 2026-09-29. This is the retrieval configuration that passed both the V3 development set and the V2 regression set.

## Frozen parameters

| Parameter | Value |
| --- | ---: |
| dense_k | 30 |
| lexical_k | 20 |
| rerank_k | 5 |
| final_k | 2 |
| rrf_k | 80 |
| dense_weight | 0.5 |
| lexical_weight | 1.5 |
| rerank_relative_threshold | 0.6 |
| rerank_absolute_threshold | 0.1 |
| rerank_min_score | 0.03 |
| rerank_tie_epsilon | 0.0001 |

## Verification results

| Dataset | Cases | Pass rate | Recall@1 | Recall@2 | MRR | No-answer accuracy | Failed cases |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V3 development | 80 | 100% | 100% | 100% | 100% | 100% | 0 |
| V2 regression | 60 | 100% | 100% | 100% | 100% | 100% | 0 |

The original V3 baseline pass rate was 85%. The frozen candidate reached 100% after Optuna search, rejection-threshold calibration, equivalent-document label correction, near-tie reranker handling, and asset-query recognition improvements.

## Holdout rule

The previous Holdout V1 has already been used and must not be used for further tuning. The next release decision must use a newly created, untouched Holdout V2 exactly once after the code and parameters are committed.
