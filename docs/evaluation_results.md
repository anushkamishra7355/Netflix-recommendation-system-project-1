# Evaluation Results

Generated: 2026-06-14 02:48:37 UTC

## Methodology

Because the Netflix titles dataset does not include user watch history or explicit ratings, evaluation uses **genre overlap** as a proxy for relevance:

- **Relevant items**: titles sharing at least one Netflix category (`listed_in`) with the query title
- **Precision@K**: fraction of top-K recommendations that share a genre with the query
- **Recall@K**: fraction of all genre-relevant titles retrieved in top-K
- **F1@K**: harmonic mean of precision and recall
- **Hit Rate@K**: fraction of queries where at least one top-K recommendation shares a genre

Queries without genre metadata or without genre overlap candidates are skipped.

## Metrics (@K = 10)

| Metric | Score |
|--------|------:|
| Precision@K | 0.5169 |
| Recall@K | 0.0039 |
| F1@K | 0.0076 |
| Hit Rate@K | 0.9633 |

## Coverage

| Stat | Value |
|------|------:|
| Evaluated queries | 8,807 |
| Skipped queries | 0 |

## Interpretation

Content-based TF-IDF recommendations align moderately well with genre metadata. Precision and hit rate are the most actionable metrics for this proxy setup because the relevant set for popular genres can be large, which lowers recall.

Re-run evaluation after retraining:

```bash
python scripts/evaluate_model.py
```
