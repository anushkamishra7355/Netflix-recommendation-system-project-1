#!/usr/bin/env python3
"""Run offline evaluation and write results to docs/evaluation_results.md."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import EVAL_TOP_K
from src.evaluation import evaluate_recommender

DOCS_DIR = PROJECT_ROOT / "docs"
RESULTS_PATH = DOCS_DIR / "evaluation_results.md"
METRICS_JSON_PATH = DOCS_DIR / "evaluation_metrics.json"


def format_markdown(metrics: dict[str, float | int]) -> str:
    return f"""# Evaluation Results

Generated: {datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")}

## Methodology

Because the Netflix titles dataset does not include user watch history or explicit ratings, evaluation uses **genre overlap** as a proxy for relevance:

- **Relevant items**: titles sharing at least one Netflix category (`listed_in`) with the query title
- **Precision@K**: fraction of top-K recommendations that share a genre with the query
- **Recall@K**: fraction of all genre-relevant titles retrieved in top-K
- **F1@K**: harmonic mean of precision and recall
- **Hit Rate@K**: fraction of queries where at least one top-K recommendation shares a genre

Queries without genre metadata or without genre overlap candidates are skipped.

## Metrics (@K = {metrics["top_k"]})

| Metric | Score |
|--------|------:|
| Precision@K | {metrics["precision_at_k"]:.4f} |
| Recall@K | {metrics["recall_at_k"]:.4f} |
| F1@K | {metrics["f1_at_k"]:.4f} |
| Hit Rate@K | {metrics["hit_rate_at_k"]:.4f} |

## Coverage

| Stat | Value |
|------|------:|
| Evaluated queries | {metrics["evaluated_queries"]:,} |
| Skipped queries | {metrics["skipped_queries"]:,} |

## Interpretation

Content-based TF-IDF recommendations align moderately well with genre metadata. Precision and hit rate are the most actionable metrics for this proxy setup because the relevant set for popular genres can be large, which lowers recall.

Re-run evaluation after retraining:

```bash
python scripts/evaluate_model.py
```
"""


def main() -> None:
    metrics = evaluate_recommender(top_k=EVAL_TOP_K).as_dict()
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    RESULTS_PATH.write_text(format_markdown(metrics), encoding="utf-8")
    METRICS_JSON_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(json.dumps(metrics, indent=2))
    print(f"\nWrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
