"""Offline evaluation metrics for the content-based recommender."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.config import EVAL_TOP_K
from src.recommender import NetflixRecommender


def parse_genres(listed_in: str | float) -> set[str]:
    if pd.isna(listed_in):
        return set()
    return {genre.strip() for genre in str(listed_in).split(",") if genre.strip()}


def build_genre_sets(titles: pd.DataFrame) -> list[set[str]]:
    return [parse_genres(value) for value in titles["listed_in"]]


def relevant_indices(genre_sets: list[set[str]], query_idx: int) -> set[int]:
    query_genres = genre_sets[query_idx]
    if not query_genres:
        return set()

    relevant: set[int] = set()
    for idx, genres in enumerate(genre_sets):
        if idx == query_idx:
            continue
        if query_genres & genres:
            relevant.add(idx)
    return relevant


def precision_at_k(recommended: set[int], relevant: set[int], k: int) -> float:
    if k == 0:
        return 0.0
    hits = len(recommended & relevant)
    return hits / k


def recall_at_k(recommended: set[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    hits = len(recommended & relevant)
    return hits / len(relevant)


def f1_at_k(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def hit_rate_at_k(recommended: set[int], relevant: set[int]) -> float:
    return 1.0 if recommended & relevant else 0.0


@dataclass(frozen=True)
class EvaluationMetrics:
    precision_at_k: float
    recall_at_k: float
    f1_at_k: float
    hit_rate_at_k: float
    evaluated_queries: int
    skipped_queries: int
    top_k: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "precision_at_k": self.precision_at_k,
            "recall_at_k": self.recall_at_k,
            "f1_at_k": self.f1_at_k,
            "hit_rate_at_k": self.hit_rate_at_k,
            "evaluated_queries": self.evaluated_queries,
            "skipped_queries": self.skipped_queries,
            "top_k": self.top_k,
        }


def evaluate_recommender(
    recommender: NetflixRecommender | None = None,
    top_k: int = EVAL_TOP_K,
    sample_size: int | None = None,
    random_state: int = 42,
) -> EvaluationMetrics:
    """Evaluate recommendations using genre overlap as a relevance proxy."""
    recommender = recommender or NetflixRecommender()
    genre_sets = build_genre_sets(recommender.titles)

    query_indices = [
        idx for idx, genres in enumerate(genre_sets) if genres and relevant_indices(genre_sets, idx)
    ]

    if sample_size is not None and sample_size < len(query_indices):
        query_indices = (
            pd.Series(query_indices)
            .sample(n=sample_size, random_state=random_state)
            .tolist()
        )

    if not query_indices:
        return EvaluationMetrics(0.0, 0.0, 0.0, 0.0, 0, len(recommender.titles), top_k)

    precisions: list[float] = []
    recalls: list[float] = []
    f1_scores: list[float] = []
    hits: list[float] = []

    for query_idx in query_indices:
        title = recommender.titles["title"].iloc[query_idx]
        relevant = relevant_indices(genre_sets, query_idx)
        recommendations = recommender.recommend(title, top_n=top_k)
        recommended_indices = {
            recommender.title_to_idx[recommended_title]
            for recommended_title, _ in recommendations
        }

        precision = precision_at_k(recommended_indices, relevant, top_k)
        recall = recall_at_k(recommended_indices, relevant, top_k)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1_at_k(precision, recall))
        hits.append(hit_rate_at_k(recommended_indices, relevant))

    skipped = len(recommender.titles) - len(query_indices)
    return EvaluationMetrics(
        precision_at_k=float(sum(precisions) / len(precisions)),
        recall_at_k=float(sum(recalls) / len(recalls)),
        f1_at_k=float(sum(f1_scores) / len(f1_scores)),
        hit_rate_at_k=float(sum(hits) / len(hits)),
        evaluated_queries=len(query_indices),
        skipped_queries=skipped,
        top_k=top_k,
    )
