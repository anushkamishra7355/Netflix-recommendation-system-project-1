"""Content-based recommendation engine using TF-IDF and cosine similarity."""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import ARTIFACT_CANDIDATES


def artifact_path(name: str) -> Path:
    """Resolve an artifact path from known deployment-safe locations."""
    for directory in ARTIFACT_CANDIDATES:
        candidate = directory / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Required artifact '{name}' not found. Run: python scripts/train_model.py"
    )


def load_pickle(name: str):
    with open(artifact_path(name), "rb") as handle:
        return pickle.load(handle)


class NetflixRecommender:
    """Load preprocessed artifacts and recommend similar titles."""

    def __init__(self) -> None:
        titles_obj = load_pickle("titles.pkl")
        if isinstance(titles_obj, pd.DataFrame):
            self.titles = titles_obj.reset_index(drop=True)
        else:
            self.titles = pd.DataFrame(titles_obj).reset_index(drop=True)

        self.tfidf_matrix = load_pickle("tfidf_matrix.pkl")
        self.title_to_idx = {
            title: idx for idx, title in enumerate(self.titles["title"])
        }

    @property
    def title_list(self) -> list[str]:
        return self.titles["title"].tolist()

    def recommend(self, title: str, top_n: int = 10) -> list[tuple[str, float]]:
        if title not in self.title_to_idx:
            raise KeyError(f"Title not found in dataset: {title}")

        idx = self.title_to_idx[title]
        similarity_scores = cosine_similarity(
            self.tfidf_matrix[idx], self.tfidf_matrix
        ).flatten()

        ranked_indices = similarity_scores.argsort()[::-1]
        recommendations: list[tuple[str, float]] = []

        for candidate_idx in ranked_indices:
            if candidate_idx == idx:
                continue
            recommendations.append(
                (
                    self.titles["title"].iloc[candidate_idx],
                    float(similarity_scores[candidate_idx]),
                )
            )
            if len(recommendations) == top_n:
                break

        return recommendations


def preprocess_netflix_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same cleaning steps used in the training notebook."""
    cleaned = df.copy()
    cleaned["country"] = cleaned["country"].fillna("United States")
    cleaned["combined"] = (
        cleaned["description"].fillna("")
        + " "
        + cleaned["cast"].fillna("")
        + " "
        + cleaned["director"].fillna("")
    )
    cleaned = cleaned.drop_duplicates(subset="title", keep="first").reset_index(drop=True)
    return cleaned


def build_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        min_df=3,
        max_features=None,
        analyzer="word",
        ngram_range=(1, 3),
        stop_words="english",
    )
