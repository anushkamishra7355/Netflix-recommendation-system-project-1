#!/usr/bin/env python3
"""Train model artifacts from the Netflix titles dataset."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    DATASET_PATH,
    MODELS_DIR,
    TFIDF_MATRIX_ARTIFACT,
    TITLES_ARTIFACT,
    VECTORIZER_ARTIFACT,
)
from src.recommender import build_vectorizer, preprocess_netflix_dataframe


def main() -> None:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATASET_PATH)
    titles = preprocess_netflix_dataframe(df)

    vectorizer = build_vectorizer()
    tfidf_matrix = vectorizer.fit_transform(titles["combined"])

    with open(MODELS_DIR / TITLES_ARTIFACT, "wb") as handle:
        pickle.dump(titles, handle)

    with open(MODELS_DIR / TFIDF_MATRIX_ARTIFACT, "wb") as handle:
        pickle.dump(tfidf_matrix, handle)

    with open(MODELS_DIR / VECTORIZER_ARTIFACT, "wb") as handle:
        pickle.dump(vectorizer, handle)

    print(f"Saved artifacts for {len(titles)} titles to {MODELS_DIR}")


if __name__ == "__main__":
    main()
