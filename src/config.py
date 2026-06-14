"""Central configuration for paths and runtime settings."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"

TITLES_ARTIFACT = "titles.pkl"
TFIDF_MATRIX_ARTIFACT = "tfidf_matrix.pkl"
VECTORIZER_ARTIFACT = "vectorizer.pkl"

TITLES_PATH = MODELS_DIR / TITLES_ARTIFACT
TFIDF_MATRIX_PATH = MODELS_DIR / TFIDF_MATRIX_ARTIFACT
VECTORIZER_PATH = MODELS_DIR / VECTORIZER_ARTIFACT
DATASET_PATH = DATA_DIR / "netflix_titles.csv"

ARTIFACT_CANDIDATES = (MODELS_DIR, PROJECT_ROOT)

OMDB_BASE_URL = "https://www.omdbapi.com/"
PLACEHOLDER_POSTER = "https://via.placeholder.com/300x450?text=No+Image"


def get_omdb_api_key() -> str:
    """Resolve the OMDb API key from env vars or Streamlit secrets."""
    key = os.environ.get("OMDB_API_KEY", "").strip()
    if key:
        return key

    try:
        import streamlit as st

        return str(st.secrets.get("OMDB_API_KEY", "")).strip()
    except Exception:
        return ""


# Backward-compatible module-level value for imports that read it directly.
OMDB_API_KEY = get_omdb_api_key()

DEFAULT_TOP_N = 10
EVAL_TOP_K = 10
