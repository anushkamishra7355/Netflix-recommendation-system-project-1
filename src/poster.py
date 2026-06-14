"""Fetch movie posters from the OMDb API."""

from __future__ import annotations

import requests

from src.config import OMDB_BASE_URL, PLACEHOLDER_POSTER, get_omdb_api_key


def fetch_poster(
    movie_title: str,
    api_key: str | None = None,
    timeout: int = 5,
) -> str:
    """Return a poster URL for *movie_title*, or a placeholder on failure."""
    api_key = (api_key or get_omdb_api_key()).strip()
    if not api_key:
        return PLACEHOLDER_POSTER

    try:
        response = requests.get(
            OMDB_BASE_URL,
            params={"apikey": api_key, "t": movie_title},
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()
        if data.get("Response") == "True":
            poster_url = data.get("Poster")
            if poster_url and poster_url != "N/A":
                return poster_url
    except (requests.RequestException, ValueError):
        pass

    return PLACEHOLDER_POSTER
