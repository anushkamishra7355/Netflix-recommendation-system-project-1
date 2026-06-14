"""Netflix content-based recommendation Streamlit application."""

from __future__ import annotations

import streamlit as st

from src.config import DEFAULT_TOP_N, PLACEHOLDER_POSTER
from src.poster import fetch_poster
from src.recommender import NetflixRecommender


@st.cache_resource(show_spinner="Loading recommendation model...")
def load_recommender() -> NetflixRecommender:
    return NetflixRecommender()


def render_recommendations(selected_title: str, top_n: int) -> None:
    try:
        recommender = load_recommender()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.info("Run `python scripts/train_model.py` locally to generate model files.")
        return
    except Exception as exc:
        st.error(f"Failed to load the recommendation model: {exc}")
        return

    try:
        with st.spinner(f"Finding titles similar to '{selected_title}'..."):
            recommendations = recommender.recommend(selected_title, top_n=top_n)
    except KeyError:
        st.error("That title is not available in the dataset. Please choose another one.")
        return
    except Exception as exc:
        st.error(f"Could not generate recommendations: {exc}")
        return

    st.subheader(f"Recommendations for '{selected_title}'")

    if not recommendations:
        st.warning("No similar titles were found.")
        return

    cols = st.columns(2)
    for index, (title, score) in enumerate(recommendations, start=1):
        column = cols[(index - 1) % 2]
        with column:
            with st.spinner(f"Loading poster for {title}..."):
                poster_url = fetch_poster(title)
            st.markdown(f"**{index}. {title}**")
            st.caption(f"Similarity score: {score:.3f}")
            if poster_url != PLACEHOLDER_POSTER:
                st.image(poster_url, width=150)
            else:
                st.info("Poster unavailable")


def main() -> None:
    st.set_page_config(
        page_title="Netflix Recommendation System",
        page_icon="🎬",
        layout="wide",
    )

    st.title("Netflix Recommendation System")
    st.markdown(
        "Discover Netflix movies and TV shows similar to a title you enjoy using "
        "content-based filtering with TF-IDF and cosine similarity."
    )

    try:
        with st.spinner("Initializing recommendation engine..."):
            recommender = load_recommender()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.info("Run `python scripts/train_model.py` to build model artifacts before starting the app.")
        st.stop()
    except Exception as exc:
        st.error(f"Unexpected error while loading the app: {exc}")
        st.stop()

    with st.sidebar:
        st.header("Settings")
        top_n = st.slider(
            "Number of recommendations",
            min_value=5,
            max_value=15,
            value=DEFAULT_TOP_N,
        )
        st.caption(f"{len(recommender.title_list):,} titles loaded")

    selected_title = st.selectbox(
        "What would you like to watch?",
        options=recommender.title_list,
        index=0,
    )

    if st.button("Recommend", type="primary"):
        render_recommendations(selected_title, top_n)


if __name__ == "__main__":
    main()
