
'''import streamlit as st
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import sigmoid_kernel

# Load preprocessed data
similarity = pickle.load(open('similarity.pkl', 'rb'))
title_dict = pickle.load(open('title.pkl', 'rb'))
titles = pd.DataFrame(title_dict)

# Precompute sigmoid kernel similarity matrix
tfv = TfidfVectorizer(stop_words='english')  # Create a TF-IDF Vectorizer
tfv_matrix = tfv.fit_transform(titles['combined'])  # Apply TF-IDF to 'combined' column
sig = sigmoid_kernel(tfv_matrix, tfv_matrix)  # Calculate sigmoid kernel similarity

# Create a mapping of movie titles to their indices
indices = pd.Series(titles.index, index=titles['title']).drop_duplicates()

# Recommendation function
def recommend(title, sig=sig):
    """
    Recommend movies based on the sigmoid similarity kernel.

    Args:
        title (str): The title of the movie to recommend similar items for.
        sig (array-like): Precomputed similarity matrix.

    Returns:
        list: List of recommended movie titles.
    """
    try:
        # Get the index of the movie
        idx = indices[title]
    except KeyError:
        return ["Title not found in dataset!"]

    # Get similarity scores for all movies with the given movie
    sig_scores = list(enumerate(sig[idx]))

    # Sort movies based on similarity scores
    sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)

    # Get the indices of the top 10 most similar movies (excluding the current one)
    sig_scores = sig_scores[1:11]  # Skip the first, which is the selected movie
    movie_indices = [i[0] for i in sig_scores]

    # Return the top 10 recommended movie titles
    return titles['title'].iloc[movie_indices].tolist()

# Streamlit UI
st.title("Netflix Recommendation System")

# Dropdown for selecting a movie title
selected_title = st.selectbox(
    'What would you like to watch?',
    titles['title'].values
)

# Display recommendations
if st.button('Recommend'):
    recommendations = recommend(selected_title)
    st.write(f"Recommendations for '{selected_title}':")
    for i, rec in enumerate(recommendations, 1):
        st.write(f"{i}. {rec}")'''
import streamlit as st
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import sigmoid_kernel
import requests  # For fetching movie posters from OMDb API

# Load preprocessed data
similarity = pickle.load(open('similarity.pkl', 'rb'))
title_dict = pickle.load(open('title.pkl', 'rb'))
titles = pd.DataFrame(title_dict)

# Precompute sigmoid kernel similarity matrix
tfv = TfidfVectorizer(stop_words='english')  # Create a TF-IDF Vectorizer
tfv_matrix = tfv.fit_transform(titles['combined'])  # Apply TF-IDF to 'combined' column
sig = sigmoid_kernel(tfv_matrix, tfv_matrix)  # Calculate sigmoid kernel similarity

# Create a mapping of movie titles to their indices
indices = pd.Series(titles.index, index=titles['title']).drop_duplicates()

# OMDb API details
OMDB_API_KEY = 'c6d230a3'  # Replace with your actual OMDb API key
OMDB_BASE_URL = "http://www.omdbapi.com/"

def fetch_poster(movie_title):
    """
    Fetch the poster URL for a given movie title using OMDb API.

    Args:
        movie_title (str): Title of the movie.

    Returns:
        str: URL of the movie poster or a placeholder image if not found.
    """
    search_url = OMDB_BASE_URL
    params = {
        "apikey": OMDB_API_KEY,
        "t": movie_title  # 't' is for title-based search
    }
    response = requests.get(search_url, params=params)
    if response.status_code == 200:
        data = response.json()
        if data['Response'] == 'True':  # Check if the request was successful
            poster_url = data.get('Poster')
            if poster_url and poster_url != 'N/A':  # Check if poster is available
                return poster_url
    # Return a placeholder image if no poster is found
    return "https://via.placeholder.com/300x450?text=No+Image"

# Recommendation function
def recommend(title, sig=sig):
    """
    Recommend movies based on the sigmoid similarity kernel.

    Args:
        title (str): The title of the movie to recommend similar items for.
        sig (array-like): Precomputed similarity matrix.

    Returns:
        list: List of tuples (movie title, poster URL).
    """
    try:
        # Get the index of the movie
        idx = indices[title]
    except KeyError:
        return [("Title not found in dataset!", "https://via.placeholder.com/300x450?text=No+Image")]

    # Get similarity scores for all movies with the given movie
    sig_scores = list(enumerate(sig[idx]))

    # Sort movies based on similarity scores
    sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)

    # Get the indices of the top 10 most similar movies (excluding the current one)
    sig_scores = sig_scores[1:11]  # Skip the first, which is the selected movie
    movie_indices = [i[0] for i in sig_scores]

    # Return the top 10 recommended movie titles and their posters
    recommendations = []
    for i in movie_indices:
        movie_title = titles['title'].iloc[i]
        poster_url = fetch_poster(movie_title)
        recommendations.append((movie_title, poster_url))
    return recommendations

# Streamlit UI
st.title("Netflix Recommendation System")

# Dropdown for selecting a movie title
selected_title = st.selectbox(
    'What would you like to watch?',
    titles['title'].values
)

# Display recommendations with posters
if st.button('Recommend'):
    recommendations = recommend(selected_title)
    st.write(f"Recommendations for '{selected_title}':")
    for rec_title, poster_url in recommendations:
        st.write(f"**{rec_title}**")
        st.image(poster_url, width=150)


