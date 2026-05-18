"""
Music Recommendation System - Streamlit Web App
MAL Exam Project

Loads a pre-trained K-Means model and recommends songs based on audio similarity.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# Page setup
# ============================================================
st.set_page_config(
    page_title="Spotify Music Recommender",
    page_icon="🎵",
    layout="wide"
)

# ============================================================
# Load model and data (cached so it only loads once)
# ============================================================
@st.cache_resource
def load_model():
    with open('kmeans_model.pkl', 'rb') as f:
        kmeans = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return kmeans, scaler

@st.cache_data
def load_data():
    df = pd.read_csv('songs_for_app.csv')
    X_scaled = np.load('X_scaled.npz')['X']
    return df, X_scaled

kmeans, scaler = load_model()
df, X_scaled = load_data()

# ============================================================
# Recommendation function (same as in the notebook)
# ============================================================
def recommend_songs(song_name, df, X_scaled, n_recommendations=10):
    matches = df[df['name'].str.lower().str.contains(song_name.lower(), na=False)]

    if len(matches) == 0:
        return None, None

    song = matches.sort_values('popularity', ascending=False).iloc[0]
    song_idx = song.name

    cluster_id = song['cluster']
    cluster_mask = df['cluster'] == cluster_id
    cluster_indices = df[cluster_mask].index

    song_vector = X_scaled[song_idx].reshape(1, -1)
    cluster_vectors = X_scaled[cluster_indices]
    similarities = cosine_similarity(song_vector, cluster_vectors)[0]

    top_indices = np.argsort(similarities)[::-1][1:n_recommendations + 1]
    top_df_indices = cluster_indices[top_indices]

    recs = df.loc[top_df_indices, ['name', 'artists', 'year', 'popularity']].copy()
    recs['similarity'] = similarities[top_indices].round(3)
    recs = recs.reset_index(drop=True)
    recs.index = recs.index + 1  # start numbering at 1

    return song, recs

# ============================================================
# UI
# ============================================================
st.title("🎵 Spotify Music Recommender")
st.markdown(
    "Find songs similar to your favourites, powered by **K-Means clustering** "
    "and **cosine similarity** on Spotify audio features."
)

st.markdown("---")

# Search input
col1, col2 = st.columns([3, 1])
with col1:
    song_input = st.text_input(
        "Enter a song name:",
        placeholder="e.g. Bohemian Rhapsody, Hotel California, Smells Like Teen Spirit"
    )
with col2:
    n_recs = st.slider("Number of recommendations:", 5, 20, 10)

if song_input:
    song, recommendations = recommend_songs(song_input, df, X_scaled, n_recommendations=n_recs)

    if song is None:
        st.error(f"No song found matching '{song_input}'. Try another title.")
    else:
        # Show the matched song
        st.success(f"**Matched:** {song['name']} — {song['artists']}")
        st.caption(f"Cluster {song['cluster']} · Popularity {song['popularity']} · Year {song['year']}")

        st.markdown("### Recommended songs")
        st.dataframe(recommendations, width='stretch')

# ============================================================
# Sidebar - info about the project
# ============================================================
with st.sidebar:
    st.header("About")
    st.markdown(
        """
        This app recommends songs based on **audio features** like
        danceability, energy, acousticness, valence, and tempo.

        **How it works:**
        1. A K-Means model has grouped 160k songs into clusters by audio similarity
        2. When you enter a song, we find its cluster
        3. Within that cluster, we rank songs by cosine similarity
        4. Top matches are returned

        **Dataset:** [160k Spotify songs (Kaggle)](https://www.kaggle.com/datasets/fcpercival/160k-spotify-songs-sorted)

        **Tech:** Python · scikit-learn · Streamlit
        """
    )

    st.markdown("---")
    st.caption(f"Loaded {len(df):,} songs across {df['cluster'].nunique()} clusters.")