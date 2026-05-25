import streamlit as st
import numpy as np
import pickle
from pathlib import Path
from tensorflow import keras


# Page setup
st.set_page_config(
    page_title="Spotify Popularity Predictor",
    page_icon="🎵",
    layout="wide"
)

HERE = Path(__file__).parent


# Load model and scaler (cached)
@st.cache_resource
def load_model_and_scaler():
    model = keras.models.load_model(HERE / 'popularity_model.keras')
    with open(HERE / 'scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model_and_scaler()


# UI
st.title("🎵 Spotify Popularity Predictor")
st.markdown(
    "A **Deep Neural Network** trained on 160,000 Spotify songs predicts whether "
    "a song will be popular (popularity ≥ 50) based on its audio features."
)
st.markdown("---")

st.markdown("### Adjust the audio features below")

# Two columns of sliders
col1, col2 = st.columns(2)

with col1:
    danceability = st.slider("Danceability", 0.0, 1.0, 0.5, 0.01,
                              help="How suitable the track is for dancing")
    energy = st.slider("Energy", 0.0, 1.0, 0.5, 0.01,
                        help="Perceived intensity and activity")
    acousticness = st.slider("Acousticness", 0.0, 1.0, 0.5, 0.01,
                              help="Confidence the track is acoustic")
    instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.0, 0.01,
                                  help="Confidence the track has no vocals")
    liveness = st.slider("Liveness", 0.0, 1.0, 0.15, 0.01,
                          help="Presence of an audience in the recording")

with col2:
    loudness = st.slider("Loudness (dB)", -60.0, 0.0, -8.0, 0.5,
                          help="Overall loudness, typical range -20 to 0 dB")
    speechiness = st.slider("Speechiness", 0.0, 1.0, 0.05, 0.01,
                             help="Presence of spoken words")
    tempo = st.slider("Tempo (BPM)", 50.0, 220.0, 120.0, 1.0,
                       help="Beats per minute")
    valence = st.slider("Valence", 0.0, 1.0, 0.5, 0.01,
                         help="Musical positivity / happiness")


# Make prediction
# Feature order MUST match the training order
features = np.array([[
    acousticness, danceability, energy, instrumentalness,
    liveness, loudness, speechiness, tempo, valence
]])

# Scale and predict
features_scaled = scaler.transform(features)
prob = float(model.predict(features_scaled, verbose=0)[0][0])


# Display result
st.markdown("---")
st.markdown("### Prediction")

result_col1, result_col2 = st.columns([1, 2])

with result_col1:
    if prob > 0.5:
        st.success("### 🎉 Likely Popular")
    else:
        st.warning("### 🎵 Likely Not Popular")
    st.metric("Probability of being popular", f"{prob*100:.1f}%")

with result_col2:
    # Confidence bar
    st.markdown("**Confidence:**")
    st.progress(prob)
    if prob > 0.7:
        st.caption("Strong confidence this song would be popular.")
    elif prob > 0.5:
        st.caption("Mild confidence this song would be popular.")
    elif prob > 0.3:
        st.caption("Mild confidence this song would NOT be popular.")
    else:
        st.caption("Strong confidence this song would NOT be popular.")


# Sidebar - info
with st.sidebar:
    st.header("About")
    st.markdown(
        """
        This web app uses a **Deep Neural Network** trained on
        ~140,000 Spotify songs to predict popularity from audio features.

        **Model architecture:**
        - Input: 9 audio features
        - Hidden layers: 64 → 32 → 16 neurons
        - ReLU activation, Dropout 0.3
        - Output: 1 neuron, sigmoid
        - Loss: binary cross-entropy
        - Optimizer: Adam

        **Dataset:** [160k Spotify songs (Kaggle)](https://www.kaggle.com/datasets/fcpercival/160k-spotify-songs-sorted)

        """
    )

    st.markdown("---")
    st.caption("MAL Exam Project · 2026")
