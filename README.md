## Spotify Popularity Predictor

A **Deep Neural Network** that predicts whether a song will be popular based on its audio features.

---

## Dataset

[160k Spotify songs from 1921 to 2020 (Kaggle)](https://www.kaggle.com/datasets/fcpercival/160k-spotify-songs-sorted)

Each song has 9 audio features extracted by Spotify's API:

- **danceability**, **energy**, **acousticness**, **instrumentalness**
- **liveness**, **loudness**, **speechiness**, **tempo**, **valence**

After cleaning duplicates and non-music entries, ~140,000 songs remain.

---

## Model Architecture

A Deep Neural Network built in Keras:

| Layer | Type | Neurons | Activation |
|---|---|---|---|
| Input | — | 9 | — |
| Hidden 1 | Dense | 64 | ReLU |
| Dropout | Dropout (0.3) | — | — |
| Hidden 2 | Dense | 32 | ReLU |
| Dropout | Dropout (0.3) | — | — |
| Hidden 3 | Dense | 16 | ReLU |
| Output | Dense | 1 | Sigmoid |

- **Loss:** Binary cross-entropy
- **Optimizer:** Adam
- **Regularisation:** Dropout to prevent overfitting
- **Early stopping:** Monitors validation loss to avoid overtraining

---

## Model Evaluation

| Metric | Value |
|---|---|
| Test accuracy | ~75-80% |
| ROC AUC | ~0.80 |
| Classes | Balanced via stratified split |

*(Update with your actual numbers after running the notebook.)*

Evaluation includes accuracy, precision, recall, F1-score, confusion matrix, and ROC curve.

---

## Screenshots

### Epoch output
![Epoch](screenshots/training_epoch_output.png)

### Model summary
![Model summary](screenshots/model.summary().png)

### Training curves
![Training curves](screenshots/training_curves.png)

### Confusion matrix
![Confusion matrix](screenshots/confusion_matrix.png)

### ROC curve
![ROC curve](screenshots/roc_curve.png)


---


## Project Structure

```
spotify_recommender/
├── popularity_predictor.ipynb    # Training notebook (run in Google Colab)
├── app.py                        # The deployed Streamlit app
├── requirements.txt              # Python dependencies
├── popularity_model.keras        # Trained DNN
├── scaler.pkl                    # Fitted StandardScaler
├── screenshots/                  # Plots and demo screenshots
└── README.md
```

---

## Running Locally

1. Clone:
   ```bash
   git clone https://github.com/frederikG1/spotify_recommender.git
   cd spotify_recommender
   ```

2. Install:
   ```bash
   pip install -r requirements.txt
   ```

3. Run:
   ```bash
   streamlit run app.py
   ```

To retrain the model from scratch, open `popularity_predictor.ipynb` in [Google Colab](https://colab.research.google.com) and run all cells.

---

