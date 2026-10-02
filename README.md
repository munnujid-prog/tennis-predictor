# ATP Tennis Match Win Probability Predictor

A machine learning web application that predicts the outcome (win probability) of a match between two ATP players, built with scikit-learn and Streamlit.

## Project Description

This application uses historical ATP match data to train a Random Forest classifier. Given two players and match context (surface, round, series, court, best-of format), it predicts the win probability for each player.

**Key design decisions:**
- Player names are used only in the UI — never as model features.
- Betting odds are excluded to avoid data leakage.
- The train/val/hold-out split is strictly chronological to prevent future data leaking into training.

## Project Structure

```
tennis-predictor/
├── data/
│   └── atp_tennis.csv          # Raw dataset (read-only)
├── processed/                  # Cleaned splits produced by the pipeline
│   ├── train.csv
│   ├── val.csv
│   └── holdout.csv
├── models/                     # Trained model and encoder artefacts
│   ├── rf_model.pkl
│   ├── encoder.pkl
│   ├── feature_columns.pkl
│   ├── feature_importance_gini.png
│   └── feature_importance_permutation.png
├── src/
│   ├── 01_prepare.py           # Module 1 — Data Preparation
│   ├── 02_features.py          # Module 2 — Feature Engineering
│   ├── 03_train.py             # Module 3 — Model Building
│   ├── 04_interpret.py         # Module 4 — Model Interpretation
│   └── 05_predict.py           # Module 5 — Live Prediction Challenge
├── app.py                      # Module 6 — Streamlit Web UI
├── requirements.txt
└── README.md
```

## Setup

### 1. Create and activate a virtual environment

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

## Running the Pipeline

Run each script in order. Each step depends on the outputs of the previous step.

```bash
# Step 1 — Clean data and create train/val/hold-out splits
python src/01_prepare.py

# Step 2 — Engineer features and save encoder artefacts
python src/02_features.py

# Step 3 — Train and evaluate the Random Forest model
python src/03_train.py

# Step 4 — Analyse and plot feature importances
python src/04_interpret.py

# Step 5 — Run the live prediction challenge on hold-out data
python src/05_predict.py
```

## Launching the Web App

After running the full pipeline:

```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501**

## Feature Set

The model uses the following pre-match features (no player names, no betting odds):

| Feature | Description |
|---|---|
| `rank_diff` | Rank_1 minus Rank_2 (positive = Player 1 is lower-ranked) |
| `rank_ratio` | Rank_1 divided by Rank_2 |
| `round_num` | Ordinal encoding of the round (1 = 1st Round, 7 = The Final) |
| `Best of` | Match format (3 or 5 sets) |
| `Surface_*` | One-hot encoded surface (Hard, Clay, Grass, Carpet) |
| `Court_*` | One-hot encoded court type (Indoor, Outdoor) |
| `Series_*` | One-hot encoded tournament tier |

## Notes

- Expected validation accuracy: 62–70%. Accuracy above 80% likely indicates data leakage.
- The "live prediction challenge" (`05_predict.py`) is a retrospective simulation using the chronological hold-out tail — no real-time data feed is used.
- Player ranks shown in the UI are each player's most recent rank from the training data.
