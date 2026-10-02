# Tennis Match Outcome Predictor — Build Plan

## Top-Level Overview

Build a machine learning web application that predicts the win probability of an ATP tennis match between two players. The application is structured as six sequential modules: data preparation, feature engineering, model building, model interpretation, a live prediction challenge, and a Streamlit web UI dashboard.

The dataset is `atp_tennis.csv` — historical ATP match records from 2000 onwards. The model is a Random Forest classifier trained exclusively on pre-match, non-leaking features (rankings, surface, round, series, court, match format). Player names are never model features; they appear only in the UI to let users select opponents.

All train/test splits are time-based (chronological) to prevent future data leaking into training.

---

## Decisions & Defaults Applied

| Decision | Choice | Reason |
|---|---|---|
| `Tournament` column | Drop entirely | Already represented by `Series` + `Surface`; ~500+ values would bloat encoding |
| `Pts_1` / `Pts_2` | Drop entirely | `-1` sentinel is pervasive in early years; `Rank_*` is cleaner and sufficient |
| Betting odds (`Odd_1`, `Odd_2`) | Drop entirely | Encode all pre-match wisdom; per brief, excluded to avoid leakage/market-bias |
| Retirements / walkovers | Filter out | Model should not learn from non-natural match endings |
| Train/val/hold-out split | Chronological (time-based) | Prevents future data leaking into training |

---

## Dataset Summary

**Source file:** `atp_tennis.csv`
**Columns (17 total):**

| Column | Type | Represents | Action |
|---|---|---|---|
| `Tournament` | string (high-cardinality) | Tournament name | **Drop** |
| `Date` | date string | Match date | Use for sorting/splitting only, then drop from features |
| `Series` | categorical | Tournament tier (International, Grand Slam, etc.) | **Encode** |
| `Court` | categorical | Indoor / Outdoor | **Encode** |
| `Surface` | categorical | Hard / Clay / Grass / Carpet | **Encode** |
| `Round` | ordinal categorical | Draw stage (1st Round → Final) | **Ordinal encode** |
| `Best of` | int | Match format (3 or 5 sets) | **Keep as int** |
| `Player_1` | string | First player name | UI only — drop from features |
| `Player_2` | string | Second player name | UI only — drop from features |
| `Winner` | string | Name of match winner | **Target derivation only** — drop from features |
| `Rank_1` | int | ATP ranking of Player 1 at match time | **Keep** |
| `Rank_2` | int | ATP ranking of Player 2 at match time | **Keep** |
| `Pts_1` | int | Ranking points of Player 1 (`-1` = missing) | **Drop** |
| `Pts_2` | int | Ranking points of Player 2 (`-1` = missing) | **Drop** |
| `Odd_1` | float | Betting odds for Player 1 (`-1.0` = missing) | **Drop** (leakage risk) |
| `Odd_2` | float | Betting odds for Player 2 (`-1.0` = missing) | **Drop** (leakage risk) |
| `Score` | string | Final match score | **Drop** (post-match leakage) |

**Target variable:** `Player_1_wins = 1` if `Winner == Player_1`, else `0`.

**Leakage flags:**
- `Score` — only known after the match; must be dropped before any feature engineering.
- `Winner` — is the label itself; never a feature.
- `Odd_1` / `Odd_2` — betting odds absorb all pre-match signals including form, injuries, and head-to-head; excluded per project brief.

**Data quality issues:**
- `Pts_1`, `Pts_2`, `Odd_1`, `Odd_2` use `-1` / `-1.0` as missing-value sentinels — must be converted to `NaN` before any processing.
- `Player_1` / `Player_2` ordering is arbitrary (not rank-ordered), so `rank_diff` must be computed directionally as `Rank_1 - Rank_2`.
- Some `Score` values contain retirement/walkover suffixes (e.g. `"ret."`, `"w/o"`) — these rows must be filtered out before training.

---

## Recommended Tech Stack

| Component | Choice | Notes |
|---|---|---|
| Language | Python 3.11 | Widely supported, stable |
| Data handling | pandas | Industry-standard DataFrame library |
| ML | scikit-learn | Random Forest, preprocessing, cross-validation, permutation importance |
| Model serialisation | joblib | Ships with scikit-learn; preferred over pickle for numpy arrays |
| Web UI | **Streamlit** | Pure-Python, no HTML/CSS required, ideal for beginners and workshops |
| Charts | matplotlib / seaborn | Sufficient for feature importance plots and probability bars |
| Environment | `venv` + `requirements.txt` | Minimal, no extra tooling needed |

**Streamlit vs. Dash:** Streamlit requires zero front-end knowledge — widgets are Python function calls and the app re-runs top-to-bottom on every interaction. Dash requires explicit callback wiring and layout components, which adds complexity inappropriate for a beginner workshop.

---

## Project Structure

```
tennis-predictor/
├── data/
│   └── atp_tennis.csv              # raw dataset (read-only)
├── processed/
│   ├── train.csv                   # Module 1 output — training split
│   ├── val.csv                     # Module 1 output — validation split
│   └── holdout.csv                 # Module 1 output — hold-out split
├── models/
│   └── rf_model.pkl                # Module 3 output — serialised model
├── src/
│   ├── 01_prepare.py               # Module 1 — data cleaning & splitting
│   ├── 02_features.py              # Module 2 — feature engineering
│   ├── 03_train.py                 # Module 3 — model training & evaluation
│   ├── 04_interpret.py             # Module 4 — feature importance
│   └── 05_predict.py               # Module 5 — live prediction challenge
├── app.py                          # Module 6 — Streamlit web UI
└── requirements.txt
```

Each `src/` script is standalone and runnable in sequence (`python src/01_prepare.py`, etc.). `app.py` loads the artefacts produced by the pipeline and serves the UI.

---

## Module-by-Module Build Plan

---

### Module 1 — Data Preparation

**Status:** `[ ] pending`

**Objective:** Produce clean, split datasets that are free of leakage and ready for feature engineering.

**Key Steps:**
1. Load `data/atp_tennis.csv` with pandas.
2. Convert `-1` and `-1.0` sentinel values in `Pts_1`, `Pts_2`, `Odd_1`, `Odd_2` to `NaN`.
3. Filter out retirement/walkover rows: drop any row where `Score` contains `"ret."`, `"RET"`, `"w/o"`, or `"W/O"` (case-insensitive).
4. Drop columns that must never reach the model: `Score`, `Odd_1`, `Odd_2`, `Pts_1`, `Pts_2`, `Tournament`.
5. Derive the binary target: `Player_1_wins = (Winner == Player_1).astype(int)`.
6. Drop rows where `Rank_1` or `Rank_2` is null (ranking is the core feature).
7. Parse `Date` to datetime and sort the full dataset chronologically.
8. Split by row position (no shuffling): 70 % train / 15 % validation / 15 % hold-out.
9. Save the three splits to `processed/train.csv`, `processed/val.csv`, `processed/holdout.csv`. Retain `Player_1`, `Player_2`, `Winner`, `Date` in saved files for traceability, but they will be stripped by Module 2 before model input.

**Expected Output:** Three clean CSV files with consistent column sets, no sentinel values, no leaking columns, sorted by date.

---

### Module 2 — Feature Engineering

**Status:** `[ ] pending`

**Objective:** Transform the clean splits into numeric feature matrices suitable for a Random Forest classifier, with no player-name or post-match information.

**Key Steps:**
1. Load each split CSV produced by Module 1.
2. Compute derived numeric features:
   - `rank_diff = Rank_1 - Rank_2` (positive = Player 1 is lower-ranked i.e. worse).
   - `rank_ratio = Rank_1 / Rank_2` (captures proportional gap).
3. Ordinal-encode `Round` using a fixed ordering: `1st Round=1, 2nd Round=2, 3rd Round=3, Round of 16=4, Quarterfinals=5, Semifinals=6, The Final=7`. Fit the mapping on training data only; apply to val and hold-out.
4. One-hot encode `Surface`, `Court`, `Series` using `pd.get_dummies` or `sklearn.OneHotEncoder`. Fit on training data only; align val/hold-out columns to match.
5. Keep `Best of` as an integer feature.
6. Drop `Player_1`, `Player_2`, `Winner`, `Date`, `Rank_1`, `Rank_2` from the feature matrix (raw ranks replaced by derived features; names and date are not model inputs).
7. Separate target column `Player_1_wins` from the feature matrix.
8. Return/save: `X_train`, `X_val`, `X_holdout`, `y_train`, `y_val`, `y_holdout`.

**Expected Output:** Numeric feature matrices (no string columns, no player names) and label arrays for each split, with identical column ordering across all three.

---

### Module 3 — Model Building

**Status:** `[ ] pending`

**Objective:** Train a Random Forest classifier, tune it on training data only, evaluate it on the validation set, and serialise the final model.

**Key Steps:**
1. Load feature matrices from Module 2.
2. Establish a baseline: train a `RandomForestClassifier` with default parameters and log validation accuracy and ROC-AUC.
3. Tune `n_estimators` (try 100, 200, 300) and `max_depth` (try None, 10, 20) using 5-fold cross-validation on the **training set only**. Use `GridSearchCV` or a manual loop.
4. Train the best-parameter model on the full training set.
5. Evaluate on the **validation set**:
   - Accuracy
   - ROC-AUC
   - Confusion matrix
   - Classification report (precision, recall, F1)
6. Serialise the trained model to `models/rf_model.pkl` using `joblib.dump`.
7. Print and save a brief evaluation summary to stdout.

**Expected Output:** `models/rf_model.pkl` — a trained, tuned Random Forest. A printed validation-set evaluation report.

---

### Module 4 — Model Interpretation

**Status:** `[ ] pending`

**Objective:** Explain which features drive the model's predictions using two complementary methods, and surface any surprising or suspicious importances.

**Key Steps:**
1. Load `models/rf_model.pkl` and the validation feature matrix from Module 2.
2. Extract **built-in (Gini impurity) feature importances** from `model.feature_importances_`. Plot as a horizontal bar chart sorted descending. Save to `models/feature_importance_gini.png`.
3. Compute **permutation importances** on the validation set using `sklearn.inspection.permutation_importance` (n_repeats=10). Plot as a horizontal bar chart with error bars. Save to `models/feature_importance_permutation.png`.
4. Print a comparison table: feature name, Gini importance rank, permutation importance rank.
5. Write a short commentary block in the script (as comments or printed output) noting:
   - Whether `rank_diff` dominates (expected).
   - Whether surface/court/series add meaningful signal.
   - Any features with near-zero importance that could be dropped in a future iteration.

**Expected Output:** Two importance chart images saved to `models/`. A printed comparison table and commentary.

---

### Module 5 — Live Prediction Challenge

**Status:** `[ ] pending`

**Objective:** Simulate a "live" prediction scenario using the hold-out set (the most recent matches the model has never seen), and compare predicted outcomes to actual results.

**Key Steps:**
1. Load `processed/holdout.csv` and apply the same feature engineering pipeline used in Module 2 (use the fitted encoders from training — do not refit on hold-out data).
2. Load `models/rf_model.pkl`.
3. Sample 10 representative hold-out matches spanning different surfaces and rounds (or use the first 10 chronologically).
4. For each match:
   - Display: Player 1 name, Player 2 name, surface, round, tournament series.
   - Feed engineered features (no names) into `model.predict_proba`.
   - Display: predicted win probability for each player.
   - Display: actual winner.
   - Flag: correct ✓ or incorrect ✗.
5. Print overall accuracy on the full hold-out set.
6. Note in comments that this is a retrospective simulation — there is no live data feed; the hold-out tail of the historical dataset serves as a proxy for unseen future matches.

**Expected Output:** A printed match-by-match prediction table with probabilities, actual outcomes, and a summary accuracy on the full hold-out set.

---

### Module 6 — Streamlit Web UI Dashboard

**Status:** `[ ] pending`

**Objective:** Deliver an interactive web app where a user selects two ATP players by name, and the app displays win probabilities using the trained model. Player names are UI labels only — they are never passed to the model.

**Key Steps:**
1. On startup, load `models/rf_model.pkl` and build a **player stats lookup table** from the training split:
   - For each unique player, record their most recent (`Rank`) value from the training data.
   - Also record their most common `Surface` win rate (optional enrichment for display only).
2. Build the UI layout:
   - Title: "ATP Match Predictor".
   - Two `st.selectbox` widgets — "Select Player 1" and "Select Player 2" — populated from the player list in the lookup table.
   - A `st.selectbox` for `Surface` (Hard, Clay, Grass, Carpet).
   - A `st.selectbox` for `Round` (1st Round through The Final).
   - A `st.selectbox` for `Series` (International, Grand Slam, etc.).
   - A `st.selectbox` for `Court` (Indoor, Outdoor).
   - A `st.selectbox` for `Best of` (3 or 5).
3. On "Predict" button click:
   - Resolve the selected players' latest ranks from the lookup table.
   - Construct the feature vector: `rank_diff`, `rank_ratio`, one-hot surface/court/series, ordinal round, `best_of`.
   - Call `model.predict_proba([feature_vector])`.
   - Display win probabilities as a horizontal bar chart (`st.bar_chart` or `matplotlib` figure via `st.pyplot`).
   - Display a "Predicted Winner" callout using `st.success` or `st.info`.
4. Add a sidebar note: "Predictions are based on historical ATP data and ranking features only. Betting odds and player names are not used by the model."

**Expected Output:** `app.py` — a runnable Streamlit app (`streamlit run app.py`) that accepts two player names, match context, and returns win probabilities.

---

## Assumptions

1. Player ordering in each CSV row is arbitrary — not sorted by rank or seeding. The `rank_diff` feature accounts for this directionally.
2. `Rank_1` and `Rank_2` represent the ATP ranking at the time of the match (pre-match information), not post-match.
3. The dataset is historical only. There is no live data feed. The "live prediction challenge" in Module 5 is a retrospective simulation using the chronological hold-out tail.
4. The Streamlit UI uses each player's most recent rank from the **training set** as a proxy for their current rank. This is a simplification appropriate for a workshop.
5. `Best of` is treated as a known pre-match feature (it is determined by the tournament format, not the match outcome).
6. `Tournament` is dropped because `Series` + `Surface` + `Court` together adequately represent the tournament context without introducing high-cardinality encoding.

---

## Open Questions (resolved by defaults)

| Question | Decision |
|---|---|
| Use `Tournament` column? | Drop — represented by `Series` + `Surface` |
| Use `Pts_1` / `Pts_2`? | Drop — sparse, `-1` sentinel pervasive in early data |
| Use betting odds? | Drop — excluded per project brief |
| Filter retirements/walkovers? | Yes — model should not learn from non-natural endings |
| Random vs. time-based split? | Time-based (chronological) |

---

## Risks & Trade-offs

### Risk 1 — Data Leakage via Random Splitting
**Problem:** Using `train_test_split(shuffle=True)` means matches from 2023 could appear in the training set while 2015 matches appear in the test set. The model learns from "the future" and evaluation accuracy is inflated.
**Mitigation:** Sort by `Date` ascending before splitting. Use row-position cutoffs (70/15/15) with no shuffling. This is enforced in Module 1 and must never be overridden.

### Risk 2 — Data Leakage via Betting Odds
**Problem:** `Odd_1` and `Odd_2` effectively encode the market's pre-match probability, which already incorporates rankings, form, surface history, and head-to-head. Including them yields unrealistically high accuracy (~80–85 %) and teaches the model nothing about tennis structure.
**Mitigation:** Both columns are dropped in Module 1, before any feature engineering or model training.

### Risk 3 — High-Cardinality `Tournament` Encoding
**Problem:** There are hundreds of distinct tournament names. One-hot encoding them produces a very wide sparse feature matrix, risks overfitting on rare tournaments, and makes the model harder to interpret.
**Mitigation:** `Tournament` is dropped entirely. `Series` (6–8 tiers) + `Surface` (4 values) + `Court` (2 values) jointly capture the relevant tournament context at low cardinality cost.
