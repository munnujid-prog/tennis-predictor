# ATP Tennis Match Predictor — Implementation Plan

## Overview

This plan breaks the build into 8 sequential phases. Each phase produces a concrete, reviewable artefact before the next phase begins. All decisions from `tennis-predictor-plan.md` are locked in:

- Time-based (chronological) 70/15/15 train/val/hold-out split
- `Tournament`, `Pts_1`, `Pts_2`, `Odd_1`, `Odd_2`, `Score` dropped
- Player names used in UI only — never model features
- Retirements/walkovers filtered out
- Random Forest classifier via scikit-learn
- Streamlit as the web UI framework

---

## Phase 1 — Project Setup

**Status:** `[ ] pending`

**Intent:** Establish the project skeleton so every subsequent phase has a consistent, reproducible environment to work within.

**Expected Outcomes:**
- Folder structure exists on disk.
- `requirements.txt` lists all dependencies with pinned or minimum versions.
- `data/atp_tennis.csv` is confirmed present and readable.
- A developer can run `pip install -r requirements.txt` and immediately proceed to Phase 2.

**Todo List:**
1. Create the following directories: `data/`, `processed/`, `models/`, `src/`.
2. Move (or confirm) `atp_tennis.csv` is located at `data/atp_tennis.csv`.
3. Create `requirements.txt` with the following dependencies:
   - `pandas>=2.0`
   - `scikit-learn>=1.4`
   - `joblib>=1.3`
   - `matplotlib>=3.8`
   - `seaborn>=0.13`
   - `streamlit>=1.35`
4. Create a `README.md` with: project description, setup instructions (`venv` creation, `pip install`), and the run order for all scripts.
5. Verify `data/atp_tennis.csv` loads without error by reading the first 5 rows in a quick sanity check.

**Relevant Files:**
- `data/atp_tennis.csv` — raw dataset (read-only throughout all phases)
- `requirements.txt` — dependency manifest
- `README.md` — project documentation

---

## Phase 2 — Module 1: Data Preparation

**Status:** `[ ] pending`

**Intent:** Produce clean, leakage-free, chronologically split datasets that are the single source of truth for all downstream modules.

**Expected Outcomes:**
- `processed/train.csv` — earliest ~70% of matches, cleaned.
- `processed/val.csv` — next ~15% of matches, cleaned.
- `processed/holdout.csv` — latest ~15% of matches, cleaned.
- No sentinel values (`-1`, `-1.0`) remain in any split.
- No retirement/walkover rows remain.
- Binary target column `Player_1_wins` is present in all three files.
- `Player_1`, `Player_2`, `Winner`, `Date` are retained in the saved files for traceability but will be stripped in Phase 3.

**Todo List:**
1. Create `src/01_prepare.py`.
2. Load `data/atp_tennis.csv` with pandas.
3. Replace `-1` and `-1.0` with `NaN` in columns: `Pts_1`, `Pts_2`, `Odd_1`, `Odd_2`.
4. Filter out retirement/walkover rows: drop rows where `Score` contains `ret`, `RET`, `w/o`, or `W/O` (case-insensitive regex).
5. Drop columns: `Score`, `Odd_1`, `Odd_2`, `Pts_1`, `Pts_2`, `Tournament`.
6. Derive target: `Player_1_wins = (df['Winner'] == df['Player_1']).astype(int)`.
7. Drop rows where `Rank_1` or `Rank_2` is null.
8. Parse `Date` to datetime; sort entire DataFrame ascending by `Date`.
9. Compute split indices: 70% train cutoff, 85% val cutoff (remaining = hold-out).
10. Save three splits to `processed/train.csv`, `processed/val.csv`, `processed/holdout.csv`.
11. Print row counts for each split as a sanity check.

**Relevant Files:**
- Input: `data/atp_tennis.csv`
- Output: `processed/train.csv`, `processed/val.csv`, `processed/holdout.csv`
- Reference: `tennis-predictor-plan.md` — Dataset Summary section

---

## Phase 3 — Module 2: Feature Engineering

**Status:** `[ ] pending`

**Intent:** Transform the cleaned splits into numeric feature matrices with no player-name or post-match information, ready for model training.

**Expected Outcomes:**
- Six artefacts: `X_train`, `X_val`, `X_holdout` (feature matrices) and `y_train`, `y_val`, `y_holdout` (label arrays).
- All matrices have identical column sets and ordering.
- No string columns, no player names, no dates remain in any feature matrix.
- Fitted encoder state is saved so the Streamlit app (Phase 7) can reuse it without refitting.

**Todo List:**
1. Create `src/02_features.py`.
2. Load all three splits from `processed/`.
3. Compute `rank_diff = Rank_1 - Rank_2` for each split.
4. Compute `rank_ratio = Rank_1 / Rank_2` for each split (handle divide-by-zero with a small epsilon or clip).
5. Define a fixed ordinal mapping for `Round`:
   `{'1st Round': 1, '2nd Round': 2, '3rd Round': 3, 'Round of 16': 4, 'Quarterfinals': 5, 'Semifinals': 6, 'The Final': 7}`
   Apply to all three splits using `.map()`. Rows with unmapped round values are dropped.
6. One-hot encode `Surface`, `Court`, `Series` using `sklearn.preprocessing.OneHotEncoder`:
   - Fit **only on training data**.
   - Transform val and hold-out using the fitted encoder.
   - Use `handle_unknown='ignore'` to handle any unseen categories in val/hold-out.
7. Keep `Best of` as an integer column (already numeric).
8. Drop from feature matrix: `Player_1`, `Player_2`, `Winner`, `Date`, `Rank_1`, `Rank_2`.
9. Separate `Player_1_wins` as the label array `y` for each split.
10. Persist the fitted `OneHotEncoder` and the round ordinal mapping to `models/encoder.pkl` using `joblib.dump`.
11. Persist the final feature column list to `models/feature_columns.pkl` so the app can reconstruct vectors correctly.
12. Save processed matrices as numpy arrays or CSVs to `processed/` for use in Phase 4.
13. Print feature matrix shape for each split as a sanity check.

**Relevant Files:**
- Input: `processed/train.csv`, `processed/val.csv`, `processed/holdout.csv`
- Output: `models/encoder.pkl`, `models/feature_columns.pkl`, processed feature arrays
- Reference: `tennis-predictor-plan.md` — Module 2 section

---

## Phase 4 — Module 3: Model Building

**Status:** `[ ] pending`

**Intent:** Train, tune, and evaluate a Random Forest classifier using only training data, then serialise the best model for use in all downstream phases.

**Expected Outcomes:**
- `models/rf_model.pkl` — trained, tuned Random Forest classifier.
- Validation-set evaluation printed to stdout: accuracy, ROC-AUC, confusion matrix, classification report.
- Hyperparameter search performed on training data only (no val/hold-out contamination).

**Todo List:**
1. Create `src/03_train.py`.
2. Load `X_train`, `y_train`, `X_val`, `y_val` from Phase 3 outputs.
3. Train a baseline `RandomForestClassifier(random_state=42)` and log validation accuracy and ROC-AUC.
4. Define a hyperparameter grid:
   - `n_estimators`: [100, 200, 300]
   - `max_depth`: [None, 10, 20]
5. Run `GridSearchCV` with 5-fold cross-validation on the **training set only**. Scoring metric: `roc_auc`.
6. Refit the best estimator on the full training set.
7. Evaluate on `X_val` / `y_val`:
   - `accuracy_score`
   - `roc_auc_score`
   - `confusion_matrix`
   - `classification_report`
8. Print all evaluation metrics clearly labelled.
9. Save the best model to `models/rf_model.pkl` using `joblib.dump`.
10. Print best hyperparameters found by grid search.

**Relevant Files:**
- Input: Phase 3 feature arrays, `models/feature_columns.pkl`
- Output: `models/rf_model.pkl`
- Reference: `tennis-predictor-plan.md` — Module 3 section

---

## Phase 5 — Module 4: Model Interpretation

**Status:** `[ ] pending`

**Intent:** Explain which features drive predictions using two complementary importance methods, enabling workshop participants to understand what the model has learned.

**Expected Outcomes:**
- `models/feature_importance_gini.png` — horizontal bar chart of built-in (Gini) importances.
- `models/feature_importance_permutation.png` — horizontal bar chart of permutation importances with error bars.
- A printed comparison table: feature name, Gini rank, permutation rank.
- Brief printed commentary on dominant features and any zero-importance features.

**Todo List:**
1. Create `src/04_interpret.py`.
2. Load `models/rf_model.pkl`, `X_val`, `y_val`, and `models/feature_columns.pkl`.
3. Extract `model.feature_importances_`; pair with feature names; sort descending.
4. Plot as a horizontal bar chart (seaborn or matplotlib). Save to `models/feature_importance_gini.png`.
5. Compute `sklearn.inspection.permutation_importance(model, X_val, y_val, n_repeats=10, random_state=42)`.
6. Plot permutation importances (mean ± std) as a horizontal bar chart. Save to `models/feature_importance_permutation.png`.
7. Build and print a comparison table: rank by Gini vs. rank by permutation importance.
8. Print commentary:
   - Does `rank_diff` dominate? (expected — flag if not)
   - Do surface/court/series add signal?
   - Any near-zero importance features that could be pruned?

**Relevant Files:**
- Input: `models/rf_model.pkl`, Phase 3 val arrays, `models/feature_columns.pkl`
- Output: `models/feature_importance_gini.png`, `models/feature_importance_permutation.png`
- Reference: `tennis-predictor-plan.md` — Module 4 section

---

## Phase 6 — Module 5: Live Prediction Challenge

**Status:** `[ ] pending`

**Intent:** Simulate a live prediction scenario by running the trained model against the hold-out set (matches the model has never seen) and comparing predictions to actual outcomes.

**Expected Outcomes:**
- A printed match-by-match table for 10 sample hold-out matches showing: player names, surface, round, predicted probabilities, actual winner, correct/incorrect flag.
- Overall hold-out set accuracy printed.
- A note confirming no hold-out data was used during training or tuning.

**Todo List:**
1. Create `src/05_predict.py`.
2. Load `models/rf_model.pkl`, `models/encoder.pkl`, `models/feature_columns.pkl`.
3. Load `processed/holdout.csv`.
4. Apply the same feature engineering as Phase 3 (using the **fitted** encoder from `models/encoder.pkl` — do NOT refit).
5. Select 10 representative matches (first 10 chronologically, or manually curated to span different surfaces and rounds).
6. For each of the 10 matches:
   - Print: Player 1 name, Player 2 name, Surface, Round, Tournament Series.
   - Call `model.predict_proba([feature_vector])` — extract `[P(Player_1_wins), P(Player_2_wins)]`.
   - Print predicted probabilities for both players.
   - Print actual winner.
   - Print correct or incorrect outcome.
7. Run `model.predict` on the entire hold-out `X_holdout`; compute and print overall hold-out accuracy.
8. Print a disclaimer comment: "This is a retrospective simulation. No hold-out data was seen during training."

**Relevant Files:**
- Input: `models/rf_model.pkl`, `models/encoder.pkl`, `models/feature_columns.pkl`, `processed/holdout.csv`
- Output: Printed prediction report (stdout)
- Reference: `tennis-predictor-plan.md` — Module 5 section

---

## Phase 7 — Module 6: Streamlit Web UI Dashboard

**Status:** `[ ] pending`

**Intent:** Deliver the interactive web application where users select two ATP players and match context to receive win probability predictions. Player names are UI labels only and never enter the model.

**Expected Outcomes:**
- `app.py` runs with `streamlit run app.py` without errors.
- User can select Player 1, Player 2, Surface, Round, Series, Court, and Best of.
- Clicking "Predict" displays a win probability bar chart and a predicted winner callout.
- Sidebar shows a model transparency note.
- App gracefully handles edge cases: same player selected twice, players not found in lookup.

**Todo List:**
1. Create `app.py`.
2. On startup (cached with `@st.cache_resource`):
   - Load `models/rf_model.pkl`.
   - Load `models/encoder.pkl` and `models/feature_columns.pkl`.
   - Build player lookup table from `processed/train.csv`: for each unique player name, record their most recent `Rank_1` or `Rank_2` value (whichever is later by date).
3. Build sidebar:
   - Title: "ATP Match Predictor".
   - Note: "Predictions use ranking and match-context features only. Player names and betting odds are not used by the model."
4. Build main panel:
   - `st.title("ATP Match Win Probability Predictor")`
   - Two `st.selectbox` widgets for Player 1 and Player 2 (sorted alphabetically from the lookup table).
   - `st.selectbox` for Surface: Hard, Clay, Grass, Carpet.
   - `st.selectbox` for Round: 1st Round through The Final.
   - `st.selectbox` for Series: populate from unique values in training data.
   - `st.selectbox` for Court: Indoor, Outdoor.
   - `st.selectbox` for Best of: 3, 5.
5. On `st.button("Predict")` click:
   - Validate Player 1 is not Player 2; show `st.error` if same player selected.
   - Resolve `Rank_1`, `Rank_2` from the lookup table.
   - Construct feature vector using the same logic as Phase 3 (rank_diff, rank_ratio, ordinal round, one-hot via loaded encoder).
   - Call `model.predict_proba([feature_vector])`.
   - Display a horizontal bar chart of win probabilities for both players using `st.pyplot` or `st.bar_chart`.
   - Display `st.success` showing the predicted winner (player with higher probability).
   - Display `st.info` showing each player's resolved rank used in the prediction.
6. Handle edge case: if a player is not in the lookup table, show `st.warning` and prompt the user to enter a manual rank override via `st.number_input`.

**Relevant Files:**
- Input: `models/rf_model.pkl`, `models/encoder.pkl`, `models/feature_columns.pkl`, `processed/train.csv`
- Output: Running Streamlit app at `http://localhost:8501`
- Reference: `tennis-predictor-plan.md` — Module 6 section

---

## Phase 8 — Testing and Validation

**Status:** `[ ] pending`

**Intent:** Verify the full pipeline runs end-to-end without errors, confirm no data leakage has been introduced, and validate that the Streamlit app is functional and displays correct outputs.

**Expected Outcomes:**
- All six scripts run sequentially without errors.
- Processed file row counts are consistent (train + val + hold-out approximates total cleaned rows).
- Val and hold-out accuracy metrics are plausible (Random Forest on ranking features typically achieves 62-70%; anything above 80% warrants a leakage investigation).
- Feature importance charts render and are saved correctly.
- Hold-out prediction table prints correctly.
- Streamlit app loads, accepts input, and returns predictions without errors.
- No player names appear in any feature matrix.
- Hold-out dates are all strictly later than the last training date.

**Todo List:**
1. Run `python src/01_prepare.py` — verify three CSV files appear in `processed/` and row counts print correctly.
2. Run `python src/02_features.py` — verify feature matrix shapes print and `models/encoder.pkl` plus `models/feature_columns.pkl` are created.
3. Run `python src/03_train.py` — verify `models/rf_model.pkl` is created; check val accuracy is in the 62-70% range. If above 80%, inspect for leakage.
4. Run `python src/04_interpret.py` — verify both PNG files appear in `models/`; confirm `rank_diff` or `rank_ratio` appears in the top 3 features.
5. Run `python src/05_predict.py` — verify the 10-match prediction table prints; confirm hold-out accuracy is printed.
6. Run `streamlit run app.py` — open browser at `http://localhost:8501`; select two players and click Predict; verify probability chart and winner callout appear.
7. Leakage check: confirm no column from `Score`, `Odd_1`, `Odd_2`, `Pts_1`, `Pts_2`, `Winner`, `Player_1`, `Player_2` appears in any feature matrix by inspecting `models/feature_columns.pkl`.
8. Temporal integrity check: load `processed/train.csv` and `processed/holdout.csv`; assert that the maximum date in train is earlier than the minimum date in hold-out.
9. Document any issues found and their resolutions in a `TESTING_NOTES.md` file.

**Relevant Files:**
- All files from Phases 1-7
- Output: `TESTING_NOTES.md`

---

## Artefact Dependency Map

```
data/atp_tennis.csv
        |
        v
[Phase 2] src/01_prepare.py
        |
        v
processed/{train,val,holdout}.csv
        |
        v
[Phase 3] src/02_features.py
        |
        +---> models/encoder.pkl
        +---> models/feature_columns.pkl
        +---> X_train, X_val, X_holdout, y_train, y_val, y_holdout
                    |
                    v
          [Phase 4] src/03_train.py
                    |
                    v
             models/rf_model.pkl
                    |
          +---------+-----------+
          v                     v
[Phase 5] 04_interpret.py  [Phase 6] 05_predict.py
          |
     models/*.png
                          [Phase 7] app.py
                                |
                                v
                     streamlit run app.py
```

---

## Run Order

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run pipeline in order
python src/01_prepare.py
python src/02_features.py
python src/03_train.py
python src/04_interpret.py
python src/05_predict.py

# 3. Launch web app
streamlit run app.py
```
