"""
Module 5 - Live Prediction Challenge
Simulates a live prediction scenario on the hold-out set.
No hold-out data was seen during training or tuning.
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder

# -- Paths ------------------------------------------------------------------
PROC = "processed"
MDIR = "models"

ROUND_MAP = {
    "1st Round":    1,
    "2nd Round":    2,
    "3rd Round":    3,
    "Round of 16":  4,
    "Quarterfinals":5,
    "Semifinals":   6,
    "The Final":    7,
}
CAT_COLS = ["Surface", "Court", "Series"]

# -- Load artefacts ---------------------------------------------------------
model        = joblib.load(os.path.join(MDIR, "rf_model.pkl"))
encoder: OneHotEncoder = joblib.load(os.path.join(MDIR, "encoder.pkl"))
feature_cols = joblib.load(os.path.join(MDIR, "feature_columns.pkl"))

holdout_df   = pd.read_csv(os.path.join(PROC, "holdout.csv"), parse_dates=["Date"])
X_hold       = np.load(os.path.join(PROC, "X_hold.npy"))
y_hold       = np.load(os.path.join(PROC, "y_hold.npy"))

# -- Overall hold-out accuracy ---------------------------------------------
y_pred_all = model.predict(X_hold)
overall_acc = accuracy_score(y_hold, y_pred_all)

# -- 10-match sample --------------------------------------------------------
sample = holdout_df.head(10).copy()
sample["round_num"] = sample["Round"].map(ROUND_MAP)
sample.dropna(subset=["round_num"], inplace=True)
sample["rank_diff"]  = sample["Rank_1"] - sample["Rank_2"]
sample["rank_ratio"] = sample["Rank_1"] / (sample["Rank_2"] + 1e-9)
sample["best_of"]    = sample["Best of"].astype(int)

cat_enc = encoder.transform(sample[CAT_COLS])
cat_df  = pd.DataFrame(cat_enc, columns=encoder.get_feature_names_out(CAT_COLS))

num_df = sample[["rank_diff", "rank_ratio", "round_num", "best_of"]].reset_index(drop=True)
X_sample = pd.concat([num_df, cat_df.reset_index(drop=True)], axis=1)[feature_cols].values

probas = model.predict_proba(X_sample)

# -- Print prediction table ------------------------------------------------
print("-- Live Prediction Challenge: 10 Hold-out Matches --")
print(f"{'#':<3} {'Player 1':<22} {'Player 2':<22} {'Surface':<8} {'Round':<16} "
      f"{'P(P1 wins)':<12} {'P(P2 wins)':<12} {'Actual Winner':<22} Result")
print("-" * 130)

correct = 0
for i, (_, row) in enumerate(sample.iterrows()):
    p1_prob = probas[i][1]
    p2_prob = probas[i][0]
    predicted_winner = row["Player_1"] if p1_prob >= 0.5 else row["Player_2"]
    actual_winner    = row["Winner"]
    flag = "OK" if predicted_winner == actual_winner else "MISS"
    if flag == "OK":
        correct += 1
    print(
        f"{i+1:<3} {row['Player_1']:<22} {row['Player_2']:<22} "
        f"{row['Surface']:<8} {row['Round']:<16} "
        f"{p1_prob:<12.3f} {p2_prob:<12.3f} "
        f"{actual_winner:<22} {flag}"
    )

sample_acc = correct / len(sample)
print("-" * 130)
print(f"\nSample accuracy (10 matches) : {sample_acc:.1%}  ({correct}/10 correct)")
print(f"Full hold-out accuracy       : {overall_acc:.4f}  ({len(holdout_df):,} matches)")
print("\nDisclaimer: This is a retrospective simulation.")
print("No hold-out data was seen during training or hyperparameter tuning.")
