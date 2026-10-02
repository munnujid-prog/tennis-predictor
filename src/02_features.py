"""
Module 2 — Feature Engineering
Transforms cleaned splits into numeric feature matrices ready for model training.
Saves fitted encoder artefacts for reuse by the Streamlit app.
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import OneHotEncoder

# ── Paths ──────────────────────────────────────────────────────────────────
PROC  = "processed"
MDIR  = "models"
os.makedirs(MDIR, exist_ok=True)

# ── Round ordinal mapping ──────────────────────────────────────────────────
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

# ── Helper ─────────────────────────────────────────────────────────────────
def engineer(df: pd.DataFrame, encoder: OneHotEncoder = None, fit: bool = False):
    df = df.copy()

    # Rank features
    df["rank_diff"]  = df["Rank_1"] - df["Rank_2"]
    df["rank_ratio"] = df["Rank_1"] / (df["Rank_2"] + 1e-9)

    # Ordinal round
    df["round_num"] = df["Round"].map(ROUND_MAP)
    df.dropna(subset=["round_num"], inplace=True)
    df["round_num"] = df["round_num"].astype(int)

    # Best of — already numeric, just rename for clarity
    df["best_of"] = df["Best of"].astype(int)

    # One-hot encode categorical columns
    if fit:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        encoder.fit(df[CAT_COLS])

    cat_array = encoder.transform(df[CAT_COLS])
    cat_cols  = encoder.get_feature_names_out(CAT_COLS)
    cat_df    = pd.DataFrame(cat_array, columns=cat_cols, index=df.index)

    # Build feature matrix (no names, no dates, no raw ranks)
    feature_df = pd.concat(
        [df[["rank_diff", "rank_ratio", "round_num", "best_of"]].reset_index(drop=True),
         cat_df.reset_index(drop=True)],
        axis=1,
    )

    y = df["Player_1_wins"].values

    return feature_df, y, encoder


# ── Load splits ────────────────────────────────────────────────────────────
train_df   = pd.read_csv(os.path.join(PROC, "train.csv"),   parse_dates=["Date"])
val_df     = pd.read_csv(os.path.join(PROC, "val.csv"),     parse_dates=["Date"])
holdout_df = pd.read_csv(os.path.join(PROC, "holdout.csv"), parse_dates=["Date"])

# ── Engineer features — fit ONLY on training data ──────────────────────────
X_train, y_train, encoder = engineer(train_df, fit=True)
X_val,   y_val,   _       = engineer(val_df,   encoder=encoder)
X_hold,  y_hold,  _       = engineer(holdout_df, encoder=encoder)

# ── Persist artefacts ──────────────────────────────────────────────────────
joblib.dump(encoder,                  os.path.join(MDIR, "encoder.pkl"))
joblib.dump(list(X_train.columns),    os.path.join(MDIR, "feature_columns.pkl"))

np.save(os.path.join(PROC, "X_train.npy"), X_train.values)
np.save(os.path.join(PROC, "y_train.npy"), y_train)
np.save(os.path.join(PROC, "X_val.npy"),   X_val.values)
np.save(os.path.join(PROC, "y_val.npy"),   y_val)
np.save(os.path.join(PROC, "X_hold.npy"),  X_hold.values)
np.save(os.path.join(PROC, "y_hold.npy"),  y_hold)

print("Feature matrix shapes:")
print(f"  X_train : {X_train.shape}")
print(f"  X_val   : {X_val.shape}")
print(f"  X_hold  : {X_hold.shape}")
print(f"\nFeatures ({len(X_train.columns)}): {list(X_train.columns)}")
print("\nDone — encoder.pkl, feature_columns.pkl, and .npy arrays saved.")
