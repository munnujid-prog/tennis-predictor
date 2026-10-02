"""
Module 1 — Data Preparation
Cleans the raw ATP dataset and produces chronological train/val/hold-out splits.
"""

import pandas as pd
import os

# ── Paths ──────────────────────────────────────────────────────────────────
RAW = os.path.join("data", "atp_tennis.csv")
OUT = "processed"
os.makedirs(OUT, exist_ok=True)

# ── Load ───────────────────────────────────────────────────────────────────
df = pd.read_csv(RAW)
print(f"Loaded  : {len(df):,} rows")

# ── Replace sentinel values (-1 / -1.0) with NaN ──────────────────────────
for col in ["Pts_1", "Pts_2", "Odd_1", "Odd_2"]:
    df[col] = df[col].replace(-1, pd.NA).replace(-1.0, pd.NA)

# ── Filter retirements / walkovers ─────────────────────────────────────────
mask_ret = df["Score"].str.contains(r"ret|w/o", case=False, na=False, regex=True)
df = df[~mask_ret].copy()
print(f"After retirement filter : {len(df):,} rows")

# ── Drop leaking / unused columns ─────────────────────────────────────────
DROP_COLS = ["Score", "Odd_1", "Odd_2", "Pts_1", "Pts_2", "Tournament"]
df.drop(columns=DROP_COLS, inplace=True)

# ── Derive binary target ───────────────────────────────────────────────────
df["Player_1_wins"] = (df["Winner"] == df["Player_1"]).astype(int)

# ── Drop rows with missing rankings ───────────────────────────────────────
df.dropna(subset=["Rank_1", "Rank_2"], inplace=True)
print(f"After rank null drop    : {len(df):,} rows")

# ── Chronological sort ────────────────────────────────────────────────────
df["Date"] = pd.to_datetime(df["Date"])
df.sort_values("Date", inplace=True)
df.reset_index(drop=True, inplace=True)

# ── Time-based split (70 / 15 / 15) ──────────────────────────────────────
n = len(df)
i_train = int(n * 0.70)
i_val   = int(n * 0.85)

train   = df.iloc[:i_train]
val     = df.iloc[i_train:i_val]
holdout = df.iloc[i_val:]

# ── Save ───────────────────────────────────────────────────────────────────
train.to_csv(os.path.join(OUT, "train.csv"),   index=False)
val.to_csv(  os.path.join(OUT, "val.csv"),     index=False)
holdout.to_csv(os.path.join(OUT, "holdout.csv"), index=False)

print(f"\nSplit summary:")
print(f"  Train   : {len(train):,} rows  ({train['Date'].min().date()} to {train['Date'].max().date()})")
print(f"  Val     : {len(val):,} rows  ({val['Date'].min().date()} to {val['Date'].max().date()})")
print(f"  Hold-out: {len(holdout):,} rows  ({holdout['Date'].min().date()} to {holdout['Date'].max().date()})")
print("\nDone — processed/train.csv, val.csv, holdout.csv written.")
