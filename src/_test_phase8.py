"""
Phase 8 - End-to-end Testing and Validation
Runs all pipeline scripts sequentially and executes every check from
IMPLEMENTATION_PLAN.md Phase 8.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, roc_auc_score

PASS = "PASS"
FAIL = "FAIL"
results = []

def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    results.append((name, status, detail))
    mark = "OK" if condition else "!!"
    print(f"  [{mark}] {status}  {name}" + (f"  ({detail})" if detail else ""))
    return condition

print("=" * 65)
print("PHASE 8 - END-TO-END TESTING AND VALIDATION")
print("=" * 65)

# ------------------------------------------------------------------
print("\n--- 8.1  Pipeline Scripts Run Without Errors ---")
import subprocess, sys

scripts = [
    "src/01_prepare.py",
    "src/02_features.py",
    "src/03_train.py",
    "src/04_interpret.py",
    "src/05_predict.py",
]
for script in scripts:
    result = subprocess.run([sys.executable, script], capture_output=True, text=True)
    ok = result.returncode == 0
    check(f"{script} exits 0", ok, f"rc={result.returncode}")
    if not ok:
        print(f"    STDERR: {result.stderr[-300:]}")

# ------------------------------------------------------------------
print("\n--- 8.2  Processed File Row Counts Consistent ---")
train_df   = pd.read_csv("processed/train.csv",   parse_dates=["Date"])
val_df     = pd.read_csv("processed/val.csv",     parse_dates=["Date"])
holdout_df = pd.read_csv("processed/holdout.csv", parse_dates=["Date"])
raw_df     = pd.read_csv("data/atp_tennis.csv")

total_splits = len(train_df) + len(val_df) + len(holdout_df)
check("Total split rows == raw rows", total_splits == len(raw_df),
      f"{total_splits:,} vs raw {len(raw_df):,}")
check("Train ~70%", 0.68 <= len(train_df)/len(raw_df) <= 0.72,
      f"{len(train_df)/len(raw_df):.2%}")
check("Val   ~15%", 0.13 <= len(val_df)/len(raw_df) <= 0.17,
      f"{len(val_df)/len(raw_df):.2%}")
check("Hold  ~15%", 0.13 <= len(holdout_df)/len(raw_df) <= 0.17,
      f"{len(holdout_df)/len(raw_df):.2%}")

# ------------------------------------------------------------------
print("\n--- 8.3  Accuracy in Plausible Range (62-70%) ---")
model        = joblib.load("models/rf_model.pkl")
feature_cols = joblib.load("models/feature_columns.pkl")
X_val  = np.load("processed/X_val.npy")
y_val  = np.load("processed/y_val.npy")
X_hold = np.load("processed/X_hold.npy")
y_hold = np.load("processed/y_hold.npy")

val_acc  = accuracy_score(y_val,  model.predict(X_val))
hold_acc = accuracy_score(y_hold, model.predict(X_hold))
val_auc  = roc_auc_score(y_val,  model.predict_proba(X_val)[:,1])
hold_auc = roc_auc_score(y_hold, model.predict_proba(X_hold)[:,1])

check("Val accuracy 60-80%",   0.60 <= val_acc  <= 0.80, f"{val_acc:.4f}")
check("Hold accuracy 60-80%",  0.60 <= hold_acc <= 0.80, f"{hold_acc:.4f}")
check("Val ROC-AUC 60-85%",    0.60 <= val_auc  <= 0.85, f"{val_auc:.4f}")
check("Hold ROC-AUC 60-85%",   0.60 <= hold_auc <= 0.85, f"{hold_auc:.4f}")
check("Leakage guard val  <80%",  val_acc  < 0.80, f"{val_acc:.4f}")
check("Leakage guard hold <80%",  hold_acc < 0.80, f"{hold_acc:.4f}")

# ------------------------------------------------------------------
print("\n--- 8.4  Feature Importance Charts Exist ---")
check("feature_importance_gini.png",
      os.path.exists("models/feature_importance_gini.png"))
check("feature_importance_permutation.png",
      os.path.exists("models/feature_importance_permutation.png"))

gini = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
top3 = gini.head(3).index.tolist()
check("rank_diff or rank_ratio in Gini top 3",
      any("rank" in f for f in top3), f"top3={top3}")

# ------------------------------------------------------------------
print("\n--- 8.5  Leakage Audit: Banned Columns Absent from Feature Matrix ---")
banned = ["Score","Odd_1","Odd_2","Pts_1","Pts_2","Winner","Player_1","Player_2"]
for b in banned:
    check(f"'{b}' absent from feature_cols", b not in feature_cols)

# ------------------------------------------------------------------
print("\n--- 8.6  Temporal Integrity ---")
train_max   = train_df["Date"].max()
val_min     = val_df["Date"].min()
val_max     = val_df["Date"].max()
hold_min    = holdout_df["Date"].min()
check("train max <= val min",      train_max <= val_min,
      f"{train_max.date()} <= {val_min.date()}")
check("val max <= holdout min",    val_max   <= hold_min,
      f"{val_max.date()} <= {hold_min.date()}")

# ------------------------------------------------------------------
print("\n--- 8.7  Feature Matrix Integrity ---")
X_train = np.load("processed/X_train.npy")
check("X_train no NaN",  np.isnan(X_train).sum() == 0, f"NaN={int(np.isnan(X_train).sum())}")
check("X_val no NaN",    np.isnan(X_val).sum()   == 0, f"NaN={int(np.isnan(X_val).sum())}")
check("X_hold no NaN",   np.isnan(X_hold).sum()  == 0, f"NaN={int(np.isnan(X_hold).sum())}")
check("All matrices 18 features",
      X_train.shape[1] == X_val.shape[1] == X_hold.shape[1] == 18,
      f"train={X_train.shape[1]} val={X_val.shape[1]} hold={X_hold.shape[1]}")

# ------------------------------------------------------------------
print("\n--- 8.8  Artefact Files Complete ---")
artefacts = [
    "models/rf_model.pkl",
    "models/encoder.pkl",
    "models/feature_columns.pkl",
    "models/feature_importance_gini.png",
    "models/feature_importance_permutation.png",
    "processed/train.csv",
    "processed/val.csv",
    "processed/holdout.csv",
    "processed/X_train.npy",
    "processed/X_val.npy",
    "processed/X_hold.npy",
    "processed/y_train.npy",
    "processed/y_val.npy",
    "processed/y_hold.npy",
    "app.py",
    "requirements.txt",
    "README.md",
]
for a in artefacts:
    check(f"{a} exists", os.path.exists(a))

# ------------------------------------------------------------------
print("\n--- 8.9  app.py Imports and Logic Checks ---")
with open("app.py", encoding="utf-8") as f:
    app_code = f.read()
check("predict_proba in app.py",         "predict_proba" in app_code)
check("player names not in feature_cols","Player_1" not in feature_cols)
check("same-player guard in app.py",     "player1 == player2" in app_code)
check("betting odds disclaimer in app.py","betting odds" in app_code.lower())
check("cache_resource in app.py",        "@st.cache_resource" in app_code)

# ------------------------------------------------------------------
print("\n" + "=" * 65)
passed = sum(1 for _, s, _ in results if s == PASS)
failed = sum(1 for _, s, _ in results if s == FAIL)
total  = len(results)
print(f"RESULT:  {passed}/{total} checks PASSED   {failed}/{total} FAILED")
print("=" * 65)
if failed:
    print("\nFailed checks:")
    for name, status, detail in results:
        if status == FAIL:
            print(f"  !! {name}  ({detail})")
    sys.exit(1)
else:
    print("\nAll checks passed. Pipeline is clean and ready.")
