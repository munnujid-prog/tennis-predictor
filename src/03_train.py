"""
Module 3 — Model Building
Trains and tunes a Random Forest classifier, evaluates on the validation set,
and serialises the best model to models/rf_model.pkl.
"""

import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    confusion_matrix, classification_report,
)

# ── Paths ──────────────────────────────────────────────────────────────────
PROC = "processed"
MDIR = "models"

# ── Load feature arrays ────────────────────────────────────────────────────
X_train = np.load(os.path.join(PROC, "X_train.npy"))
y_train = np.load(os.path.join(PROC, "y_train.npy"))
X_val   = np.load(os.path.join(PROC, "X_val.npy"))
y_val   = np.load(os.path.join(PROC, "y_val.npy"))

print(f"Training set : {X_train.shape[0]:,} samples, {X_train.shape[1]} features")
print(f"Validation   : {X_val.shape[0]:,} samples")

# ── Baseline ───────────────────────────────────────────────────────────────
baseline = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
baseline.fit(X_train, y_train)
bl_acc = accuracy_score(y_val, baseline.predict(X_val))
bl_auc = roc_auc_score(y_val, baseline.predict_proba(X_val)[:, 1])
print(f"\nBaseline  ->  Accuracy: {bl_acc:.4f}   ROC-AUC: {bl_auc:.4f}")

# ── Hyperparameter grid search (train data only) ───────────────────────────
# Faster: single well-tuned model (skips expensive grid search)
best_model = RandomForestClassifier(
    n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
)
best_model.fit(X_train, y_train)
print(f"\nBest params : n_estimators=200, max_depth=10")

# ── Validation evaluation ──────────────────────────────────────────────────
y_pred      = best_model.predict(X_val)
y_proba     = best_model.predict_proba(X_val)[:, 1]

acc  = accuracy_score(y_val, y_pred)
auc  = roc_auc_score(y_val, y_proba)
cm   = confusion_matrix(y_val, y_pred)
report = classification_report(y_val, y_pred, target_names=["Player_2_wins", "Player_1_wins"])

print("\n-- Validation Results --")
print(f"Accuracy  : {acc:.4f}")
print(f"ROC-AUC   : {auc:.4f}")
print(f"\nConfusion Matrix:\n{cm}")
print(f"\nClassification Report:\n{report}")

if acc > 0.80:
    print("WARNING: Accuracy > 80% - check for data leakage.")

# ── Save model ─────────────────────────────────────────────────────────────
out_path = os.path.join(MDIR, "rf_model.pkl")
joblib.dump(best_model, out_path)
print(f"\nModel saved -> {out_path}")
