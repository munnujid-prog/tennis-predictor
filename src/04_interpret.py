"""
Module 4 - Model Interpretation
Plots Gini and permutation feature importances, prints a comparison table,
and saves charts to models/.
"""

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

# -- Paths ------------------------------------------------------------------
PROC = "processed"
MDIR = "models"

# -- Load artefacts ---------------------------------------------------------
model        = joblib.load(os.path.join(MDIR, "rf_model.pkl"))
feature_cols = joblib.load(os.path.join(MDIR, "feature_columns.pkl"))
X_val        = np.load(os.path.join(PROC, "X_val.npy"))
y_val        = np.load(os.path.join(PROC, "y_val.npy"))

# -- Gini importances -------------------------------------------------------
gini_imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(10, max(5, len(feature_cols) * 0.35)))
gini_imp.sort_values().plot.barh(ax=ax, color="steelblue")
ax.set_title("Feature Importances - Gini (Built-in)")
ax.set_xlabel("Mean Decrease in Impurity")
plt.tight_layout()
gini_path = os.path.join(MDIR, "feature_importance_gini.png")
plt.savefig(gini_path, dpi=120)
plt.close()
print(f"Gini chart saved -> {gini_path}")

# -- Permutation importances ------------------------------------------------
print("Computing permutation importances (this may take ~30s) ...")
perm = permutation_importance(model, X_val, y_val, n_repeats=10, random_state=42, n_jobs=-1)
perm_mean = pd.Series(perm.importances_mean, index=feature_cols).sort_values(ascending=False)
perm_std  = pd.Series(perm.importances_std,  index=feature_cols)

fig, ax = plt.subplots(figsize=(10, max(5, len(feature_cols) * 0.35)))
sorted_idx = perm_mean.sort_values().index
ax.barh(
    sorted_idx,
    perm_mean[sorted_idx],
    xerr=perm_std[sorted_idx],
    color="darkorange",
    capsize=3,
)
ax.set_title("Feature Importances - Permutation (Validation Set)")
ax.set_xlabel("Mean Decrease in Accuracy")
plt.tight_layout()
perm_path = os.path.join(MDIR, "feature_importance_permutation.png")
plt.savefig(perm_path, dpi=120)
plt.close()
print(f"Permutation chart saved -> {perm_path}")

# -- Comparison table -------------------------------------------------------
gini_ranks = gini_imp.rank(ascending=False).astype(int)
perm_ranks = perm_mean.rank(ascending=False).astype(int)

cmp = pd.DataFrame({
    "Gini Importance": gini_imp.round(4),
    "Gini Rank":       gini_ranks,
    "Perm Importance": perm_mean.round(4),
    "Perm Rank":       perm_ranks,
}).sort_values("Gini Rank")

print("\n-- Feature Importance Comparison --")
print(cmp.to_string())

# -- Commentary -------------------------------------------------------------
top3_gini = gini_imp.head(3).index.tolist()
print("\n-- Commentary --")
print(f"Top 3 features (Gini): {top3_gini}")

if any("rank" in f for f in top3_gini):
    print("OK: Rank-based features dominate - expected behaviour.")
else:
    print("WARNING: Rank features not in top 3 - investigate possible leakage.")

near_zero = perm_mean[perm_mean <= 0].index.tolist()
if near_zero:
    print(f"Features with near-zero/negative permutation importance (pruning candidates): {near_zero}")
else:
    print("All features contribute positively to permutation importance.")
