import os
import numpy as np
import joblib
from PIL import Image

print('=== Chart Files Exist ===')
for f in ['models/feature_importance_gini.png',
          'models/feature_importance_permutation.png']:
    exists = os.path.exists(f)
    print(f'  {f}: {"PASS" if exists else "FAIL - MISSING"}')
    if exists:
        img = Image.open(f)
        print(f'    Size: {img.size[0]}x{img.size[1]} px')

print()
print('=== Rank Features in Top 3 (Gini) ===')
model        = joblib.load('models/rf_model.pkl')
feature_cols = joblib.load('models/feature_columns.pkl')
import pandas as pd
gini = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
top3 = gini.head(3).index.tolist()
print(f'  Top 3: {top3}')
rank_in_top3 = any('rank' in f for f in top3)
print(f'  Rank feature in top 3: {"PASS" if rank_in_top3 else "FAIL"}')

print()
print('=== Gini Importances Sum to 1 ===')
total = model.feature_importances_.sum()
print(f'  Sum = {total:.6f} -> {"PASS" if abs(total - 1.0) < 1e-6 else "FAIL"}')

print()
print('=== Top 5 Gini Features ===')
for i, (feat, imp) in enumerate(gini.head(5).items(), 1):
    print(f'  {i}. {feat}: {imp:.4f}')

print()
print('=== Permutation Importance Computed ===')
from sklearn.inspection import permutation_importance
X_val = np.load('processed/X_val.npy')
y_val = np.load('processed/y_val.npy')
perm = permutation_importance(model, X_val, y_val, n_repeats=5, random_state=42, n_jobs=-1)
perm_mean = pd.Series(perm.importances_mean, index=feature_cols).sort_values(ascending=False)
print(f'  Top permutation feature: {perm_mean.index[0]} ({perm_mean.iloc[0]:.4f})')
rank_top_perm = any('rank' in f for f in perm_mean.head(2).index.tolist())
print(f'  Rank feature in top 2 permutation: {"PASS" if rank_top_perm else "FAIL"}')
