import numpy as np
import joblib
import os

feature_cols = joblib.load('models/feature_columns.pkl')
X_train = np.load('processed/X_train.npy')
X_val   = np.load('processed/X_val.npy')
X_hold  = np.load('processed/X_hold.npy')
y_train = np.load('processed/y_train.npy')
y_val   = np.load('processed/y_val.npy')
y_hold  = np.load('processed/y_hold.npy')

print('=== Shape Check ===')
print(f'X_train : {X_train.shape}')
print(f'X_val   : {X_val.shape}')
print(f'X_hold  : {X_hold.shape}')
print(f'y_train : {y_train.shape}')
print(f'y_val   : {y_val.shape}')
print(f'y_hold  : {y_hold.shape}')

print()
print('=== Column Count Consistency ===')
cols_ok = X_train.shape[1] == X_val.shape[1] == X_hold.shape[1]
print(f'All matrices have {X_train.shape[1]} columns: {"PASS" if cols_ok else "FAIL"}')

print()
print('=== Feature Names (18 expected) ===')
print(f'Count: {len(feature_cols)} -> {"PASS" if len(feature_cols) == 18 else "FAIL"}')
print(feature_cols)

print()
print('=== No-Name / No-Date Check ===')
banned_in_features = ['Player_1','Player_2','Winner','Date','Tournament',
                      'Score','Odd_1','Odd_2','Pts_1','Pts_2']
for b in banned_in_features:
    found = b in feature_cols
    print(f'  {b}: {"FAIL - PRESENT" if found else "PASS - absent"}')

print()
print('=== Numeric-Only Check ===')
import pandas as pd
df = pd.DataFrame(X_train, columns=feature_cols)
non_numeric = [c for c in df.columns if df[c].dtype == object]
print(f'  Non-numeric columns: {non_numeric if non_numeric else "None"} -> {"FAIL" if non_numeric else "PASS"}')

print()
print('=== NaN Check ===')
nan_train = int(np.isnan(X_train).sum())
nan_val   = int(np.isnan(X_val).sum())
nan_hold  = int(np.isnan(X_hold).sum())
print(f'  X_train NaNs : {nan_train} -> {"FAIL" if nan_train else "PASS"}')
print(f'  X_val   NaNs : {nan_val}   -> {"FAIL" if nan_val else "PASS"}')
print(f'  X_hold  NaNs : {nan_hold}  -> {"FAIL" if nan_hold else "PASS"}')

print()
print('=== Target Balance ===')
import collections
tc = collections.Counter(y_train.tolist())
print(f'  y_train: {dict(tc)}')
vc = collections.Counter(y_val.tolist())
print(f'  y_val  : {dict(vc)}')

print()
print('=== Artefact Files ===')
for f in ['models/encoder.pkl','models/feature_columns.pkl',
          'processed/X_train.npy','processed/y_train.npy',
          'processed/X_val.npy','processed/y_val.npy',
          'processed/X_hold.npy','processed/y_hold.npy']:
    exists = os.path.exists(f)
    print(f'  {f}: {"PASS" if exists else "FAIL - MISSING"}')

print()
print('=== Key Feature Sanity (rank_diff / rank_ratio present) ===')
for key in ['rank_diff', 'rank_ratio', 'round_num', 'best_of']:
    print(f'  {key}: {"PASS" if key in feature_cols else "FAIL"}')
