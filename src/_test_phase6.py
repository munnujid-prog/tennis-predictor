import numpy as np
import pandas as pd
import joblib
import os
from sklearn.metrics import accuracy_score, roc_auc_score

print('=== Hold-out Data Not Used in Training (design check) ===')
train   = pd.read_csv('processed/train.csv',   parse_dates=['Date'])
holdout = pd.read_csv('processed/holdout.csv', parse_dates=['Date'])
train_max   = train['Date'].max()
holdout_min = holdout['Date'].min()
print(f'  Train max date   : {train_max.date()}')
print(f'  Holdout min date : {holdout_min.date()}')
print(f'  Temporal separation: {"PASS" if train_max <= holdout_min else "FAIL"}')

print()
print('=== Full Hold-out Accuracy ===')
model        = joblib.load('models/rf_model.pkl')
feature_cols = joblib.load('models/feature_columns.pkl')
X_hold = np.load('processed/X_hold.npy')
y_hold = np.load('processed/y_hold.npy')
y_pred  = model.predict(X_hold)
y_proba = model.predict_proba(X_hold)[:, 1]
acc = accuracy_score(y_hold, y_pred)
auc = roc_auc_score(y_hold, y_proba)
print(f'  Accuracy : {acc:.4f} -> {"PASS" if 0.58 <= acc <= 0.80 else "FAIL"}')
print(f'  ROC-AUC  : {auc:.4f} -> {"PASS" if 0.58 <= auc <= 0.85 else "FAIL"}')
print(f'  Leakage guard (< 80%): {"PASS" if acc < 0.80 else "FAIL"}')

print()
print('=== Probabilities in [0, 1] range ===')
proba_ok = bool((y_proba >= 0).all() and (y_proba <= 1).all())
print(f'  All probabilities in [0,1]: {"PASS" if proba_ok else "FAIL"}')

print()
print('=== 10-Sample Prediction Correctness ===')
ROUND_MAP = {'1st Round':1,'2nd Round':2,'3rd Round':3,'Round of 16':4,
             'Quarterfinals':5,'Semifinals':6,'The Final':7}
CAT_COLS  = ['Surface','Court','Series']
encoder   = joblib.load('models/encoder.pkl')

sample = holdout.head(10).copy()
sample['round_num'] = sample['Round'].map(ROUND_MAP)
sample.dropna(subset=['round_num'], inplace=True)
sample['rank_diff']  = sample['Rank_1'] - sample['Rank_2']
sample['rank_ratio'] = sample['Rank_1'] / (sample['Rank_2'] + 1e-9)
sample['best_of']    = sample['Best of'].astype(int)

cat_enc = encoder.transform(sample[CAT_COLS])
cat_df  = pd.DataFrame(cat_enc, columns=encoder.get_feature_names_out(CAT_COLS))
num_df  = sample[['rank_diff','rank_ratio','round_num','best_of']].reset_index(drop=True)
X_s     = pd.concat([num_df, cat_df.reset_index(drop=True)], axis=1)[feature_cols].values

probas  = model.predict_proba(X_s)
correct = 0
for i, (_, row) in enumerate(sample.iterrows()):
    pred = row['Player_1'] if probas[i][1] >= 0.5 else row['Player_2']
    if pred == row['Winner']:
        correct += 1
print(f'  10-sample correct: {correct}/10  ({correct/10:.0%})')
print(f'  Sample >= 50%: {"PASS" if correct >= 5 else "NOTE: below 50% on this sample"}')

print()
print('=== Disclaimer Present in Script ===')
with open('src/05_predict.py') as f:
    content = f.read()
retrospective = 'retrospective' in content.lower()
print(f'  Retrospective disclaimer in script: {"PASS" if retrospective else "FAIL"}')
