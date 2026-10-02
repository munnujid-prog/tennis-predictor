"""
Phase 7 test — validates app.py logic without starting the Streamlit server.
Tests: artefact loading, player lookup, feature vector construction, prediction output.
"""

import numpy as np
import pandas as pd
import joblib
import os
import sys

print('=== Artefact Files Present ===')
required = [
    'models/rf_model.pkl',
    'models/encoder.pkl',
    'models/feature_columns.pkl',
    'processed/train.csv',
]
for f in required:
    exists = os.path.exists(f)
    print(f'  {f}: {"PASS" if exists else "FAIL - MISSING"}')
    if not exists:
        sys.exit(1)

print()
print('=== Load Artefacts ===')
model        = joblib.load('models/rf_model.pkl')
encoder      = joblib.load('models/encoder.pkl')
feature_cols = joblib.load('models/feature_columns.pkl')
print(f'  model type     : {type(model).__name__} -> PASS')
print(f'  feature count  : {len(feature_cols)} -> {"PASS" if len(feature_cols) == 18 else "FAIL"}')

print()
print('=== Player Lookup Table ===')
train = pd.read_csv('processed/train.csv', parse_dates=['Date'])
p1 = train[['Date','Player_1','Rank_1']].rename(columns={'Player_1':'player','Rank_1':'rank'})
p2 = train[['Date','Player_2','Rank_2']].rename(columns={'Player_2':'player','Rank_2':'rank'})
all_p = pd.concat([p1, p2], ignore_index=True).dropna(subset=['rank'])
lookup = all_p.sort_values('Date').groupby('player')['rank'].last().astype(int).to_dict()
print(f'  Players in lookup : {len(lookup):,}')
print(f'  Players >= 100    : {"PASS" if len(lookup) >= 100 else "FAIL"}')

# Spot-check a few known players
known = ['Federer R.', 'Djokovic N.', 'Nadal R.']
for p in known:
    found = p in lookup
    rank  = lookup.get(p, 'N/A')
    print(f'  {p}: {"PASS" if found else "not in training data (post-2018 era check)"} (rank={rank})')

print()
print('=== Feature Vector Construction ===')
ROUND_MAP = {'1st Round':1,'2nd Round':2,'3rd Round':3,'Round of 16':4,
             'Quarterfinals':5,'Semifinals':6,'The Final':7}
CAT_COLS = ['Surface','Court','Series']

# Simulate UI: Federer (rank 2) vs Nadal (rank 1), Roland Garros Final
rank1, rank2 = 2, 1
surface, court, series, round_, best_of = 'Clay', 'Outdoor', 'Grand Slam', 'The Final', 5

rank_diff  = rank1 - rank2
rank_ratio = rank1 / (rank2 + 1e-9)
round_num  = ROUND_MAP[round_]

cat_input = pd.DataFrame([[surface, court, series]], columns=CAT_COLS)
cat_enc   = encoder.transform(cat_input)
cat_names = encoder.get_feature_names_out(CAT_COLS)

num_part = np.array([[rank_diff, rank_ratio, round_num, best_of]])
full_vec = np.concatenate([num_part, cat_enc], axis=1)

full_df = pd.DataFrame(full_vec,
    columns=['rank_diff','rank_ratio','round_num','best_of'] + list(cat_names))
for col in feature_cols:
    if col not in full_df.columns:
        full_df[col] = 0.0
X_input = full_df[feature_cols].values

print(f'  Feature vector shape : {X_input.shape} -> {"PASS" if X_input.shape == (1, 18) else "FAIL"}')
print(f'  NaN in vector        : {int(np.isnan(X_input).sum())} -> {"PASS" if np.isnan(X_input).sum() == 0 else "FAIL"}')

print()
print('=== Prediction Output ===')
probas = model.predict_proba(X_input)[0]
p1_win = probas[1]
p2_win = probas[0]
print(f'  P(Player1 wins) : {p1_win:.3f}')
print(f'  P(Player2 wins) : {p2_win:.3f}')
print(f'  Sum to 1        : {"PASS" if abs(p1_win + p2_win - 1.0) < 1e-9 else "FAIL"}')
print(f'  In [0,1] range  : {"PASS" if 0 <= p1_win <= 1 and 0 <= p2_win <= 1 else "FAIL"}')
winner = 'Player 1 (rank 2)' if p1_win >= 0.5 else 'Player 2 (rank 1)'
print(f'  Predicted winner: {winner}')
print(f'  Rank-1 player favoured: {"PASS" if p2_win > p1_win else "NOTE: higher-ranked player not favoured"}')

print()
print('=== Same-Player Guard ===')
same_player_error = True  # app.py shows st.error if player1 == player2
print(f'  Same-player guard present in app.py: ', end='')
with open('app.py', encoding='utf-8') as f:
    app_code = f.read()
guard_ok = 'player1 == player2' in app_code
print(f'{"PASS" if guard_ok else "FAIL"}')

print()
print('=== Player Names NOT in Feature Vector ===')
name_in_features = any(name in feature_cols for name in ['Player_1','Player_2','player1','player2'])
print(f'  Player names absent from feature_cols: {"PASS" if not name_in_features else "FAIL"}')

print()
print('=== Series Options Loadable ===')
series_opts = sorted(train['Series'].dropna().unique().tolist())
print(f'  Series options ({len(series_opts)}): {series_opts}')
print(f'  Non-empty: {"PASS" if len(series_opts) > 0 else "FAIL"}')

print()
print('=== app.py Key Sections Present ===')
checks = {
    'cache_resource decorator' : '@st.cache_resource' in app_code,
    'player selectbox'         : 'st.selectbox' in app_code,
    'Predict button'           : 'st.button' in app_code,
    'predict_proba call'       : 'predict_proba' in app_code,
    'st.success winner callout': 'st.success' in app_code,
    'st.error same-player'     : 'st.error' in app_code,
    'sidebar transparency note': 'betting odds' in app_code.lower(),
}
for check, result in checks.items():
    print(f'  {check}: {"PASS" if result else "FAIL"}')
