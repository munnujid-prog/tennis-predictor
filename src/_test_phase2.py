import pandas as pd

train   = pd.read_csv('processed/train.csv')
val     = pd.read_csv('processed/val.csv')
holdout = pd.read_csv('processed/holdout.csv')

print('=== Split Row Counts ===')
print(f'Train  : {len(train):,}')
print(f'Val    : {len(val):,}')
print(f'Holdout: {len(holdout):,}')
print(f'Total  : {len(train)+len(val)+len(holdout):,}')

print()
print('=== Columns in train ===')
print(list(train.columns))

print()
print('=== Leakage check (banned columns must NOT appear) ===')
banned = ['Score', 'Odd_1', 'Odd_2', 'Pts_1', 'Pts_2', 'Tournament']
for b in banned:
    status = 'FAIL - PRESENT' if b in train.columns else 'PASS - absent'
    print(f'  {b}: {status}')

print()
print('=== Target distribution (train) ===')
print(train['Player_1_wins'].value_counts().to_dict())

print()
print('=== Null check on Rank_1 / Rank_2 ===')
r1 = train['Rank_1'].isna().sum()
r2 = train['Rank_2'].isna().sum()
print(f'  Rank_1 nulls: {r1}')
print(f'  Rank_2 nulls: {r2}')

print()
print('=== Date range check ===')
train['Date']   = pd.to_datetime(train['Date'])
val['Date']     = pd.to_datetime(val['Date'])
holdout['Date'] = pd.to_datetime(holdout['Date'])
print(f'  Train   max date : {train["Date"].max().date()}')
print(f'  Val     min date : {val["Date"].min().date()}')
print(f'  Val     max date : {val["Date"].max().date()}')
print(f'  Holdout min date : {holdout["Date"].min().date()}')
ok = train['Date'].max() <= val['Date'].min() and val['Date'].max() <= holdout['Date'].min()
print(f'  Temporal order   : {"PASS" if ok else "FAIL"}')

print()
print('=== Retirement filter check (Score column absent) ===')
score_absent = 'Score' not in train.columns
print(f'  Score column absent: {"PASS" if score_absent else "FAIL"}')
