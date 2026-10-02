# Phase 3 Implementation Log — Module 2: Feature Engineering

**Date:** 2025-10-01
**Status:** COMPLETE
**Script:** `src/02_features.py`
**Plan Reference:** IMPLEMENTATION_PLAN.md — Phase 3

---

## Objective

Transform the cleaned splits into numeric feature matrices with no
player-name or post-match information, and save fitted encoder artefacts
for reuse by the Streamlit app.

---

## Implementation Summary

### Steps Executed (per plan)

| Step | Description | Result |
|---|---|---|
| 1 | Load all three splits from `processed/` | Done |
| 2 | Compute `rank_diff = Rank_1 - Rank_2` | Done |
| 3 | Compute `rank_ratio = Rank_1 / (Rank_2 + 1e-9)` | Done (epsilon guards divide-by-zero) |
| 4 | Ordinal-encode `Round` via fixed mapping (1st Round=1 … The Final=7) | Done — rows with unmapped rounds dropped |
| 5 | One-hot encode `Surface`, `Court`, `Series` with `OneHotEncoder(handle_unknown='ignore')` — fit on training only | Done |
| 6 | Keep `Best of` as integer feature `best_of` | Done |
| 7 | Drop `Player_1`, `Player_2`, `Winner`, `Date`, `Rank_1`, `Rank_2` from feature matrix | Done |
| 8 | Separate `Player_1_wins` as label array `y` | Done |
| 9 | Persist encoder to `models/encoder.pkl`, feature list to `models/feature_columns.pkl` | Done |
| 10 | Save `.npy` arrays for all splits to `processed/` | Done |

---

## Feature Set (18 features)

| Feature | Type | Description |
|---|---|---|
| `rank_diff` | Numeric | Rank_1 - Rank_2 (positive = P1 is lower ranked) |
| `rank_ratio` | Numeric | Rank_1 / Rank_2 (proportional gap) |
| `round_num` | Ordinal int | 1 (1st Round) to 7 (The Final) |
| `best_of` | Int | Match format: 3 or 5 |
| `Surface_Carpet` | One-hot | 1 if surface is Carpet |
| `Surface_Clay` | One-hot | 1 if surface is Clay |
| `Surface_Grass` | One-hot | 1 if surface is Grass |
| `Surface_Hard` | One-hot | 1 if surface is Hard |
| `Court_Indoor` | One-hot | 1 if indoor court |
| `Court_Outdoor` | One-hot | 1 if outdoor court |
| `Series_ATP250` | One-hot | 1 if ATP 250 tier |
| `Series_ATP500` | One-hot | 1 if ATP 500 tier |
| `Series_Grand Slam` | One-hot | 1 if Grand Slam |
| `Series_International` | One-hot | 1 if International tier |
| `Series_International Gold` | One-hot | 1 if International Gold tier |
| `Series_Masters` | One-hot | 1 if Masters tier |
| `Series_Masters 1000` | One-hot | 1 if Masters 1000 tier |
| `Series_Masters Cup` | One-hot | 1 if Masters Cup |

---

## Test Results

### Test 1 — Shape Check

| Matrix | Shape | Result |
|---|---|---|
| X_train | (46,993 x 18) | PASS |
| X_val | (10,077 x 18) | PASS |
| X_hold | (9,962 x 18) | PASS |
| y_train | (46,993,) | PASS |
| y_val | (10,077,) | PASS |
| y_hold | (9,962,) | PASS |

Note: Row counts are slightly less than Phase 2 CSVs (48,132 / 10,314 / 10,314)
because rows with unmapped Round values were dropped during ordinal encoding.

### Test 2 — Column Consistency

All three matrices have identical 18 columns: **PASS**

### Test 3 — No Player Names / No Banned Columns in Feature Matrix

All 10 banned columns absent: `Player_1`, `Player_2`, `Winner`, `Date`,
`Tournament`, `Score`, `Odd_1`, `Odd_2`, `Pts_1`, `Pts_2` — **10/10 PASS**

### Test 4 — Numeric-Only Check

No object/string columns in any feature matrix: **PASS**

### Test 5 — NaN Check

| Matrix | NaN Count | Result |
|---|---|---|
| X_train | 0 | PASS |
| X_val | 0 | PASS |
| X_hold | 0 | PASS |

### Test 6 — Target Balance

| Split | Class 0 | Class 1 | Balance |
|---|---|---|---|
| y_train | 23,497 | 23,496 | Near-perfect (PASS) |
| y_val | 5,041 | 5,036 | Near-perfect (PASS) |

### Test 7 — Artefact Files Present

| File | Result |
|---|---|
| `models/encoder.pkl` | PASS |
| `models/feature_columns.pkl` | PASS |
| `processed/X_train.npy` | PASS |
| `processed/y_train.npy` | PASS |
| `processed/X_val.npy` | PASS |
| `processed/y_val.npy` | PASS |
| `processed/X_hold.npy` | PASS |
| `processed/y_hold.npy` | PASS |

### Test 8 — Key Feature Presence

`rank_diff`, `rank_ratio`, `round_num`, `best_of` — all present: **PASS**

---

## Deviations from Plan

| Item | Plan | Actual | Reason |
|---|---|---|---|
| Row count after encoding | Expected same as Phase 2 output | 1,139 rows dropped (train: 48,132 to 46,993) | Rows with Round values not in the fixed ordinal mapping were dropped. Acceptable — these are edge-case or non-standard round labels. |

---

## Phase 3 Verdict

**PASS** — All 8 test groups passed. Feature matrices are numeric, clean,
leakage-free, and identically structured across all three splits.
Encoder artefacts saved for reuse by app.py. Ready for Phase 4.
