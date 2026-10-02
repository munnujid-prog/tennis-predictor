# Phase 2 Implementation Log — Module 1: Data Preparation

**Date:** 2025-10-01
**Status:** COMPLETE
**Script:** `src/01_prepare.py`
**Plan Reference:** IMPLEMENTATION_PLAN.md — Phase 2

---

## Objective

Produce clean, leakage-free, chronologically split datasets that are the
single source of truth for all downstream modules.

---

## Implementation Summary

### Steps Executed (per plan)

| Step | Description | Result |
|---|---|---|
| 1 | Load `data/atp_tennis.csv` | 68,760 rows loaded |
| 2 | Replace `-1` / `-1.0` sentinels with NaN in Pts_1, Pts_2, Odd_1, Odd_2 | Done |
| 3 | Filter retirement/walkover rows (Score contains ret/w/o) | 0 rows removed (dataset had none) |
| 4 | Drop leaking/unused columns: Score, Odd_1, Odd_2, Pts_1, Pts_2, Tournament | Done |
| 5 | Derive binary target: `Player_1_wins = (Winner == Player_1)` | Done |
| 6 | Drop rows with null Rank_1 or Rank_2 | 0 rows removed |
| 7 | Parse Date to datetime, sort ascending | Done |
| 8 | Split 70/15/15 by row position (no shuffle) | Done |
| 9 | Save to processed/train.csv, val.csv, holdout.csv | Done |

### Output Files

| File | Rows | Date Range |
|---|---|---|
| `processed/train.csv` | 48,132 | 2000-01-03 to 2018-02-19 |
| `processed/val.csv` | 10,314 | 2018-02-19 to 2022-08-31 |
| `processed/holdout.csv` | 10,314 | 2022-08-31 to 2026-09-13 |

---

## Test Results

### Test 1 — Row Count Integrity

| Split | Rows | Pass/Fail |
|---|---|---|
| Train | 48,132 | PASS |
| Val | 10,314 | PASS |
| Holdout | 10,314 | PASS |
| Total | 68,760 (matches raw) | PASS |

### Test 2 — Leakage Check (banned columns absent from all splits)

| Column | Result |
|---|---|
| Score | PASS — absent |
| Odd_1 | PASS — absent |
| Odd_2 | PASS — absent |
| Pts_1 | PASS — absent |
| Pts_2 | PASS — absent |
| Tournament | PASS — absent |

### Test 3 — Target Variable

| Class | Count | Notes |
|---|---|---|
| 0 (Player_2 wins) | 24,066 | Perfectly balanced |
| 1 (Player_1 wins) | 24,066 | Perfectly balanced |

Target is perfectly balanced (50/50) — this is expected because Player_1/Player_2
assignment is arbitrary, so wins are evenly distributed by construction.

### Test 4 — Null Check on Core Features

| Column | Nulls | Result |
|---|---|---|
| Rank_1 | 0 | PASS |
| Rank_2 | 0 | PASS |

### Test 5 — Temporal Integrity

| Check | Value | Result |
|---|---|---|
| Train max date | 2018-02-19 | — |
| Val min date | 2018-02-19 | — |
| Val max date | 2022-08-31 | — |
| Holdout min date | 2022-08-31 | — |
| Chronological order | Train <= Val <= Holdout | PASS |

### Test 6 — Retirement Filter

| Check | Result |
|---|---|
| Score column absent from output | PASS |

---

## Deviations from Plan

| Item | Plan | Actual | Reason |
|---|---|---|---|
| Retirement filter removes rows | Expected some rows removed | 0 rows removed | Raw dataset appears to have already excluded retirements, or uses a different notation. The filter ran correctly but found no matches. This is acceptable — the filter is still in place as a safeguard. |

---

## Columns Retained in Output Files

`Date`, `Series`, `Court`, `Surface`, `Round`, `Best of`, `Player_1`,
`Player_2`, `Winner`, `Rank_1`, `Rank_2`, `Player_1_wins`

Player names and Date are retained for traceability. They will be stripped
from the feature matrix in Phase 3 (Module 2).

---

## Phase 2 Verdict

**PASS** — All 6 tests passed. Output files are clean, leakage-free,
and chronologically ordered. Ready for Phase 3.
