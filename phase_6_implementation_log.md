# Phase 6 Implementation Log — Module 5: Live Prediction Challenge

**Date:** 2025-10-01
**Status:** COMPLETE
**Script:** `src/05_predict.py`
**Plan Reference:** IMPLEMENTATION_PLAN.md — Phase 6

---

## Objective

Simulate a live prediction scenario by running the trained model against
the hold-out set (matches never seen during training) and comparing
predictions to actual outcomes.

---

## Implementation Summary

### Steps Executed (per plan)

| Step | Description | Result |
|---|---|---|
| 1 | Load rf_model.pkl, encoder.pkl, feature_columns.pkl | Done |
| 2 | Load processed/holdout.csv | Done — 10,314 rows |
| 3 | Apply fitted feature engineering pipeline (no refitting) | Done |
| 4 | Select first 10 hold-out matches chronologically | Done |
| 5 | For each match: print names, surface, round, probabilities, actual winner, result | Done |
| 6 | Run predict on full X_hold, print overall accuracy | Done |
| 7 | Print retrospective simulation disclaimer | Done |

---

## Live Prediction Challenge — 10 Match Results

| # | Player 1 | Player 2 | Surface | Round | P(P1) | P(P2) | Actual Winner | Result |
|---|---|---|---|---|---|---|---|---|
| 1 | Tabilo A. | Wolf J.J. | Hard | 2nd Round | 0.617 | 0.383 | Wolf J.J. | MISS |
| 2 | Moutet C. | Van De Zandschulp B. | Hard | 2nd Round | 0.151 | 0.849 | Moutet C. | MISS |
| 3 | Kyrgios N. | Bonzi B. | Hard | 2nd Round | 0.630 | 0.370 | Kyrgios N. | OK |
| 4 | Korda S. | Paul T. | Hard | 2nd Round | 0.481 | 0.519 | Paul T. | OK |
| 5 | Galan D.E. | Thompson J. | Hard | 2nd Round | 0.576 | 0.424 | Galan D.E. | OK |
| 6 | Nava E. | Murray A. | Hard | 2nd Round | 0.263 | 0.737 | Murray A. | OK |
| 7 | Davidovich Fokina A. | Fucsovics M. | Hard | 2nd Round | 0.682 | 0.318 | Davidovich Fokina A. | OK |
| 8 | Bublik A. | Carreno Busta P. | Hard | 2nd Round | 0.251 | 0.749 | Carreno Busta P. | OK |
| 9 | Borges N. | Wu Y. | Hard | 2nd Round | 0.669 | 0.331 | Wu Y. | MISS |
| 10 | Berrettini M. | Grenier H. | Hard | 2nd Round | 0.876 | 0.124 | Berrettini M. | OK |

**Sample accuracy: 70.0% (7/10 correct)**

### Notable Upsets Missed

- **Match 1:** Tabilo A. (rank better) vs Wolf J.J. — model predicted Tabilo, Wolf won
- **Match 2:** Moutet C. vs Van De Zandschulp B. — model strongly predicted Van De Zandschulp (84.9%), Moutet won (lower-ranked player upset)
- **Match 9:** Borges N. vs Wu Y. — model predicted Borges (66.9%), Wu won

These are genuine upsets — ranking-based models will always miss some upsets by
design. This is expected and healthy model behaviour.

---

## Full Hold-out Evaluation

| Metric | Value | Expected Range | Result |
|---|---|---|---|
| Accuracy | 0.6382 | 62–70% | PASS |
| ROC-AUC | 0.6969 | 60–85% | PASS |
| Leakage guard (< 80%) | 0.6382 < 0.80 | — | PASS |

---

## Test Results

### Test 1 — Temporal Separation (hold-out never seen in training)

| Check | Value | Result |
|---|---|---|
| Train max date | 2018-02-19 | — |
| Hold-out min date | 2022-08-31 | — |
| Gap | ~4.5 years | PASS |

### Test 2 — Full Hold-out Accuracy in Range
0.6382 within 58–80%: **PASS**

### Test 3 — ROC-AUC in Range
0.6969 within 58–85%: **PASS**

### Test 4 — Leakage Guard
Accuracy < 80%: **PASS**

### Test 5 — Probabilities in Valid Range
All predicted probabilities in [0, 1]: **PASS**

### Test 6 — 10-Sample Correctness
7/10 correct (70%): **PASS** (above 50% baseline)

### Test 7 — Retrospective Disclaimer Present
Script contains disclaimer: **PASS**

---

## Hold-out vs Validation Comparison

| Metric | Validation Set | Hold-out Set | Notes |
|---|---|---|---|
| Accuracy | 0.6245 | 0.6382 | Hold-out slightly better — consistent, no anomaly |
| ROC-AUC | 0.6825 | 0.6969 | Hold-out slightly better — consistent |

Hold-out performance is marginally better than validation, which is a healthy
sign — the model generalises well to unseen future data.

---

## Deviations from Plan

None. All planned steps executed as specified.

---

## Phase 6 Verdict

**PASS** — All 7 tests passed. The model correctly predicts 70% of the 10-sample
matches and achieves 63.8% accuracy on the full 10,314-match hold-out set.
No leakage detected. Ready for Phase 7.
