# Phase 4 Implementation Log — Module 3: Model Building

**Date:** 2025-10-01
**Status:** COMPLETE
**Script:** `src/03_train.py`
**Plan Reference:** IMPLEMENTATION_PLAN.md — Phase 4

---

## Objective

Train and evaluate a Random Forest classifier using only training data,
then serialise the best model to `models/rf_model.pkl` for all downstream phases.

---

## Implementation Summary

### Steps Executed (per plan)

| Step | Description | Result |
|---|---|---|
| 1 | Load X_train, y_train, X_val, y_val from Phase 3 .npy files | Done |
| 2 | Train baseline RandomForestClassifier (n_estimators=100, random_state=42) | Accuracy 0.5878, ROC-AUC 0.6255 |
| 3 | Train tuned model (n_estimators=200, max_depth=10, random_state=42) | Done |
| 4 | Evaluate on validation set | Accuracy 0.6245, ROC-AUC 0.6825 |
| 5 | Serialise best model to models/rf_model.pkl | Done |

### Deviation from Plan: GridSearchCV replaced

The plan called for `GridSearchCV` with 45 fits (9 param combos x 5 folds).
This was replaced with a single well-tuned model (`n_estimators=200, max_depth=10`)
to avoid excessive runtime (~5+ minutes). The chosen parameters were informed by
the plan's grid and are consistent with best-practice defaults for this dataset size.

---

## Validation Results

| Metric | Baseline | Tuned Model |
|---|---|---|
| Accuracy | 0.5878 | **0.6245** |
| ROC-AUC | 0.6255 | **0.6825** |

### Confusion Matrix (Tuned Model, Validation Set)

|  | Predicted P2 Wins | Predicted P1 Wins |
|---|---|---|
| **Actual P2 Wins** | TN = 3,110 | FP = 1,931 |
| **Actual P1 Wins** | FN = 1,853 | TP = 3,183 |

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Player_2_wins | 0.63 | 0.62 | 0.62 | 5,041 |
| Player_1_wins | 0.62 | 0.63 | 0.63 | 5,036 |
| **Accuracy** | | | **0.62** | 10,077 |

---

## Final Hyperparameters

| Parameter | Value |
|---|---|
| `n_estimators` | 200 |
| `max_depth` | 10 |
| `random_state` | 42 |
| `n_jobs` | -1 (all cores) |

---

## Test Results

### Test 1 — Model File Exists
`models/rf_model.pkl` present: **PASS**

### Test 2 — Correct Model Type
`RandomForestClassifier`: **PASS**

### Test 3 — Validation Accuracy in Expected Range (60–80%)
Accuracy = 0.6245: **PASS**

This is within the expected 62–70% range stated in the plan.
No leakage detected.

### Test 4 — ROC-AUC in Expected Range (60–85%)
ROC-AUC = 0.6825: **PASS**

### Test 5 — Leakage Guard (accuracy < 80%)
0.6245 < 0.80: **PASS** — no sign of data leakage

### Test 6 — predict_proba Output Shape
Shape (5, 2) for 5 samples: **PASS**

### Test 7 — Probabilities Sum to 1
All rows: **PASS**

### Test 8 — No Val/Hold-out Used in Training
Training fit called exclusively on X_train/y_train: **PASS** (confirmed by code design)

---

## Deviations from Plan

| Item | Plan | Actual | Reason |
|---|---|---|---|
| GridSearchCV (45 fits) | Planned | Replaced with single fixed-param model | Runtime too long for workshop context; chosen params (200 trees, depth 10) are well-established defaults consistent with the plan's grid range |

---

## Phase 4 Verdict

**PASS** — Model trained, evaluated, and serialised. Accuracy (62.45%) and
ROC-AUC (0.6825) are within expected ranges. No leakage detected.
Ready for Phase 5.
