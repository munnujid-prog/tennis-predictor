# Phase 5 Implementation Log — Module 4: Model Interpretation

**Date:** 2025-10-01
**Status:** COMPLETE
**Script:** `src/04_interpret.py`
**Plan Reference:** IMPLEMENTATION_PLAN.md — Phase 5

---

## Objective

Explain which features drive predictions using Gini (built-in) and permutation
importance methods, produce comparison charts, and surface any surprising or
near-zero-importance features.

---

## Implementation Summary

### Steps Executed (per plan)

| Step | Description | Result |
|---|---|---|
| 1 | Load rf_model.pkl, feature_columns.pkl, X_val, y_val | Done |
| 2 | Extract model.feature_importances_, sort, plot horizontal bar chart | Done |
| 3 | Save Gini chart to models/feature_importance_gini.png | Done (1200x756 px) |
| 4 | Compute permutation_importance (n_repeats=10, random_state=42) | Done |
| 5 | Plot permutation importances with error bars | Done |
| 6 | Save permutation chart to models/feature_importance_permutation.png | Done (1200x756 px) |
| 7 | Build and print Gini vs. permutation comparison table | Done |
| 8 | Print commentary on dominant features and pruning candidates | Done |

---

## Feature Importance Results

### Gini (Built-in) Importances — Top 5

| Rank | Feature | Gini Importance |
|---|---|---|
| 1 | `rank_ratio` | 0.5366 |
| 2 | `rank_diff` | 0.3888 |
| 3 | `round_num` | 0.0274 |
| 4 | `best_of` | 0.0065 |
| 5 | `Series_Grand Slam` | 0.0054 |

Combined, `rank_ratio` and `rank_diff` account for **92.5%** of all Gini importance.
This is expected — ATP ranking is the single strongest predictor of match outcome.

### Permutation Importances — Top 3

| Rank | Feature | Perm Importance (mean) |
|---|---|---|
| 1 | `rank_ratio` | 0.0907 |
| 2 | `rank_diff` | 0.0085 |
| 3 | `Series_ATP500` | 0.0013 |

### Gini vs. Permutation Comparison

Both methods agree that `rank_ratio` and `rank_diff` are the top 2 features.
`round_num` ranks 3rd by Gini but drops to 13th by permutation importance —
indicating it contributes to tree splits (Gini) but does not improve overall
prediction accuracy when shuffled (permutation). This suggests `round_num` has
limited true predictive value despite being used by the trees.

### Pruning Candidates (near-zero or negative permutation importance)

The following features have near-zero or negative permutation importance and
could be dropped in a future iteration without materially harming accuracy:

`Series_International`, `Surface_Carpet`, `Series_International Gold`,
`Series_Masters Cup`, `Series_Masters`, `round_num`, `Court_Outdoor`,
`Surface_Hard`, `Series_Masters 1000`, `Surface_Clay`, `best_of`

These are retained for now as they cause no harm and provide marginal
contextual signal for the Streamlit UI.

---

## Test Results

### Test 1 — Chart Files Present and Valid

| File | Exists | Size | Result |
|---|---|---|---|
| `models/feature_importance_gini.png` | Yes | 1200x756 px | PASS |
| `models/feature_importance_permutation.png` | Yes | 1200x756 px | PASS |

### Test 2 — Rank Features in Top 3 (Gini)

Top 3: `rank_ratio`, `rank_diff`, `round_num` — rank features in top 2: **PASS**

### Test 3 — Gini Importances Sum to 1

Sum = 1.000000: **PASS**

### Test 4 — Top Permutation Feature is Rank-Based

`rank_ratio` is top permutation feature (0.0907): **PASS**

### Test 5 — Rank Feature in Top 2 Permutation

`rank_ratio` and `rank_diff` are top 2 by permutation: **PASS**

---

## Key Insight

> The model has correctly learned that **ATP ranking is the dominant predictor**
> of match outcome. `rank_ratio` alone explains 53.7% of all tree splits (Gini)
> and produces a permutation importance of 0.09 — meaning shuffling it causes
> accuracy to drop by ~9 percentage points.
>
> Surface, court type, and series contribute very small but non-zero signal,
> consistent with real tennis — surface matters somewhat (grass specialists,
> clay kings) but ranking dominates.

---

## Deviations from Plan

None. All planned steps executed as specified.

---

## Phase 5 Verdict

**PASS** — Both importance charts produced and saved. Rank features dominate
as expected. No anomalies detected. Ready for Phase 6.
