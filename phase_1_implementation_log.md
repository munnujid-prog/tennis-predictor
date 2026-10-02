# Phase 1 Implementation Log — Project Setup

**Date:** 2025-10-01
**Status:** COMPLETE
**Plan Reference:** IMPLEMENTATION_PLAN.md — Phase 1

---

## Objective

Establish the project skeleton so every subsequent phase has a consistent,
reproducible environment. Confirm the dataset is present and readable.

---

## Actions Taken

### 1. Directory Structure Created

| Directory | Purpose |
|---|---|
| `data/` | Raw dataset (read-only) |
| `processed/` | Cleaned train/val/holdout CSV splits |
| `models/` | Trained model and encoder artefacts |
| `src/` | Pipeline scripts (Modules 1-5) |
| `.streamlit/` | Streamlit config (gatherUsageStats disabled) |

### 2. Dataset Placed

- `atp_tennis.csv` copied from workspace root to `data/atp_tennis.csv`
- Original file retained at root for reference

### 3. requirements.txt Created

```
pandas>=2.0
scikit-learn>=1.4
joblib>=1.3
matplotlib>=3.8
seaborn>=0.13
streamlit>=1.35
```

### 4. README.md Created

Covers: project description, folder structure, setup instructions (venv),
run order for all pipeline scripts, feature set documentation, and notes
on expected accuracy and the retrospective simulation caveat.

### 5. Streamlit Config Created

`.streamlit/config.toml` — disables usage stats prompt on first run:
```toml
[browser]
gatherUsageStats = false
```

---

## Test Results

### Artefact Existence Check

| Item | Result |
|---|---|
| `data/atp_tennis.csv` | PASS |
| `requirements.txt` | PASS |
| `README.md` | PASS |
| `src/` directory | PASS |
| `processed/` directory | PASS |
| `models/` directory | PASS |

### Dataset Sanity Check

| Property | Value |
|---|---|
| Rows | 68,760 |
| Columns | 17 |
| Column names | Tournament, Date, Series, Court, Surface, Round, Best of, Player_1, Player_2, Winner, Rank_1, Rank_2, Pts_1, Pts_2, Odd_1, Odd_2, Score |
| Load errors | None |

### Installed Package Versions

| Package | Version |
|---|---|
| pandas | 3.0.6 |
| scikit-learn | 1.9.1 |
| joblib | 1.6.0 |
| matplotlib | 3.11.2 |
| seaborn | 0.13.2 |
| streamlit | 1.64.0 |

All versions satisfy the `requirements.txt` minimum constraints.

---

## Deviations from Plan

| Item | Plan | Actual | Reason |
|---|---|---|---|
| `.streamlit/config.toml` | Not in plan | Added | Required to suppress interactive email prompt on first Streamlit run in headless environments |
| `atp_tennis.csv` at root | Plan expected only `data/` copy | Both locations exist | Original left at root; no impact on pipeline (all scripts reference `data/atp_tennis.csv`) |

---

## Phase 1 Verdict

**PASS** — All required artefacts created and verified. Dataset loads cleanly.
Environment is ready for Phase 2.
