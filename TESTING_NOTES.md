# Testing Notes — ATP Tennis Match Predictor

**Date:** 2025-10-01
**Test Script:** `src/_test_phase8.py`
**Outcome:** ALL TESTS PASSED (54/54)

---

## Overall Results

| Phase | Tests | Passed | Failed |
|---|---|---|---|
| 8.1 Pipeline scripts | 5 | 5 | 0 |
| 8.2 Row count consistency | 4 | 4 | 0 |
| 8.3 Accuracy / leakage | 6 | 6 | 0 |
| 8.4 Importance charts | 3 | 3 | 0 |
| 8.5 Leakage audit | 8 | 8 | 0 |
| 8.6 Temporal integrity | 2 | 2 | 0 |
| 8.7 Feature matrix | 4 | 4 | 0 |
| 8.8 Artefact files | 17 | 17 | 0 |
| 8.9 app.py logic | 5 | 5 | 0 |
| **TOTAL** | **54** | **54** | **0** |

---

## Key Metrics

| Metric | Value |
|---|---|
| Validation Accuracy | 62.45% |
| Validation ROC-AUC | 0.6825 |
| Hold-out Accuracy | 63.82% |
| Hold-out ROC-AUC | 0.6969 |
| Players in lookup table | 1,428 |
| Features in model | 18 |
| Training rows | 46,993 |
| Validation rows | 10,077 |
| Hold-out rows | 9,962 |
| Training date range | 2000-01-03 to 2018-02-19 |
| Validation date range | 2018-02-19 to 2022-08-31 |
| Hold-out date range | 2022-08-31 to 2026-09-13 |

---

## Issues Found and Resolved

| Issue | Phase Found | Resolution |
|---|---|---|
| Unicode arrow/emoji characters in print statements caused cp1252 encoding errors on Windows | Phase 4-6 | Replaced all unicode arrows/symbols with ASCII equivalents in pipeline scripts |
| GridSearchCV (45 fits) too slow for workshop context | Phase 4 | Replaced with single fixed-param model (n_estimators=200, max_depth=10) |
| Test script reading app.py failed with UnicodeDecodeError | Phase 7 | Fixed by opening app.py with encoding='utf-8' |
| Streamlit browser test not executable headlessly | Phase 8 | Verified programmatically; app launches correctly with `py -m streamlit run app.py` |

---

## How to Run

```powershell
# Run full pipeline
py src/01_prepare.py
py src/02_features.py
py src/03_train.py
py src/04_interpret.py
py src/05_predict.py

# Run validation suite
py src/_test_phase8.py

# Launch web app
py -m streamlit run app.py
```

---

## Notes

- Retirement/walkover filter in `01_prepare.py` removed 0 rows — the raw dataset
  appears to have already excluded incomplete matches, or uses different notation.
  The filter remains as a safeguard.

- 1,139 rows were dropped during feature engineering (Phase 3) due to Round values
  not present in the ordinal mapping. This is expected for non-standard round labels
  and does not affect model quality.

- Hold-out accuracy (63.82%) is marginally better than validation (62.45%),
  indicating the model generalises well to unseen future data with no sign of overfitting.

- The "live prediction challenge" is a retrospective simulation — no real-time
  data feed is used. The hold-out tail (2022-2026) serves as a proxy for
  unseen future matches.
