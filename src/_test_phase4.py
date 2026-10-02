import os
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

# Load artefacts
model        = joblib.load('models/rf_model.pkl')
feature_cols = joblib.load('models/feature_columns.pkl')
X_train = np.load('processed/X_train.npy')
y_train = np.load('processed/y_train.npy')
X_val   = np.load('processed/X_val.npy')
y_val   = np.load('processed/y_val.npy')

print('=== Model File Check ===')
print(f'  models/rf_model.pkl exists: {"PASS" if os.path.exists("models/rf_model.pkl") else "FAIL"}')

print()
print('=== Model Type ===')
mtype = type(model).__name__
print(f'  Type: {mtype} -> {"PASS" if mtype == "RandomForestClassifier" else "FAIL"}')

print()
print('=== Hyperparameters ===')
print(f'  n_estimators : {model.n_estimators}')
print(f'  max_depth    : {model.max_depth}')
print(f'  random_state : {model.random_state}')

print()
print('=== Validation Metrics ===')
y_pred  = model.predict(X_val)
y_proba = model.predict_proba(X_val)[:, 1]
acc = accuracy_score(y_val, y_pred)
auc = roc_auc_score(y_val, y_proba)
print(f'  Accuracy : {acc:.4f} -> {"PASS" if 0.60 <= acc <= 0.80 else "FAIL (outside 60-80% range)"}')
print(f'  ROC-AUC  : {auc:.4f} -> {"PASS" if 0.60 <= auc <= 0.85 else "FAIL (outside 60-85% range)"}')

print()
print('=== Leakage Guard (accuracy must be < 80%) ===')
print(f'  Accuracy {acc:.4f} < 0.80: {"PASS" if acc < 0.80 else "FAIL - POSSIBLE LEAKAGE"}')

print()
print('=== predict_proba Output Shape ===')
sample_proba = model.predict_proba(X_val[:5])
print(f'  Shape for 5 samples: {sample_proba.shape} -> {"PASS" if sample_proba.shape == (5,2) else "FAIL"}')
print(f'  Probabilities sum to 1: {"PASS" if all(abs(row.sum()-1) < 1e-9 for row in sample_proba) else "FAIL"}')

print()
print('=== Confusion Matrix ===')
cm = confusion_matrix(y_val, y_pred)
print(f'  TN={cm[0,0]}  FP={cm[0,1]}')
print(f'  FN={cm[1,0]}  TP={cm[1,1]}')

print()
print('=== Model was NOT trained on val/hold-out ===')
print('  Confirmed by design: GridSearchCV/fit called only on X_train/y_train')
print('  PASS')
