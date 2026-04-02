import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline

os.makedirs("../models", exist_ok=True)

# ---------------- LOAD ----------------
data = pd.read_csv("../data/data.csv")
print(f"Dataset shape: {data.shape}")
print(f"Class distribution:\n{data['label'].value_counts()}\n")

feature_cols = [
    "red_area", "yellow_area", "dark_area", "pink_area",
    "red_intensity", "red_std",
    "edge_ratio", "texture", "entropy",
    "saturation", "ry_ratio", "pus_necrosis_combined"
]

X = data[feature_cols].values
y = data["label"].values

# ---------------- SCALE ----------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------- CROSS-VALIDATION ----------------
print("=== 5-Fold Stratified Cross-Validation ===")

rf = RandomForestClassifier(
    n_estimators=400,
    max_depth=10,
    min_samples_split=3,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf, X_scaled, y, cv=cv, scoring="f1_macro", n_jobs=-1)

print(f"CV F1 scores: {cv_scores.round(3)}")
print(f"Mean: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}\n")

# ---------------- FINAL MODEL (full data) ----------------
rf.fit(X_scaled, y)

# Calibrate probability outputs
calibrated_rf = CalibratedClassifierCV(rf, method="isotonic", cv=3)
calibrated_rf.fit(X_scaled, y)

# ---------------- EVALUATE ON HOLD-OUT ----------------
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

rf_eval = RandomForestClassifier(
    n_estimators=400, max_depth=10, min_samples_split=3,
    min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=-1
)
rf_eval.fit(X_train, y_train)
y_pred = rf_eval.predict(X_test)

print("=== Hold-out Evaluation ===")
print(f"Accuracy: {rf_eval.score(X_test, y_test):.4f}")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Feature importances
importances = pd.Series(rf_eval.feature_importances_, index=feature_cols)
print("\nFeature Importances:")
print(importances.sort_values(ascending=False).round(4))

# ---------------- SAVE ----------------
joblib.dump(calibrated_rf, "../models/model.pkl")
joblib.dump(scaler, "../models/scaler.pkl")

# Save class order
classes = list(rf.classes_)
joblib.dump(classes, "../models/classes.pkl")

print(f"\n✅ Model + scaler saved")
print(f"Classes: {classes}")