"""
Module 6 - Deployment model (leakage-free)
==========================================

Why this script exists
----------------------
The Module 3 models reach ~99.8% accuracy because the training data contains
``Salary Package``, a column that is only known AFTER a student is placed.
A model that uses it looks perfect but cannot actually forecast anything.

This script trains an honest replacement:

    * features  : all 32 columns EXCEPT StudentID, Salary Package, IsAnomaly
    * target    : PlacementStatus (1 = placed, 0 = not placed)
    * model     : Random Forest inside a full sklearn Pipeline
    * split     : 80 / 20 stratified, random_state = 42 (same convention as M3)
    * report    : accuracy, precision, recall, F1, ROC-AUC + 5-fold CV

Everything it writes goes to a NEW folder and no existing file is modified.
The saved pipeline is the one the website serves on the Prediction pages.

Run:  python src/Deployment_Model_LeakageFree_M6.py
"""

import json
import os

import joblib
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from pandas.api.types import is_numeric_dtype

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET = os.path.join(BASE_DIR, "dataset", "placement_predict_50K_Raw.csv")
OUT = os.path.join(BASE_DIR, "outputs", "Deployment_Model_M6_Outputs")

for sub in ("model", "metrics", "charts", "feature_importance", "confusion_matrix", "predictions"):
    os.makedirs(os.path.join(OUT, sub), exist_ok=True)

RANDOM_STATE = 42

# Columns that must never be used as features:
#   StudentID       - just a row number
#   Salary Package  - only known AFTER placement  (the leak)
#   IsAnomaly       - a post-hoc quality flag
DROP_COLUMNS = ["StudentID", "Salary Package", "IsAnomaly"]
TARGET = "PlacementStatus"

print("=" * 68)
print("Module 6 - leakage-free deployment model")
print("=" * 68)

# ----------------------------------------------------------------------
# 1. Load
# ----------------------------------------------------------------------

df = pd.read_csv(DATASET)
print(f"dataset      : {df.shape[0]:,} rows x {df.shape[1]} columns")

features = [c for c in df.columns if c not in DROP_COLUMNS + [TARGET]]
X = df[features].copy()
y = df[TARGET].astype(int)

print(f"dropped      : {', '.join(DROP_COLUMNS)}")
print(f"features used: {len(features)}")

numeric_cols = [c for c in features if is_numeric_dtype(X[c])]
categorical_cols = [c for c in features if not is_numeric_dtype(X[c])]
print(f"numeric      : {len(numeric_cols)}   categorical: {len(categorical_cols)}")

# ----------------------------------------------------------------------
# 2. Pipeline
# ----------------------------------------------------------------------

numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipe, numeric_cols),
    ("cat", categorical_pipe, categorical_cols),
])

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_leaf=20,
    min_samples_split=40,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", model)])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)
print(f"train / test : {len(X_train):,} / {len(X_test):,}")

pipeline.fit(X_train, y_train)

# ----------------------------------------------------------------------
# 3. Metrics
# ----------------------------------------------------------------------

y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

cv = cross_val_score(
    pipeline, X_train, y_train, cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE),
    scoring="f1",
)

metrics = pd.DataFrame([{
    "Model": "Random Forest (leakage-free)",
    "Accuracy": round(accuracy, 4),
    "Precision": round(precision, 4),
    "Recall": round(recall, 4),
    "F1": round(f1, 4),
    "ROC_AUC": round(roc_auc, 4),
    "CV_F1_mean": round(float(cv.mean()), 4),
    "CV_F1_std": round(float(cv.std()), 4),
    "Features": len(features),
    "Train_rows": len(X_train),
    "Test_rows": len(X_test),
}])
metrics.T.to_csv(os.path.join(OUT, "metrics", "leakage_free_metrics.csv"), header=False)

report = classification_report(y_test, y_pred, target_names=["Not placed", "Placed"], output_dict=True)
pd.DataFrame(report).T.round(4).to_csv(os.path.join(OUT, "metrics", "classification_report.csv"))

params = pd.DataFrame([{
    "n_estimators": 300, "max_depth": 10, "min_samples_leaf": 20,
    "min_samples_split": 40, "random_state": RANDOM_STATE,
    "dropped": ", ".join(DROP_COLUMNS),
}])
params.T.to_csv(os.path.join(OUT, "metrics", "model_parameters.csv"), header=False)

print("-" * 68)
print(f"accuracy     : {accuracy:.4f}")
print(f"precision    : {precision:.4f}")
print(f"recall       : {recall:.4f}")
print(f"F1           : {f1:.4f}")
print(f"ROC-AUC      : {roc_auc:.4f}")
print(f"5-fold CV F1 : {cv.mean():.4f} (+/- {cv.std():.4f})")

# ----------------------------------------------------------------------
# 4. Confusion matrix
# ----------------------------------------------------------------------

cm = confusion_matrix(y_test, y_pred)
pd.DataFrame(cm, index=["Actual not placed", "Actual placed"],
             columns=["Predicted not placed", "Predicted placed"]
             ).to_csv(os.path.join(OUT, "confusion_matrix", "confusion_matrix.csv"))

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", cbar=False,
            xticklabels=["Not placed", "Placed"], yticklabels=["Not placed", "Placed"], ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title(f"Leakage-free Random Forest\naccuracy {accuracy:.2%}")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "confusion_matrix", "confusion_matrix.png"), dpi=130)
plt.close(fig)

# ----------------------------------------------------------------------
# 5. Feature importance
# ----------------------------------------------------------------------

feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
importances = pipeline.named_steps["classifier"].feature_importances_
imp = (pd.DataFrame({"feature": feature_names, "importance": importances})
       .sort_values("importance", ascending=False))
imp["feature"] = (imp["feature"]
                  .str.replace("num__", "", regex=False)
                  .str.replace("cat__", "", regex=False))
imp.to_csv(os.path.join(OUT, "feature_importance", "feature_importance.csv"), index=False)

top = imp.head(15).iloc[::-1]
fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(top["feature"], top["importance"], color="#2E86D6")
ax.set_xlabel("Importance")
ax.set_title("What actually drives the prediction (leakage removed)")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "feature_importance", "feature_importance.png"), dpi=130)
plt.close(fig)

# ----------------------------------------------------------------------
# 6. ROC curve + probability spread
# ----------------------------------------------------------------------

fpr, tpr, _ = roc_curve(y_test, y_proba)
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr, tpr, color="#1769AA", lw=2, label=f"AUC = {roc_auc:.3f}")
ax.plot([0, 1], [0, 1], "--", color="#7B93AE", lw=1)
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
ax.set_title("ROC curve - leakage-free model")
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "charts", "roc_curve.png"), dpi=130)
plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.hist(y_proba[y_test == 0], bins=30, alpha=.75, color="#EF4444", label="Not placed")
ax.hist(y_proba[y_test == 1], bins=30, alpha=.75, color="#1769AA", label="Placed")
ax.set_xlabel("Predicted probability of placement")
ax.set_ylabel("Students")
ax.set_title("The model is confident but not perfect")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT, "charts", "probability_distribution.png"), dpi=130)
plt.close(fig)

# ----------------------------------------------------------------------
# 7. Save predictions + model + metadata
# ----------------------------------------------------------------------

pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred,
    "Placed_probability": np.round(y_proba, 4),
    "Correct": (y_pred == y_test.values).astype(int),
}).to_csv(os.path.join(OUT, "predictions", "test_predictions.csv"), index=False)

joblib.dump(pipeline, os.path.join(OUT, "model", "placement_clean_model.pkl"))

with open(os.path.join(OUT, "metrics", "model_summary.json"), "w", encoding="utf-8") as fh:
    json.dump({
        "model": "RandomForestClassifier (leakage-free)",
        "target": TARGET,
        "dropped_columns": DROP_COLUMNS,
        "n_features": len(features),
        "numeric_features": numeric_cols,
        "categorical_features": categorical_cols,
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4),
        "cv_f1_mean": round(float(cv.mean()), 4),
    }, fh, indent=2)

print("-" * 68)
print(f"saved to     : {OUT}")
print("done")
