import numpy as np
import pandas as pd

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE


# ============================================================
# 1. CREATE AN IMBALANCED DATASET
# ============================================================

X, y = make_classification(
    n_samples=5000,
    n_features=10,
    n_informative=6,
    n_redundant=2,
    weights=[0.95, 0.05],
    class_sep=1.2,
    random_state=42
)

print("Class distribution:")
print(pd.Series(y).value_counts())

print("\nClass percentage:")
print(pd.Series(y).value_counts(normalize=True))


# ============================================================
# 2. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)


# ============================================================
# 3. BASELINE MODEL
# ============================================================

baseline_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000))
])


# ============================================================
# 4. CLASS-WEIGHTED MODEL
# ============================================================

weighted_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    (
        "model",
        LogisticRegression(
            class_weight="balanced",
            max_iter=1000
        )
    )
])


# ============================================================
# 5. SMOTE MODEL
# ============================================================

smote_pipeline = ImbPipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("smote", SMOTE(random_state=42)),
    (
        "model",
        LogisticRegression(max_iter=1000)
    )
])


# ============================================================
# 6. TRAIN MODELS
# ============================================================

models = {
    "Baseline": baseline_pipeline,
    "Class Weighted": weighted_pipeline,
    "SMOTE": smote_pipeline
}

results = []

for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    report = classification_report(
        y_test,
        predictions,
        output_dict=True,
        zero_division=0
    )

    results.append({
        "Model": name,
        "Precision": report["1"]["precision"],
        "Recall": report["1"]["recall"],
        "F1": report["1"]["f1-score"],
        "ROC-AUC": roc_auc_score(
            y_test,
            probabilities
        )
    })


# ============================================================
# 7. COMPARE
# ============================================================

results_df = pd.DataFrame(results)

print("\nMODEL COMPARISON")
print("=" * 60)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# 8. CONFUSION MATRIX
# ============================================================

for name, model in models.items():

    predictions = model.predict(X_test)

    print(f"\n{name}")
    print("-" * 40)

    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )


# ============================================================
# DAY 41 — PHASE 2 REVIEW
# ============================================================

# BIAS
# High bias → model too simple → underfitting
#
# VARIANCE
# High variance → model too sensitive → overfitting


# METRICS
# Accuracy  = overall correctness
# Precision  = TP / (TP + FP)
# Recall     = TP / (TP + FN)
# F1         = harmonic mean of precision and recall
# ROC-AUC    = ranking/discrimination ability across thresholds


# PIPELINE
# Preprocessing + model = one reusable workflow
#
# Golden rule:
# FIT on training data
# TRANSFORM validation/test data


# FEATURE ENGINEERING
# Polynomial features → x², x³, ...
# Interaction features → x1 * x2
# Binning → continuous values → ranges
# Target encoding → categorical → target statistics
#
# Target encoding must be leakage-safe.


# IMBALANCED DATA
# Stratification → preserve class proportions
# Class weights → penalize minority mistakes more
# SMOTE → generate synthetic minority samples
#
# SMOTE should be applied to TRAINING data,
# preferably inside a CV-aware pipeline.


# INTERPRETABILITY
# SHAP → feature contribution to predictions
# PDP  → effect of a feature on model predictions


# CORE MENTAL MODEL
#
# DATA
#   ↓
# SPLIT
#   ↓
# FEATURE ENGINEERING
#   ↓
# PREPROCESSING
#   ↓
# IMBALANCE HANDLING
#   ↓
# MODEL
#   ↓
# CROSS VALIDATION
#   ↓
# HYPERPARAMETER TUNING
#   ↓
# EVALUATION
#   ↓
# INTERPRETATION

'''
KEY TAKEAWAYS
Bias and variance describe different failure modes.
Accuracy alone can be dangerous on imbalanced datasets.
Feature engineering can be as important as model selection.
Polynomial and interaction features allow models to represent richer relationships.
Target encoding must be carefully designed to prevent leakage.
Stratification preserves class proportions during splitting.
Class weights and SMOTE are two different approaches to imbalance.
SMOTE belongs on the training side, not the test set.
Pipelines help make preprocessing and validation safer and reproducible.
SHAP and PDP move us from “what did the model predict?” toward “why did it predict that?”
'''