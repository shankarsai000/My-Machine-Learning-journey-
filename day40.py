# ============================================================
# DAY 40/60 — ML PIPELINES
# Built-in Scikit-Learn Pipeline vs Our Own Custom Pipeline
# Real-world scenario: Customer Churn Prediction
# ============================================================

import numpy as np
import pandas as pd

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


# ============================================================
# 1. CREATE A REALISTIC CUSTOMER CHURN DATASET
# ============================================================

# Imagine these represent customers of a subscription company.
#
# Features:
#   age
#   monthly_spend
#   tenure_months
#   support_calls
#   usage_score
#   satisfaction_score
#
# Target:
#   churn
#
# 0 = Customer stayed
# 1 = Customer churned

X, y = make_classification(
    n_samples=3000,
    n_features=6,
    n_informative=5,
    n_redundant=1,
    n_clusters_per_class=2,
    weights=[0.72, 0.28],
    class_sep=1.2,
    random_state=42
)

feature_names = [
    "age",
    "monthly_spend",
    "tenure_months",
    "support_calls",
    "usage_score",
    "satisfaction_score"
]

X = pd.DataFrame(X, columns=feature_names)
y = pd.Series(y, name="churn")


# ============================================================
# 2. MAKE THE DATA MORE REALISTIC
# ============================================================

# Convert generated values into realistic-looking ranges.

X["age"] = np.clip(
    40 + X["age"] * 10,
    18,
    80
)

X["monthly_spend"] = np.clip(
    100 + X["monthly_spend"] * 50,
    20,
    500
)

X["tenure_months"] = np.clip(
    30 + X["tenure_months"] * 15,
    1,
    120
)

X["support_calls"] = np.clip(
    5 + X["support_calls"] * 2,
    0,
    20
)

X["usage_score"] = np.clip(
    50 + X["usage_score"] * 15,
    0,
    100
)

X["satisfaction_score"] = np.clip(
    5 + X["satisfaction_score"],
    1,
    10
)


# ============================================================
# 3. ADD SOME MISSING VALUES
# ============================================================

# Real-world datasets often contain missing values.
#
# Our pipeline will handle them using median imputation.

rng = np.random.default_rng(42)

missing_mask = rng.random(X.shape) < 0.03
X = X.mask(missing_mask)


print("=" * 70)
print("CUSTOMER CHURN DATASET")
print("=" * 70)

print(f"Dataset shape: {X.shape}")
print(f"Number of missing values: {X.isna().sum().sum()}")
print(f"Churn rate: {y.mean():.2%}")


# ============================================================
# 4. TRAIN / TEST SPLIT
# ============================================================

# IMPORTANT:
# Split BEFORE fitting preprocessing.
#
# This prevents data leakage.

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTrain shape:", X_train.shape)
print("Test shape :", X_test.shape)


# ============================================================
# 5. BUILT-IN SCIKIT-LEARN PIPELINE
# ============================================================

# Scikit-Learn already provides a Pipeline implementation.
#
# Workflow:
#
# Missing values
#       ↓
# StandardScaler
#       ↓
# Logistic Regression
#
# Every step is fitted correctly using training data.

sklearn_pipeline = SklearnPipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),
    (
        "scaler",
        StandardScaler()
    ),
    (
        "model",
        LogisticRegression(
            max_iter=1000,
            random_state=42
        )
    )
])


# Train the built-in pipeline.

sklearn_pipeline.fit(
    X_train,
    y_train
)

# Make predictions.

sklearn_predictions = sklearn_pipeline.predict(
    X_test
)


# ============================================================
# 6. OUR OWN CUSTOM PIPELINE
# ============================================================

# Now we implement the SAME idea ourselves.
#
# This demonstrates that we understand what a pipeline
# actually does internally.
#
# Our pipeline will:
#
# 1. Fit each preprocessing step on training data
# 2. Transform the training data
# 3. Train the model
# 4. During prediction, transform new data using the
#    already-fitted preprocessing steps
# 5. Generate predictions


class CustomMLPipeline:
    """
    A simple custom ML pipeline.

    It sequentially applies transformers and then
    trains a final estimator.
    """

    def __init__(self, steps):
        self.steps = steps

    def fit(self, X, y):
        """
        Fit every transformation sequentially,
        then fit the final model.
        """

        X_current = X.copy()

        # All steps except the final step are transformers.

        for name, transformer in self.steps[:-1]:

            # Learn transformation parameters
            # ONLY from training data.

            transformer.fit(X_current, y)

            # Apply transformation.

            X_current = transformer.transform(X_current)

            print(f"  Fitted transformer: {name}")

        # Final step is the ML model.

        model_name, model = self.steps[-1]

        model.fit(
            X_current,
            y
        )

        self.model = model

        print(f"  Fitted model: {model_name}")

        return self

    def predict(self, X):
        """
        Transform new data using already-fitted
        transformers and generate predictions.
        """

        X_current = X.copy()

        # Apply transformations.

        for name, transformer in self.steps[:-1]:

            X_current = transformer.transform(
                X_current
            )

        # Generate predictions.

        return self.model.predict(
            X_current
        )


# ============================================================
# 7. CREATE OUR CUSTOM PIPELINE
# ============================================================

custom_pipeline = CustomMLPipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),
    (
        "scaler",
        StandardScaler()
    ),
    (
        "model",
        LogisticRegression(
            max_iter=1000,
            random_state=42
        )
    )
])


print("\n" + "=" * 70)
print("TRAINING OUR CUSTOM PIPELINE")
print("=" * 70)

custom_pipeline.fit(
    X_train,
    y_train
)

custom_predictions = custom_pipeline.predict(
    X_test
)


# ============================================================
# 8. EVALUATION FUNCTION
# ============================================================

def evaluate_model(name, y_true, y_pred):
    """
    Calculate important classification metrics.
    """

    return {
        "Pipeline": name,
        "Accuracy": accuracy_score(
            y_true,
            y_pred
        ),
        "Precision": precision_score(
            y_true,
            y_pred,
            zero_division=0
        ),
        "Recall": recall_score(
            y_true,
            y_pred,
            zero_division=0
        ),
        "F1 Score": f1_score(
            y_true,
            y_pred,
            zero_division=0
        )
    }


# ============================================================
# 9. COMPARE BOTH PIPELINES
# ============================================================

results = pd.DataFrame([
    evaluate_model(
        "Scikit-Learn Pipeline",
        y_test,
        sklearn_predictions
    ),
    evaluate_model(
        "Our Custom Pipeline",
        y_test,
        custom_predictions
    )
])

print("\n" + "=" * 70)
print("PIPELINE PERFORMANCE COMPARISON")
print("=" * 70)

print(
    results.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# 10. CHECK WHETHER BOTH PIPELINES AGREE
# ============================================================

same_predictions = np.array_equal(
    sklearn_predictions,
    custom_predictions
)

agreement_percentage = np.mean(
    sklearn_predictions == custom_predictions
) * 100

print("\n" + "=" * 70)
print("PREDICTION CONSISTENCY")
print("=" * 70)

print(
    f"Predictions identical: {same_predictions}"
)

print(
    f"Prediction agreement: {agreement_percentage:.2f}%"
)


# ============================================================
# 11. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("SCIKIT-LEARN PIPELINE REPORT")
print("=" * 70)

print(
    classification_report(
        y_test,
        sklearn_predictions,
        target_names=[
            "Stayed",
            "Churned"
        ]
    )
)


print("\n" + "=" * 70)
print("CUSTOM PIPELINE REPORT")
print("=" * 70)

print(
    classification_report(
        y_test,
        custom_predictions,
        target_names=[
            "Stayed",
            "Churned"
        ]
    )
)


# ============================================================
# 12. UNDERSTANDING WHAT HAPPENED
# ============================================================

print("\n" + "=" * 70)
print("WHAT WE BUILT")
print("=" * 70)

print("""
BUILT-IN SCIKIT-LEARN PIPELINE
--------------------------------

Raw Data
   ↓
SimpleImputer
   ↓
StandardScaler
   ↓
LogisticRegression
   ↓
Prediction


OUR CUSTOM PIPELINE
-------------------

Raw Data
   ↓
CustomMLPipeline
   ↓
SimpleImputer
   ↓
StandardScaler
   ↓
LogisticRegression
   ↓
Prediction


The important point:

Both pipelines follow the same ML workflow.

The difference is that Scikit-Learn provides a
production-ready Pipeline implementation, while we
implemented the basic pipeline logic ourselves.
""")


# ============================================================
# 13. WHY OUR CUSTOM PIPELINE IS USEFUL FOR LEARNING
# ============================================================

print("=" * 70)
print("WHAT OUR CUSTOM IMPLEMENTATION TEACHES")
print("=" * 70)

print("""
1. A pipeline is fundamentally a sequence of steps.

2. Transformers usually follow:

       fit()
          ↓
       transform()

3. The final estimator follows:

       fit()
          ↓
       predict()

4. Training data is used to LEARN preprocessing parameters.

5. Test data is only TRANSFORMED using those learned parameters.

6. This helps prevent data leakage.

7. The pipeline makes preprocessing + model behave as
   one reusable ML object.

8. In real projects, we normally prefer Scikit-Learn's
   tested Pipeline implementation instead of reinventing it.
""")


# ============================================================
# 14. IMPORTANT ENGINEERING COMPARISON
# ============================================================

comparison = pd.DataFrame({
    "Aspect": [
        "Implementation",
        "Preprocessing",
        "Data Leakage Protection",
        "Cross-Validation Support",
        "Hyperparameter Tuning",
        "Production Readiness",
        "Learning Value"
    ],
    "Scikit-Learn Pipeline": [
        "Battle-tested library implementation",
        "Built-in transformers",
        "Strong",
        "Excellent",
        "Excellent",
        "High",
        "High"
    ],
    "Our Custom Pipeline": [
        "Implemented from scratch",
        "Can use custom transformers",
        "Depends on implementation",
        "Needs additional work",
        "Needs additional work",
        "Requires more engineering",
        "Excellent for understanding"
    ]
})

print("\n" + "=" * 70)
print("ENGINEERING COMPARISON")
print("=" * 70)

print(comparison.to_string(index=False))


# ============================================================
# 15. FINAL TAKEAWAY
# ============================================================

print("\n" + "=" * 70)
print("DAY 40 — FINAL TAKEAWAY")
print("=" * 70)

print("""
The goal isn't to replace Scikit-Learn's Pipeline.

The goal is to understand what is happening underneath it.

Scikit-Learn gives us:

    Reliable infrastructure
    + tested components
    + cross-validation integration
    + hyperparameter tuning
    + production-friendly APIs

We provide:

    Workflow design
    + custom transformations
    + domain-specific logic
    + ML engineering decisions


FINAL MENTAL MODEL:

        RAW DATA
            ↓
      PREPROCESSING
            ↓
       TRANSFORMATION
            ↓
          MODEL
            ↓
        PREDICTION

A pipeline packages this entire workflow into
one reproducible process.

DAY 40 COMPLETE ✅
""")



# ============================================================
# DAY 40/60 — SCIKIT-LEARN PIPELINES
# KEY REVISION NOTES
# ============================================================


# ------------------------------------------------------------
# 1. WHAT IS A PIPELINE?
# ------------------------------------------------------------

# A Pipeline combines multiple ML steps into ONE workflow.
#
# Example:
#
# Raw Data
#    ↓
# Missing Value Handling
#    ↓
# Feature Scaling
#    ↓
# PCA
#    ↓
# ML Model
#
# Instead of manually managing every step,
# Pipeline manages the sequence for us.


# ------------------------------------------------------------
# 2. WHY DO WE USE PIPELINES?
# ------------------------------------------------------------

# Main benefits:
#
# + Cleaner code
# + Reproducible workflow
# + Consistent preprocessing
# + Reduces data leakage
# + Works well with Cross-Validation
# + Works with GridSearchCV / RandomizedSearchCV
# + Easier deployment
# + Easier maintenance


# ------------------------------------------------------------
# 3. BASIC SCIKIT-LEARN PIPELINE
# ------------------------------------------------------------

# from sklearn.pipeline import Pipeline
#
# pipeline = Pipeline([
#     ("scaler", StandardScaler()),
#     ("model", LogisticRegression())
# ])
#
# pipeline.fit(X_train, y_train)
#
# predictions = pipeline.predict(X_test)


# ------------------------------------------------------------
# 4. HOW PIPELINE WORKS DURING TRAINING
# ------------------------------------------------------------

# pipeline.fit(X_train, y_train)
#
# Internally:
#
# X_train
#    ↓
# StandardScaler.fit_transform()
#    ↓
# Transformed X_train
#    ↓
# Model.fit()
#
#
# IMPORTANT:
# Transformers learn their parameters ONLY from training data.


# ------------------------------------------------------------
# 5. HOW PIPELINE WORKS DURING PREDICTION
# ------------------------------------------------------------

# pipeline.predict(X_test)
#
# Internally:
#
# X_test
#    ↓
# StandardScaler.transform()
#    ↓
# Transformed X_test
#    ↓
# Model.predict()
#
#
# The scaler does NOT fit again on X_test.
#
# Training:
#     fit_transform()
#
# Testing:
#     transform()


# ------------------------------------------------------------
# 6. FIT vs TRANSFORM
# ------------------------------------------------------------

# fit()
# → Learns parameters from data.
#
# Example:
# StandardScaler.fit()
# → learns mean and standard deviation.
#
#
# transform()
# → Applies learned parameters to data.
#
#
# fit_transform()
# → fit() + transform()
#
#
# General rule:
#
# TRAIN DATA:
#     fit_transform()
#
# TEST DATA:
#     transform()


# ------------------------------------------------------------
# 7. DATA LEAKAGE
# ------------------------------------------------------------

# Data leakage happens when information from validation/test
# data influences the training process.
#
#
# WRONG:
#
# scaler.fit_transform(X)
# train_test_split(...)
#
# The scaler has already seen the complete dataset.
#
#
# CORRECT:
#
# train_test_split(...)
#
# scaler.fit(X_train)
# scaler.transform(X_train)
# scaler.transform(X_test)
#
#
# Pipeline helps enforce this workflow.


# ------------------------------------------------------------
# 8. PIPELINE + CROSS VALIDATION
# ------------------------------------------------------------

# Pipelines are especially important with Cross-Validation.
#
# Example:
#
# cross_val_score(
#     pipeline,
#     X,
#     y,
#     cv=5
# )
#
#
# Each fold:
#
# Training Fold
#     ↓
# Fit preprocessing
#     ↓
# Transform training data
#     ↓
# Train model
#     ↓
# Validation Fold
#     ↓
# Transform using learned parameters
#     ↓
# Evaluate
#
#
# This helps prevent preprocessing leakage between folds.


# ------------------------------------------------------------
# 9. COLUMNTRANSFORMER
# ------------------------------------------------------------

# Real-world datasets often contain different data types.
#
# Example:
#
# Age          → Numerical
# Income       → Numerical
# City         → Categorical
# Education    → Categorical
#
#
# We need different preprocessing for different columns.
#
#
# ColumnTransformer allows this.
#
#
# Numerical:
#     StandardScaler()
#
# Categorical:
#     OneHotEncoder()
#
#
# Example:
#
# preprocessor = ColumnTransformer([
#     ("num", StandardScaler(), numerical_features),
#     ("cat", OneHotEncoder(), categorical_features)
# ])


# ------------------------------------------------------------
# 10. PIPELINE + COLUMNTRANSFORMER
# ------------------------------------------------------------

# These are commonly used together.
#
#
# Raw Data
#     ↓
# ColumnTransformer
#     ├── Numerical → Scaling
#     └── Categorical → One-Hot Encoding
#     ↓
# ML Model
#
#
# Example:
#
# pipeline = Pipeline([
#     ("preprocessing", preprocessor),
#     ("model", LogisticRegression())
# ])


# ------------------------------------------------------------
# 11. ONEHOTENCODER
# ------------------------------------------------------------

# OneHotEncoder converts categorical values into numerical
# representations.
#
# Example:
#
# City:
#
# Bangalore
# Delhi
# Mumbai
#
# becomes something like:
#
# Bangalore → [1, 0, 0]
# Delhi     → [0, 1, 0]
# Mumbai    → [0, 0, 1]
#
#
# In production, use:
#
# OneHotEncoder(handle_unknown="ignore")
#
# This prevents errors when unseen categories appear.


# ------------------------------------------------------------
# 12. CUSTOM PIPELINES
# ------------------------------------------------------------

# We are NOT limited to predefined workflows.
#
# We can create our own pipeline logic.
#
#
# Concept:
#
# Custom Pipeline
#      ↓
# Transformer 1
#      ↓
# Transformer 2
#      ↓
# Model
#
#
# Example:
#
# class CustomMLPipeline:
#
#     def __init__(self, steps):
#         self.steps = steps


# ------------------------------------------------------------
# 13. CUSTOM TRANSFORMERS
# ------------------------------------------------------------

# We can create our own Scikit-Learn compatible transformer.
#
#
# from sklearn.base import BaseEstimator, TransformerMixin
#
#
# class MyTransformer(
#         BaseEstimator,
#         TransformerMixin
# ):
#
#     def fit(self, X, y=None):
#         # Learn parameters
#         return self
#
#     def transform(self, X):
#         # Apply transformation
#         return X
#
#
# Now it can be used inside Pipeline.


# ------------------------------------------------------------
# 14. IMPORTANT TRANSFORMER INTERFACE
# ------------------------------------------------------------

# Most Scikit-Learn transformers follow:
#
#     fit()
#       ↓
#     transform()
#
#
# Example:
#
# StandardScaler
# SimpleImputer
# PCA
# OneHotEncoder
#
#
# Models generally follow:
#
#     fit()
#       ↓
#     predict()
#
#
# This common interface allows components to be composed.


# ------------------------------------------------------------
# 15. CUSTOM PIPELINE vs SCIKIT-LEARN PIPELINE
# ------------------------------------------------------------

# Scikit-Learn Pipeline:
#
# + Tested implementation
# + Production-ready
# + Cross-validation integration
# + GridSearchCV integration
# + Less code
# + Better edge-case handling
#
#
# Custom Pipeline:
#
# + Excellent for learning
# + Complete control
# + Can implement domain-specific logic
# - Requires more engineering
# - Easy to introduce bugs
# - Need to implement additional functionality yourself
#
#
# Real-world recommendation:
#
# Use Scikit-Learn Pipeline when it already solves
# the problem.
#
# Build custom components when your problem requires
# custom logic.


# ------------------------------------------------------------
# 16. PIPELINE + PCA
# ------------------------------------------------------------

# Pipeline can combine concepts learned earlier.
#
#
# Raw Features
#     ↓
# StandardScaler
#     ↓
# PCA
#     ↓
# LogisticRegression
#
#
# Example:
#
# pipeline = Pipeline([
#     ("scaler", StandardScaler()),
#     ("pca", PCA(n_components=2)),
#     ("model", LogisticRegression())
# ])


# ------------------------------------------------------------
# 17. PIPELINE + GRIDSEARCHCV
# ------------------------------------------------------------

# Pipelines can be used directly with hyperparameter tuning.
#
#
# Example:
#
# pipeline = Pipeline([
#     ("scaler", StandardScaler()),
#     ("model", LogisticRegression())
# ])
#
#
# param_grid = {
#     "model__C": [0.01, 0.1, 1, 10]
# }
#
#
# Notice:
#
# model__C
#      ↑
#      └── pipeline step + parameter
#
#
# "__" (double underscore) accesses parameters
# inside pipeline steps.


# ------------------------------------------------------------
# 18. PIPELINE + RANDOMIZEDSEARCHCV
# ------------------------------------------------------------

# Same concept works with RandomizedSearchCV.
#
# Pipeline:
#     preprocessing
#          ↓
#        model
#
# RandomizedSearchCV:
#     tries different parameter combinations
#          ↓
#     Cross-validation
#          ↓
#     Best pipeline


# ------------------------------------------------------------
# 19. PIPELINE vs MANUAL PREPROCESSING
# ------------------------------------------------------------

# MANUAL:
#
# scaler.fit(X_train)
#
# X_train_scaled = scaler.transform(X_train)
# X_test_scaled = scaler.transform(X_test)
#
# model.fit(X_train_scaled, y_train)
#
# predictions = model.predict(X_test_scaled)
#
#
# PIPELINE:
#
# pipeline.fit(X_train, y_train)
#
# predictions = pipeline.predict(X_test)
#
#
# Pipeline is cleaner and reduces the chance of mistakes.


# ------------------------------------------------------------
# 20. PIPELINE DOES NOT IMPROVE THE ALGORITHM
# ------------------------------------------------------------

# Important:
#
# Pipeline ≠ better ML algorithm
#
# Pipeline = better ML WORKFLOW
#
#
# It mainly improves:
#
# - Reproducibility
# - Consistency
# - Maintainability
# - Leakage prevention
# - Experimentation


# ------------------------------------------------------------
# 21. PIPELINE IN PRODUCTION
# ------------------------------------------------------------

# Training:
#
# Raw Data
#     ↓
# Validation
#     ↓
# Preprocessing
#     ↓
# Feature Engineering
#     ↓
# Model
#
#
# Prediction:
#
# New Data
#     ↓
# Same preprocessing
#     ↓
# Same transformations
#     ↓
# Same model
#     ↓
# Prediction
#
#
# This consistency is extremely important in production.


# ------------------------------------------------------------
# 22. MODEL PIPELINE vs PRODUCTION ML PIPELINE
# ------------------------------------------------------------

# MODEL PIPELINE:
#
# Data
#   ↓
# Preprocessing
#   ↓
# Feature Engineering
#   ↓
# Model
#   ↓
# Prediction
#
#
# PRODUCTION ML PIPELINE:
#
# Data Sources
#   ↓
# Data Ingestion
#   ↓
# Data Validation
#   ↓
# Data Cleaning
#   ↓
# Feature Engineering
#   ↓
# Training
#   ↓
# Evaluation
#   ↓
# Model Registry
#   ↓
# Deployment
#   ↓
# Monitoring
#
#
# Scikit-Learn Pipeline mainly addresses the FIRST type.


# ------------------------------------------------------------
# 23. COMMON BEGINNER MISTAKES
# ------------------------------------------------------------

# Mistake 1:
# Scaling before train/test split.
#
#
# Mistake 2:
# Fitting scaler separately on test data.
#
# WRONG:
# scaler.fit_transform(X_test)
#
#
# Mistake 3:
# Applying numerical transformations to categorical columns.
#
#
# Mistake 4:
# Performing preprocessing outside the pipeline during
# cross-validation.
#
#
# Mistake 5:
# Forgetting handle_unknown="ignore" for categorical data.
#
#
# Mistake 6:
# Assuming Pipeline automatically improves accuracy.
#
#
# Mistake 7:
# Rebuilding reliable Scikit-Learn functionality without
# a real reason.


# ------------------------------------------------------------
# 24. ADVANTAGES
# ------------------------------------------------------------

# + Reproducibility
# + Cleaner architecture
# + Less repetitive code
# + Reduces preprocessing mistakes
# + Helps prevent data leakage
# + Works with cross-validation
# + Works with hyperparameter tuning
# + Easier deployment
# + Consistent inference


# ------------------------------------------------------------
# 25. LIMITATIONS
# ------------------------------------------------------------

# - Complex pipelines can become difficult to debug.
# - Custom components require careful implementation.
# - Pipeline does not guarantee a good ML model.
# - Poor preprocessing decisions can still produce
#   poor results.
# - Very large production workflows may require
#   dedicated orchestration tools.


# ------------------------------------------------------------
# 26. INTERVIEW QUESTIONS
# ------------------------------------------------------------

# Q1: What is a Scikit-Learn Pipeline?
#
# → A mechanism for chaining preprocessing and modeling
#   steps into one estimator/workflow.


# Q2: Why use Pipeline?
#
# → Cleaner code, reproducibility, consistent preprocessing,
#   leakage prevention, CV and hyperparameter tuning.


# Q3: What is ColumnTransformer?
#
# → It applies different transformations to different
#   groups of columns.


# Q4: Why put preprocessing inside Pipeline?
#
# → So preprocessing is learned only from the appropriate
#   training data, especially during cross-validation.


# Q5: What does model__C mean?
#
# → Parameter C of the pipeline step named "model".


# Q6: Pipeline vs ColumnTransformer?
#
# Pipeline:
# → Chains sequential steps.
#
# ColumnTransformer:
# → Applies different transformations to different columns.


# Q7: Can we create our own pipeline?
#
# → Yes.
#
# We can:
# - Create custom pipeline logic.
# - Create custom transformers.
# - Combine custom and Scikit-Learn components.


# Q8: Should we always build our own Pipeline?
#
# → No.
#
# Prefer tested Scikit-Learn components when they solve
# the problem. Build custom components when custom logic
# is actually required.


# ------------------------------------------------------------
# 27. DAY 40 MEMORY MODEL
# ------------------------------------------------------------

# Remember:
#
#             RAW DATA
#                 ↓
#          PREPROCESSING
#                 ↓
#        FEATURE TRANSFORMATION
#                 ↓
#              MODEL
#                 ↓
#            PREDICTION
#
#
# Pipeline packages these steps into ONE workflow.


# ------------------------------------------------------------
# 28. MOST IMPORTANT RULE
# ------------------------------------------------------------

# LEARN FROM TRAINING DATA.
# APPLY TO TEST DATA.
#
#
# fit()
# → Learn
#
# transform()
# → Apply
#
# predict()
# → Predict
#
#
# This simple idea is fundamental to leakage-safe ML.


# ------------------------------------------------------------
# 29. ONE-MINUTE REVISION
# ------------------------------------------------------------

# Pipeline:
# → Sequential ML workflow.
#
# Transformer:
# → fit() + transform()
#
# Model:
# → fit() + predict()
#
# ColumnTransformer:
# → Different preprocessing for different columns.
#
# Pipeline + CV:
# → Safer preprocessing during cross-validation.
#
# Pipeline + GridSearchCV:
# → Hyperparameter tuning of complete workflows.
#
# Custom Pipeline:
# → Possible, useful for understanding/custom logic.
#
# Production:
# → Prefer tested framework components unless custom
#   behavior is genuinely needed.


# ============================================================
# FINAL MEMORY TRICK
# ============================================================

# "PIPELINE = PREPROCESS + TRANSFORM + MODEL"
#
# And the golden rule:
#
# "FIT ON TRAIN, TRANSFORM TEST."
#
# If you remember these two ideas, you understand
# the core of Day 40.
# ============================================================