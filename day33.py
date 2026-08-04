# =============================================================================
# DAY 33 : CROSS VALIDATION
# PART 4.1 : TRAIN-TEST SPLIT VS K-FOLD
# =============================================================================

"""
Topics Covered

1. Load Dataset
2. Train-Test Split
3. Logistic Regression Baseline
4. Train-Test Accuracy
5. K-Fold Cross Validation
6. Fold Scores
7. Mean Accuracy
8. Standard Deviation
9. Interpretation
10. Top ML Engineer Insight

"""

# =============================================================================
# IMPORT LIBRARIES
# =============================================================================

import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import (

    train_test_split,

    cross_val_score

)

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

# =============================================================================
# 1. LOAD DATASET
# =============================================================================

print("="*70)
print("LOADING DATASET")
print("="*70)

data = load_breast_cancer()

X = pd.DataFrame(
    data.data,
    columns=data.feature_names
)

y = data.target

print("\nDataset Shape :", X.shape)

print("Target Classes :", data.target_names)

# =============================================================================
# 2. TRAIN TEST SPLIT
# =============================================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42,

    stratify=y

)

# =============================================================================
# 3. FEATURE SCALING
# =============================================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

# =============================================================================
# 4. TRAIN MODEL
# =============================================================================

model = LogisticRegression(

    max_iter=1000,

    random_state=42

)

model.fit(

    X_train,

    y_train

)

pred = model.predict(

    X_test

)

accuracy = accuracy_score(

    y_test,

    pred

)

print("\n")
print("="*70)
print("TRAIN-TEST SPLIT RESULT")
print("="*70)

print("Accuracy :", round(accuracy,4))

# =============================================================================
# 5. K-FOLD CROSS VALIDATION
# =============================================================================

print("\n")
print("="*70)
print("5-FOLD CROSS VALIDATION")
print("="*70)

pipeline = Pipeline([

    ("scaler", StandardScaler()),

    ("model", LogisticRegression(

        max_iter=1000,

        random_state=42

    ))

])

scores = cross_val_score(

    pipeline,

    X,

    y,

    cv=5,

    scoring="accuracy"

)

print("Fold Scores")

print(np.round(scores,4))

print("\nMean Accuracy")

print(round(scores.mean(),4))

print("\nStandard Deviation")

print(round(scores.std(),4))

# =============================================================================
# 6. INTERPRETATION
# =============================================================================

print("\n")
print("="*70)
print("INTERPRETATION")
print("="*70)

print("""

Train-Test Split

↓

One Accuracy Score

------------------------------------------------

Cross Validation

↓

Multiple Accuracy Scores

↓

Average Performance

↓

More Reliable Estimate

""")

# =============================================================================
# 7. TOP ML ENGINEER INSIGHT
# =============================================================================

print("\n")
print("="*70)
print("TOP ML ENGINEER INSIGHT")
print("="*70)

print("""

Never report

only

one accuracy.

Always report

Mean Accuracy

+

Standard Deviation

using Cross Validation.

A stable model is usually
more valuable than
a lucky high score.

""")

# =============================================================================
# DAY 33 : CROSS VALIDATION
# PART 4.2 : KFOLD vs STRATIFIED KFOLD
# =============================================================================

"""
Topics Covered

1. KFold
2. StratifiedKFold
3. Cross Validation Scores
4. Mean Accuracy
5. Standard Deviation
6. Comparison Table
7. Accuracy Visualization
8. Why StratifiedKFold?
9. Industry Recommendation
10. Top ML Engineer Insight

"""

# =============================================================================
# IMPORT LIBRARIES
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import (

    KFold,

    StratifiedKFold,

    cross_val_score

)

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

# =============================================================================
# 1. LOAD DATASET
# =============================================================================

print("="*70)
print("LOADING DATASET")
print("="*70)

data = load_breast_cancer()

X = data.data

y = data.target

print("Dataset Shape :", X.shape)

print("Class Distribution")

unique, counts = np.unique(y, return_counts=True)

for cls, count in zip(unique, counts):

    print(f"Class {cls} : {count}")

# =============================================================================
# 2. PIPELINE
# =============================================================================

pipeline = Pipeline([

    ("scaler", StandardScaler()),

    ("model", LogisticRegression(

        max_iter=1000,

        random_state=42

    ))

])

# =============================================================================
# 3. KFOLD
# =============================================================================

print("\n")
print("="*70)
print("KFOLD CROSS VALIDATION")
print("="*70)

kfold = KFold(

    n_splits=5,

    shuffle=True,

    random_state=42

)

k_scores = cross_val_score(

    pipeline,

    X,

    y,

    cv=kfold,

    scoring="accuracy"

)

print("Fold Scores")

print(np.round(k_scores,4))

print("\nMean Accuracy")

print(round(k_scores.mean(),4))

print("\nStandard Deviation")

print(round(k_scores.std(),4))

# =============================================================================
# 4. STRATIFIED KFOLD
# =============================================================================

print("\n")
print("="*70)
print("STRATIFIED KFOLD")
print("="*70)

stratified = StratifiedKFold(

    n_splits=5,

    shuffle=True,

    random_state=42

)

s_scores = cross_val_score(

    pipeline,

    X,

    y,

    cv=stratified,

    scoring="accuracy"

)

print("Fold Scores")

print(np.round(s_scores,4))

print("\nMean Accuracy")

print(round(s_scores.mean(),4))

print("\nStandard Deviation")

print(round(s_scores.std(),4))

# =============================================================================
# 5. COMPARISON TABLE
# =============================================================================

comparison = pd.DataFrame({

    "Validation Method":[

        "KFold",

        "StratifiedKFold"

    ],

    "Mean Accuracy":[

        round(k_scores.mean(),4),

        round(s_scores.mean(),4)

    ],

    "Std Deviation":[

        round(k_scores.std(),4),

        round(s_scores.std(),4)

    ]

})

print("\n")
print("="*70)
print("COMPARISON")
print("="*70)

print(comparison)

# =============================================================================
# 6. VISUALIZATION
# =============================================================================

plt.figure(figsize=(8,5))

plt.plot(

    range(1,6),

    k_scores,

    marker="o",

    label="KFold"

)

plt.plot(

    range(1,6),

    s_scores,

    marker="s",

    label="StratifiedKFold"

)

plt.xticks(range(1,6))

plt.xlabel("Fold Number")

plt.ylabel("Accuracy")

plt.title("KFold vs StratifiedKFold")

plt.legend()

plt.grid(True)

plt.show()

# =============================================================================
# 7. INTERPRETATION
# =============================================================================

print("\n")
print("="*70)
print("INTERPRETATION")
print("="*70)

print("""

KFold

↓

Randomly splits data.

May create folds with
different class distributions.

Suitable for

✔ Regression

✔ Balanced datasets

------------------------------------------------

StratifiedKFold

↓

Maintains class proportions
in every fold.

Suitable for

✔ Classification

✔ Imbalanced datasets

✔ Production ML

""")

# =============================================================================
# 8. WHEN TO USE WHAT?
# =============================================================================

print("\n")
print("="*70)
print("WHEN TO USE WHICH?")
print("="*70)

print("""

Use KFold

✔ House Price Prediction

✔ Sales Forecasting

✔ Regression Problems

------------------------------------------------

Use StratifiedKFold

✔ Cancer Detection

✔ Spam Detection

✔ Fraud Detection

✔ Disease Prediction

✔ Customer Churn

Any Classification Problem

""")

# =============================================================================
# 9. INDUSTRY INSIGHT
# =============================================================================

print("\n")
print("="*70)
print("INDUSTRY INSIGHT")
print("="*70)

print("""

Most production
classification pipelines use

StratifiedKFold

because every fold
represents the real dataset.

This leads to

✔ Fair Evaluation

✔ Stable Metrics

✔ Better Generalization Estimates

""")

# =============================================================================
# 10. TOP ML ENGINEER INSIGHT
# =============================================================================

print("\n")
print("="*70)
print("TOP ML ENGINEER INSIGHT")
print("="*70)

print("""

Don't choose a validation
strategy randomly.

Choose it based on
the nature of your data.

Regression

↓

KFold

Classification

↓

StratifiedKFold

Time Series

↓

TimeSeriesSplit

Using the wrong validation
strategy can produce
misleading evaluation results,
even if your model is good.

""")

# =============================================================================
# DAY 33 : CROSS VALIDATION
# PART 4.3 : MODEL COMPARISON USING CROSS_VALIDATE()
# =============================================================================

"""
Topics Covered

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. cross_validate()
5. Multiple Evaluation Metrics
6. Mean Accuracy
7. Mean Precision
8. Mean Recall
9. Mean F1 Score
10. Mean ROC-AUC
11. Fit Time
12. Score Time
13. Model Benchmarking
14. Best Model Selection
15. Top ML Engineer Insight

"""

# =============================================================================
# IMPORT LIBRARIES
# =============================================================================

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import (

    StratifiedKFold,

    cross_validate

)

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

# =============================================================================
# 1. LOAD DATASET
# =============================================================================

print("=" * 70)
print("LOADING DATASET")
print("=" * 70)

data = load_breast_cancer()

X = data.data

y = data.target

print("Dataset Shape :", X.shape)

# =============================================================================
# 2. CROSS VALIDATION SETTINGS
# =============================================================================

cv = StratifiedKFold(

    n_splits=5,

    shuffle=True,

    random_state=42

)

scoring = {

    "accuracy": "accuracy",

    "precision": "precision",

    "recall": "recall",

    "f1": "f1",

    "roc_auc": "roc_auc"

}

# =============================================================================
# 3. DEFINE MODELS
# =============================================================================

models = {

    "Logistic Regression":

    Pipeline([

        ("scaler", StandardScaler()),

        ("model",

        LogisticRegression(

            max_iter=1000,

            random_state=42

        ))

    ]),

    "Decision Tree":

    DecisionTreeClassifier(

        random_state=42

    ),

    "Random Forest":

    RandomForestClassifier(

        n_estimators=100,

        random_state=42

    )

}

# =============================================================================
# 4. EVALUATE MODELS
# =============================================================================

results = []

for name, model in models.items():

    scores = cross_validate(

        model,

        X,

        y,

        cv=cv,

        scoring=scoring,

        return_train_score=False

    )

    results.append({

        "Model": name,

        "Accuracy":

        scores["test_accuracy"].mean(),

        "Precision":

        scores["test_precision"].mean(),

        "Recall":

        scores["test_recall"].mean(),

        "F1 Score":

        scores["test_f1"].mean(),

        "ROC-AUC":

        scores["test_roc_auc"].mean(),

        "Fit Time":

        scores["fit_time"].mean(),

        "Score Time":

        scores["score_time"].mean(),

        "Std Accuracy":

        scores["test_accuracy"].std()

    })

# =============================================================================
# 5. MODEL COMPARISON TABLE
# =============================================================================

comparison = pd.DataFrame(results)

comparison = comparison.round(4)

print("\n")
print("=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(comparison)

# =============================================================================
# 6. BEST MODEL
# =============================================================================

best = comparison.loc[

    comparison["Accuracy"].idxmax()

]

print("\n")
print("=" * 70)
print("BEST MODEL")
print("=" * 70)

print("Model :", best["Model"])

print("Accuracy :", best["Accuracy"])

print("ROC-AUC :", best["ROC-AUC"])

# =============================================================================
# 7. ACCURACY COMPARISON
# =============================================================================

plt.figure(figsize=(8,5))

plt.bar(

    comparison["Model"],

    comparison["Accuracy"]

)

plt.title("Mean Accuracy Comparison")

plt.ylabel("Accuracy")

plt.grid(True)

plt.show()

# =============================================================================
# 8. ROC-AUC COMPARISON
# =============================================================================

plt.figure(figsize=(8,5))

plt.bar(

    comparison["Model"],

    comparison["ROC-AUC"]

)

plt.title("Mean ROC-AUC Comparison")

plt.ylabel("ROC-AUC")

plt.grid(True)

plt.show()

# =============================================================================
# 9. FIT TIME COMPARISON
# =============================================================================

plt.figure(figsize=(8,5))

plt.bar(

    comparison["Model"],

    comparison["Fit Time"]

)

plt.title("Average Training Time")

plt.ylabel("Seconds")

plt.grid(True)

plt.show()

# =============================================================================
# 10. SCORE TIME COMPARISON
# =============================================================================

plt.figure(figsize=(8,5))

plt.bar(

    comparison["Model"],

    comparison["Score Time"]

)

plt.title("Average Prediction Time")

plt.ylabel("Seconds")

plt.grid(True)

plt.show()

# =============================================================================
# 11. MODEL INTERPRETATION
# =============================================================================

print("\n")
print("=" * 70)
print("MODEL INTERPRETATION")
print("=" * 70)

print("""

Accuracy

↓

Overall Correct Predictions

--------------------------------------------

Precision

↓

False Alarm Control

--------------------------------------------

Recall

↓

Finding Positive Cases

--------------------------------------------

F1 Score

↓

Balance of Precision & Recall

--------------------------------------------

ROC-AUC

↓

Overall Class Separation

--------------------------------------------

Fit Time

↓

Training Speed

--------------------------------------------

Score Time

↓

Prediction Speed

""")

# =============================================================================
# 12. MODEL BENCHMARK
# =============================================================================

print("\n")
print("=" * 70)
print("MODEL BENCHMARK")
print("=" * 70)

print("""

Logistic Regression

↓

Fast

Simple

Interpretable

--------------------------------------------

Decision Tree

↓

Easy to Explain

Can Overfit

--------------------------------------------

Random Forest

↓

Better Generalization

More Robust

Higher Training Cost

""")

# =============================================================================
# 13. INDUSTRY RECOMMENDATION
# =============================================================================

print("\n")
print("=" * 70)
print("INDUSTRY RECOMMENDATION")
print("=" * 70)

print("""

Choose models based on

✔ Accuracy

✔ Stability

✔ ROC-AUC

✔ Training Time

✔ Prediction Time

✔ Business Requirements

Never choose a model
using only Accuracy.

""")

# =============================================================================
# 14. TOP ML ENGINEER INSIGHT
# =============================================================================

print("\n")
print("=" * 70)
print("TOP ML ENGINEER INSIGHT")
print("=" * 70)

print("""

Professional ML engineers
don't compare models using

ONE

metric.

They compare

✔ Accuracy

✔ Precision

✔ Recall

✔ F1 Score

✔ ROC-AUC

✔ Training Time

✔ Prediction Time

✔ Standard Deviation

The best model is not always
the one with the highest accuracy.

It is the one that provides the
best balance of performance,
stability, speed, and business value.

""")

# =============================================================================
# DAY 33 : CROSS VALIDATION
# PART 4.4 : FINAL NOTES, INTERVIEW PREP & GITHUB SUMMARY
# =============================================================================

# =============================================================================
# 15. PRACTICE EXERCISES
# =============================================================================

print("\n")
print("=" * 70)
print("PRACTICE EXERCISES")
print("=" * 70)

print("""

Beginner

1. Perform 5-Fold Cross Validation
   using Logistic Regression.

------------------------------------------------------------

2. Calculate

Mean Accuracy

Standard Deviation

------------------------------------------------------------

Intermediate

3. Compare

Train-Test Split

vs

Cross Validation

------------------------------------------------------------

4. Compare

KFold

vs

StratifiedKFold

------------------------------------------------------------

Advanced

5. Compare

Logistic Regression

Decision Tree

Random Forest

using

Accuracy

Precision

Recall

F1 Score

ROC-AUC

------------------------------------------------------------

6. Perform

10-Fold Cross Validation

Compare with

5-Fold Cross Validation

Observe

Mean Accuracy

Standard Deviation

Training Time

""")

# =============================================================================
# 16. MINI CHALLENGE
# =============================================================================

print("\n")
print("=" * 70)
print("MINI CHALLENGE")
print("=" * 70)

print("""

You are building
a Disease Detection Model.

Tasks

✔ Train Logistic Regression

✔ Evaluate using Train-Test Split

✔ Evaluate using 5-Fold Cross Validation

✔ Evaluate using StratifiedKFold

✔ Compare Mean Accuracy

✔ Compare Standard Deviation

✔ Decide

Which evaluation
is more trustworthy?

Explain WHY.

""")

# =============================================================================
# 17. COMMON INTERVIEW QUESTIONS
# =============================================================================

print("\n")
print("=" * 70)
print("COMMON INTERVIEW QUESTIONS")
print("=" * 70)

questions = [

"What is Cross Validation?",

"Why is Train-Test Split unreliable?",

"What is Hold-Out Validation?",

"What is K-Fold Cross Validation?",

"What is Stratified K-Fold?",

"Difference between KFold and StratifiedKFold?",

"What is Leave-One-Out Cross Validation (LOOCV)?",

"When should LOOCV be used?",

"What is Repeated K-Fold?",

"What is ShuffleSplit?",

"What is TimeSeriesSplit?",

"Why can't we use KFold for time-series data?",

"What is cross_val_score()?",

"What is cross_validate()?",

"Difference between cross_val_score() and cross_validate()?",

"Why report Mean Accuracy?",

"Why report Standard Deviation?",

"Why is StratifiedKFold preferred for classification?",

"Does Cross Validation improve model accuracy?",

"Does Cross Validation replace the final test set?",

"How many folds should we choose?",

"What metrics should be averaged?",

"What is Data Leakage in Cross Validation?",

"How is Cross Validation used in Hyperparameter Tuning?"

]

for q in questions:

    print("✔", q)

# =============================================================================
# 18. SENIOR ML ENGINEER NOTES
# =============================================================================

print("\n")
print("=" * 70)
print("SENIOR ML ENGINEER NOTES")
print("=" * 70)

print("""

Cross Validation

↓

Evaluation Technique

NOT

a Machine Learning Algorithm.

--------------------------------------------------

Train-Test Split

↓

Quick Baseline

--------------------------------------------------

KFold

↓

General Regression Problems

--------------------------------------------------

StratifiedKFold

↓

Classification Problems

Industry Standard

--------------------------------------------------

TimeSeriesSplit

↓

Sequential Data

Stocks

Weather

IoT Sensors

--------------------------------------------------

Repeated KFold

↓

More Stable Estimates

--------------------------------------------------

LOOCV

↓

Small Datasets Only

--------------------------------------------------

Always report

Mean Accuracy

+

Standard Deviation

--------------------------------------------------

Cross Validation

does NOT

increase model accuracy.

It improves

confidence in
model evaluation.

--------------------------------------------------

Use a Pipeline

when preprocessing
is required.

Example

StandardScaler

↓

Model

This prevents

Data Leakage.

--------------------------------------------------

Never perform

Feature Scaling

before

Cross Validation.

Scale inside

the Pipeline.

""")

# =============================================================================
# 19. REAL-WORLD APPLICATIONS
# =============================================================================

print("\n")
print("=" * 70)
print("REAL-WORLD APPLICATIONS")
print("=" * 70)

applications = [

"Healthcare AI",

"Fraud Detection",

"Credit Risk Assessment",

"Customer Churn Prediction",

"Recommendation Systems",

"Kaggle Competitions",

"Academic Research",

"Production ML Pipelines",

"AutoML Systems",

"Hyperparameter Optimization"

]

for app in applications:

    print("✔", app)

# =============================================================================
# 20. GITHUB REVISION NOTES
# =============================================================================

print("\n")
print("=" * 70)
print("GITHUB REVISION NOTES")
print("=" * 70)

print("""

Cross Validation

↓

Model Evaluation Technique

--------------------------------------------------

Purpose

↓

Estimate how well
a model generalizes
to unseen data.

--------------------------------------------------

Validation Methods

✔ Hold-Out Validation

✔ K-Fold

✔ Stratified K-Fold

✔ LOOCV

✔ Repeated K-Fold

✔ ShuffleSplit

✔ TimeSeriesSplit

--------------------------------------------------

Important APIs

✔ cross_val_score()

✔ cross_validate()

✔ KFold()

✔ StratifiedKFold()

--------------------------------------------------

Metrics to Report

✔ Mean Accuracy

✔ Standard Deviation

✔ Precision

✔ Recall

✔ F1 Score

✔ ROC-AUC

--------------------------------------------------

Best Practices

✔ Use Pipeline

✔ Avoid Data Leakage

✔ Use StratifiedKFold
  for Classification

✔ Use TimeSeriesSplit
  for Sequential Data

✔ Keep a Final Test Set
  untouched until the end

""")

# =============================================================================
# 21. FINAL TAKEAWAYS
# =============================================================================

print("\n")
print("=" * 70)
print("FINAL TAKEAWAYS")
print("=" * 70)

takeaways = [

"Train-Test Split provides only one estimate.",

"Cross Validation evaluates models across multiple splits.",

"KFold is suitable for regression tasks.",

"StratifiedKFold is preferred for classification.",

"LOOCV is accurate but computationally expensive.",

"Repeated KFold provides more stable estimates.",

"TimeSeriesSplit preserves chronological order.",

"Mean Accuracy measures average performance.",

"Standard Deviation measures stability.",

"Cross Validation improves evaluation reliability, not model accuracy.",

"Always use a Pipeline to prevent data leakage."

]

for i, item in enumerate(takeaways, start=1):

    print(f"{i}. {item}")

# =============================================================================
# 22. WHAT I LEARNED TODAY
# =============================================================================

print("\n")
print("=" * 70)
print("WHAT I LEARNED TODAY")
print("=" * 70)

print("""

Today I learned

✔ Hold-Out Validation

✔ K-Fold Cross Validation

✔ Stratified K-Fold

✔ Leave-One-Out Cross Validation

✔ Repeated K-Fold

✔ ShuffleSplit

✔ TimeSeriesSplit

✔ cross_val_score()

✔ cross_validate()

✔ Mean Accuracy

✔ Standard Deviation

✔ Pipeline

✔ Data Leakage Prevention

✔ Model Benchmarking

✔ Production Evaluation Workflow

Biggest Realization

Building a model
is only half the job.

The real challenge is proving
that the model performs
consistently on unseen data.

Cross Validation provides
that confidence.

""")

# =============================================================================
# 23. TOP ML ENGINEER INSIGHT
# =============================================================================

print("\n")
print("=" * 70)
print("TOP ML ENGINEER INSIGHT")
print("=" * 70)

print("""

Beginners celebrate

95% Accuracy.

Professionals ask

✔ Was it evaluated
  using Cross Validation?

✔ What's the Mean Accuracy?

✔ What's the Standard Deviation?

✔ Was there Data Leakage?

✔ Was the Test Set
  kept untouched?

Trustworthy models
are built on
trustworthy evaluation.

""")

# =============================================================================
# 24. END OF DAY 33
# =============================================================================

print("\n")
print("=" * 70)
print("END OF DAY 33")
print("=" * 70)

print("Next Topic : Hyperparameter Tuning (GridSearchCV & RandomizedSearchCV) 🚀")