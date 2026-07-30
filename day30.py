# =============================================================================
# DAY 30 : GRADIENT BOOSTING, XGBOOST & LIGHTGBM
# PART 4.1 : GRADIENT BOOSTING CLASSIFIER
# =============================================================================

"""
Topics Covered

1. Load Dataset
2. Train-Test Split
3. Train Gradient Boosting Classifier
4. Predictions
5. Accuracy
6. Precision
7. Recall
8. F1 Score
9. ROC-AUC
10. Confusion Matrix
11. Classification Report
12. Feature Importance
13. New Sample Prediction
14. Model Summary
15. Top ML Engineer Insight

"""

# =============================================================================
# IMPORT LIBRARIES
# =============================================================================

import pandas as pd
import numpy as np

from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import train_test_split

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import (

    accuracy_score,

    precision_score,

    recall_score,

    f1_score,

    roc_auc_score,

    confusion_matrix,

    classification_report

)

import matplotlib.pyplot as plt

# =============================================================================
# 1. LOAD DATASET
# =============================================================================

print("=" * 70)
print("LOADING DATASET")
print("=" * 70)

data = load_breast_cancer()

X = pd.DataFrame(

    data.data,

    columns=data.feature_names

)

y = data.target

print("\nDataset Shape :", X.shape)

print("\nTarget Classes")

print(data.target_names)

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

print("\nTraining Samples :", len(X_train))

print("Testing Samples  :", len(X_test))

# =============================================================================
# WHY NO FEATURE SCALING?
# =============================================================================

print("\nFeature Scaling")

print("Gradient Boosting uses Decision Trees.")

print("Decision Trees split using thresholds.")

print("Therefore Feature Scaling is NOT required.")

# =============================================================================
# 3. TRAIN MODEL
# =============================================================================

print("\n")
print("=" * 70)
print("TRAINING GRADIENT BOOSTING")
print("=" * 70)

gb = GradientBoostingClassifier(

    n_estimators=100,

    learning_rate=0.1,

    max_depth=3,

    random_state=42

)

gb.fit(

    X_train,

    y_train

)

print("Model Trained Successfully!")

# =============================================================================
# 4. PREDICTIONS
# =============================================================================

pred = gb.predict(

    X_test

)

prob = gb.predict_proba(

    X_test

)[:,1]

print("\nFirst Five Predictions")

print(pred[:5])

print("\nPrediction Probabilities")

print(np.round(prob[:5],4))

# =============================================================================
# 5. EVALUATION METRICS
# =============================================================================

accuracy = accuracy_score(

    y_test,

    pred

)

precision = precision_score(

    y_test,

    pred

)

recall = recall_score(

    y_test,

    pred

)

f1 = f1_score(

    y_test,

    pred

)

roc = roc_auc_score(

    y_test,

    prob

)

print("\n")
print("=" * 70)
print("MODEL EVALUATION")
print("=" * 70)

print("Accuracy  :", round(accuracy,4))

print("Precision :", round(precision,4))

print("Recall    :", round(recall,4))

print("F1 Score  :", round(f1,4))

print("ROC-AUC   :", round(roc,4))

# =============================================================================
# 6. CONFUSION MATRIX
# =============================================================================

print("\n")
print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

cm = confusion_matrix(

    y_test,

    pred

)

print(cm)

# =============================================================================
# 7. CLASSIFICATION REPORT
# =============================================================================

print("\n")
print("=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(

    classification_report(

        y_test,

        pred,

        target_names=data.target_names

    )

)

# =============================================================================
# 8. FEATURE IMPORTANCE
# =============================================================================

importance = pd.DataFrame({

    "Feature": X.columns,

    "Importance": gb.feature_importances_

})

importance = importance.sort_values(

    by="Importance",

    ascending=False

)

print("\n")
print("=" * 70)
print("TOP 10 IMPORTANT FEATURES")
print("=" * 70)

print(

    importance.head(10)

)

plt.figure(figsize=(10,6))

plt.barh(

    importance["Feature"][:10],

    importance["Importance"][:10]

)

plt.gca().invert_yaxis()

plt.title("Gradient Boosting Feature Importance")

plt.xlabel("Importance")

plt.grid(True)

plt.show()

# =============================================================================
# 9. NEW SAMPLE PREDICTION
# =============================================================================

new_sample = X.iloc[[0]]

prediction = gb.predict(

    new_sample

)

probability = gb.predict_proba(

    new_sample

)

print("\n")
print("=" * 70)
print("NEW SAMPLE PREDICTION")
print("=" * 70)

print("Predicted Class :", prediction[0])

print("Prediction Probability")

print(np.round(probability,4))

# =============================================================================
# 10. MODEL INFORMATION
# =============================================================================

print("\n")
print("=" * 70)
print("MODEL INFORMATION")
print("=" * 70)

print("Algorithm      : Gradient Boosting")

print("Trees          :", gb.n_estimators)

print("Learning Rate  :", gb.learning_rate)

print("Max Depth      :", gb.max_depth)

# =============================================================================
# 11. MODEL SUMMARY
# =============================================================================

print("\n")
print("=" * 70)
print("MODEL SUMMARY")
print("=" * 70)

print("""

Algorithm

↓

Gradient Boosting

Learning Style

↓

Sequential Learning

Weak Learner

↓

Decision Tree

Prediction Strategy

↓

Correct Previous Errors

Feature Scaling

↓

Not Required

Supports

↓

Classification

Regression

""")

# =============================================================================
# 12. TOP ML ENGINEER INSIGHT
# =============================================================================

print("\n")
print("=" * 70)
print("TOP ML ENGINEER INSIGHT")
print("=" * 70)

print("""

Gradient Boosting does not build
many independent trees.

Instead,

every new tree learns from
the mistakes of the previous trees.

This allows the model to
gradually improve prediction quality.

Always tune:

✔ learning_rate

✔ n_estimators

✔ max_depth

These three hyperparameters
have the biggest impact
on model performance.

Never evaluate the model
using only Accuracy.

Always analyse:

✔ Precision

✔ Recall

✔ F1 Score

✔ ROC-AUC

✔ Confusion Matrix

""")

# =============================================================================
# DAY 30 : GRADIENT BOOSTING, XGBOOST & LIGHTGBM
# PART 4.2 : XGBOOST CLASSIFIER
# =============================================================================

"""
Topics Covered

1. Import XGBoost
2. Train XGBoost Classifier
3. Predictions
4. Accuracy
5. Precision
6. Recall
7. F1 Score
8. ROC-AUC
9. Confusion Matrix
10. Classification Report
11. Feature Importance
12. Hyperparameters
13. New Sample Prediction
14. Model Summary
15. Top ML Engineer Insight

"""

# =============================================================================
# IMPORT LIBRARIES
# =============================================================================

from xgboost import XGBClassifier

# =============================================================================
# 1. TRAIN XGBOOST MODEL
# =============================================================================

print("\n")
print("="*70)
print("TRAINING XGBOOST")
print("="*70)

xgb = XGBClassifier(

    n_estimators=100,

    learning_rate=0.1,

    max_depth=3,

    subsample=0.8,

    colsample_bytree=0.8,

    random_state=42,

    eval_metric="logloss"

)

xgb.fit(

    X_train,

    y_train

)

print("XGBoost Model Trained Successfully!")

# =============================================================================
# 2. PREDICTIONS
# =============================================================================

pred = xgb.predict(

    X_test

)

prob = xgb.predict_proba(

    X_test

)[:,1]

print("\nFirst Five Predictions")

print(pred[:5])

print("\nPrediction Probabilities")

print(np.round(prob[:5],4))

# =============================================================================
# 3. MODEL EVALUATION
# =============================================================================

accuracy = accuracy_score(

    y_test,

    pred

)

precision = precision_score(

    y_test,

    pred

)

recall = recall_score(

    y_test,

    pred

)

f1 = f1_score(

    y_test,

    pred

)

roc = roc_auc_score(

    y_test,

    prob

)

print("\n")
print("="*70)
print("MODEL EVALUATION")
print("="*70)

print("Accuracy  :", round(accuracy,4))

print("Precision :", round(precision,4))

print("Recall    :", round(recall,4))

print("F1 Score  :", round(f1,4))

print("ROC-AUC   :", round(roc,4))

# =============================================================================
# 4. CONFUSION MATRIX
# =============================================================================

print("\n")
print("="*70)
print("CONFUSION MATRIX")
print("="*70)

cm = confusion_matrix(

    y_test,

    pred

)

print(cm)

# =============================================================================
# 5. CLASSIFICATION REPORT
# =============================================================================

print("\n")
print("="*70)
print("CLASSIFICATION REPORT")
print("="*70)

print(

    classification_report(

        y_test,

        pred,

        target_names=data.target_names

    )

)

# =============================================================================
# 6. FEATURE IMPORTANCE
# =============================================================================

importance = pd.DataFrame({

    "Feature":X.columns,

    "Importance":xgb.feature_importances_

})

importance = importance.sort_values(

    by="Importance",

    ascending=False

)

print("\n")
print("="*70)
print("TOP 10 IMPORTANT FEATURES")
print("="*70)

print(

    importance.head(10)

)

plt.figure(figsize=(10,6))

plt.barh(

    importance["Feature"][:10],

    importance["Importance"][:10]

)

plt.gca().invert_yaxis()

plt.title("XGBoost Feature Importance")

plt.xlabel("Importance")

plt.grid(True)

plt.show()

# =============================================================================
# 7. HYPERPARAMETERS
# =============================================================================

print("\n")
print("="*70)
print("IMPORTANT HYPERPARAMETERS")
print("="*70)

hyper = pd.DataFrame({

    "Hyperparameter":[

        "n_estimators",

        "learning_rate",

        "max_depth",

        "subsample",

        "colsample_bytree"

    ],

    "Value":[

        xgb.n_estimators,

        xgb.learning_rate,

        xgb.max_depth,

        xgb.subsample,

        xgb.colsample_bytree

    ]

})

print(hyper)

# =============================================================================
# 8. NEW SAMPLE PREDICTION
# =============================================================================

new_sample = X.iloc[[0]]

prediction = xgb.predict(

    new_sample

)

probability = xgb.predict_proba(

    new_sample

)

print("\n")
print("="*70)
print("NEW SAMPLE PREDICTION")
print("="*70)

print("Predicted Class :", prediction[0])

print("\nPrediction Probability")

print(np.round(probability,4))

# =============================================================================
# 9. MODEL SUMMARY
# =============================================================================

print("\n")
print("="*70)
print("MODEL SUMMARY")
print("="*70)

print("""

Algorithm

↓

XGBoost

Full Form

↓

Extreme Gradient Boosting

Learning Style

↓

Sequential Learning

Supports

↓

Classification

Regression

Feature Scaling

↓

Not Required

Strength

↓

Fast

Accurate

Regularized

Handles Missing Values

""")

# =============================================================================
# 10. WHY XGBOOST IS BETTER
# =============================================================================

print("\n")
print("="*70)
print("WHY XGBOOST?")
print("="*70)

print("""

Compared to traditional
Gradient Boosting,

XGBoost provides

✔ Regularization

✔ Tree Pruning

✔ Missing Value Handling

✔ Parallel Processing

✔ Better Optimization

✔ Faster Training

✔ Better Generalization

These improvements make
XGBoost one of the strongest
algorithms for structured data.

""")

# =============================================================================
# 11. TOP ML ENGINEER INSIGHT
# =============================================================================

print("\n")
print("="*70)
print("TOP ML ENGINEER INSIGHT")
print("="*70)

print("""

Never assume XGBoost
will always outperform
every other algorithm.

Always compare

✔ Logistic Regression

✔ Random Forest

✔ Gradient Boosting

✔ XGBoost

✔ LightGBM

✔ CatBoost

Use Cross Validation,

Hyperparameter Tuning,

and Business Metrics

before selecting
the final model.

The best model is the one
that performs best on
your specific problem,
not the most popular one.

""")

# =============================================================================
# DAY 30 : GRADIENT BOOSTING, XGBOOST & LIGHTGBM
# PART 4.3 : LIGHTGBM & MODEL COMPARISON
# =============================================================================

"""
Topics Covered

1. Train LightGBM
2. Model Evaluation
3. Feature Importance
4. Compare All Boosting Models
5. Training Time
6. Prediction Time
7. Accuracy Comparison
8. Feature Importance
9. Best Model Selection
10. Industry Insight

"""

# =============================================================================
# IMPORT LIBRARIES
# =============================================================================

import time

from lightgbm import LGBMClassifier

# =============================================================================
# 1. TRAIN LIGHTGBM
# =============================================================================

print("\n")
print("="*70)
print("TRAINING LIGHTGBM")
print("="*70)

start = time.time()

lgbm = LGBMClassifier(

    n_estimators=100,

    learning_rate=0.1,

    max_depth=3,

    random_state=42

)

lgbm.fit(

    X_train,

    y_train

)

lgb_train_time = time.time() - start

print("LightGBM Model Trained Successfully!")

# =============================================================================
# 2. PREDICTIONS
# =============================================================================

start = time.time()

lgb_pred = lgbm.predict(

    X_test

)

lgb_prediction_time = time.time() - start

lgb_prob = lgbm.predict_proba(

    X_test

)[:,1]

# =============================================================================
# 3. MODEL EVALUATION
# =============================================================================

lgb_accuracy = accuracy_score(

    y_test,

    lgb_pred

)

lgb_precision = precision_score(

    y_test,

    lgb_pred

)

lgb_recall = recall_score(

    y_test,

    lgb_pred

)

lgb_f1 = f1_score(

    y_test,

    lgb_pred

)

lgb_roc = roc_auc_score(

    y_test,

    lgb_prob

)

print("\n")
print("="*70)
print("LIGHTGBM EVALUATION")
print("="*70)

print("Accuracy  :", round(lgb_accuracy,4))

print("Precision :", round(lgb_precision,4))

print("Recall    :", round(lgb_recall,4))

print("F1 Score  :", round(lgb_f1,4))

print("ROC-AUC   :", round(lgb_roc,4))

# =============================================================================
# 4. FEATURE IMPORTANCE
# =============================================================================

importance = pd.DataFrame({

    "Feature":X.columns,

    "Importance":lgbm.feature_importances_

})

importance = importance.sort_values(

    by="Importance",

    ascending=False

)

print("\n")
print("="*70)
print("TOP 10 IMPORTANT FEATURES")
print("="*70)

print(

    importance.head(10)

)

plt.figure(figsize=(10,6))

plt.barh(

    importance["Feature"][:10],

    importance["Importance"][:10]

)

plt.gca().invert_yaxis()

plt.title("LightGBM Feature Importance")

plt.xlabel("Importance")

plt.grid(True)

plt.show()

# =============================================================================
# 5. TRAINING TIME COMPARISON
# =============================================================================

print("\n")
print("="*70)
print("TRAINING TIME COMPARISON")
print("="*70)

gb_start = time.time()

gb.fit(

    X_train,

    y_train

)

gb_time = time.time() - gb_start

xgb_start = time.time()

xgb.fit(

    X_train,

    y_train

)

xgb_time = time.time() - xgb_start

training = pd.DataFrame({

    "Model":[

        "Gradient Boosting",

        "XGBoost",

        "LightGBM"

    ],

    "Training Time (sec)":[

        gb_time,

        xgb_time,

        lgb_train_time

    ]

})

print(training)

# =============================================================================
# 6. PREDICTION TIME COMPARISON
# =============================================================================

gb_start = time.time()

gb.predict(

    X_test

)

gb_pred_time = time.time() - gb_start

xgb_start = time.time()

xgb.predict(

    X_test

)

xgb_pred_time = time.time() - xgb_start

prediction = pd.DataFrame({

    "Model":[

        "Gradient Boosting",

        "XGBoost",

        "LightGBM"

    ],

    "Prediction Time (sec)":[

        gb_pred_time,

        xgb_pred_time,

        lgb_prediction_time

    ]

})

print("\nPrediction Time")

print(prediction)

# =============================================================================
# 7. ACCURACY COMPARISON
# =============================================================================

comparison = pd.DataFrame({

    "Model":[

        "Gradient Boosting",

        "XGBoost",

        "LightGBM"

    ],

    "Accuracy":[

        accuracy,

        accuracy_score(y_test, xgb.predict(X_test)),

        lgb_accuracy

    ]

})

print("\n")
print("="*70)
print("MODEL COMPARISON")
print("="*70)

print(comparison)

plt.figure(figsize=(8,5))

plt.bar(

    comparison["Model"],

    comparison["Accuracy"]

)

plt.ylabel("Accuracy")

plt.title("Boosting Algorithm Comparison")

plt.grid(True)

plt.show()

# =============================================================================
# 8. BEST MODEL
# =============================================================================

best = comparison.loc[

    comparison["Accuracy"].idxmax()

]

print("\n")
print("="*70)
print("BEST MODEL")
print("="*70)

print("Model :", best["Model"])

print("Accuracy :", round(best["Accuracy"],4))

# =============================================================================
# 9. LIGHTGBM SUMMARY
# =============================================================================

print("\n")
print("="*70)
print("LIGHTGBM SUMMARY")
print("="*70)

print("""

Full Form

↓

Light Gradient Boosting Machine

Advantages

↓

✔ Very Fast

✔ Low Memory

✔ Excellent for Large Datasets

✔ High Accuracy

✔ Histogram Learning

✔ Leaf-wise Growth

Feature Scaling

↓

Not Required

""")

# =============================================================================
# 10. TOP ML ENGINEER INSIGHT
# =============================================================================

print("\n")
print("="*70)
print("TOP ML ENGINEER INSIGHT")
print("="*70)

print("""

Do NOT choose a model

only because it has

the highest Accuracy.

Always compare

✔ Accuracy

✔ Precision

✔ Recall

✔ F1 Score

✔ ROC-AUC

✔ Training Time

✔ Prediction Time

✔ Memory Usage

✔ Explainability

✔ Business Requirements

Small datasets

↓

XGBoost

Large datasets

↓

LightGBM

Many categorical features

↓

CatBoost

Interpretability

↓

Decision Tree

Strong baseline

↓

Random Forest

""")

# =============================================================================
# DAY 30 : GRADIENT BOOSTING, XGBOOST & LIGHTGBM
# PART 4.4 : FINAL NOTES, INTERVIEW PREP & GITHUB SUMMARY
# =============================================================================

# =============================================================================
# 16. PRACTICE EXERCISES
# =============================================================================

print("\n")
print("="*70)
print("PRACTICE EXERCISES")
print("="*70)

print("""

Beginner

1. Train a Gradient Boosting model
   with

   learning_rate = 0.05

   learning_rate = 0.1

   learning_rate = 0.2

Compare Accuracy.

------------------------------------------------------------

2. Change

n_estimators

50

100

200

500

Observe

Training Accuracy

Testing Accuracy

Training Time

------------------------------------------------------------

Intermediate

3. Compare

Gradient Boosting

XGBoost

LightGBM

using

Accuracy

Precision

Recall

F1 Score

ROC-AUC

------------------------------------------------------------

4. Compare Feature Importance

Which features consistently remain important?

------------------------------------------------------------

Advanced

5. Perform Hyperparameter Tuning

GridSearchCV

or

RandomizedSearchCV

for

XGBoost

------------------------------------------------------------

6. Download a Kaggle Dataset

Train

Random Forest

Gradient Boosting

XGBoost

LightGBM

Compare all models.

""")

# =============================================================================
# 17. MINI CHALLENGE
# =============================================================================

print("\n")
print("="*70)
print("MINI CHALLENGE")
print("="*70)

print("""

Build a Disease Prediction Model

Steps

✔ Data Cleaning

✔ Train-Test Split

✔ Gradient Boosting

✔ XGBoost

✔ LightGBM

✔ Model Comparison

✔ Feature Importance

✔ ROC-AUC

✔ Confusion Matrix

✔ Classification Report

Bonus

Perform Hyperparameter Tuning

and explain

WHY

one model performed better.

""")

# =============================================================================
# 18. COMMON INTERVIEW QUESTIONS
# =============================================================================

print("\n")
print("="*70)
print("COMMON INTERVIEW QUESTIONS")
print("="*70)

questions = [

"What is Boosting?",

"What is Gradient Boosting?",

"What is a Weak Learner?",

"What are Residuals?",

"Why is it called Gradient Boosting?",

"What is Sequential Learning?",

"What is Learning Rate?",

"What is Shrinkage?",

"Why are shallow trees used?",

"Difference between Random Forest and Gradient Boosting?",

"What is XGBoost?",

"What is the full form of XGBoost?",

"Why does XGBoost perform better?",

"What is Regularization in XGBoost?",

"What is Tree Pruning?",

"How does XGBoost handle missing values?",

"What is LightGBM?",

"What is the full form of LightGBM?",

"What is Histogram-based Learning?",

"What is Leaf-wise Growth?",

"What is CatBoost?",

"When should you use CatBoost?",

"Why do Kaggle winners prefer XGBoost?",

"When would you choose Random Forest over XGBoost?"

]

for question in questions:

    print("✔", question)

# =============================================================================
# 19. SENIOR ML ENGINEER NOTES
# =============================================================================

print("\n")
print("="*70)
print("SENIOR ML ENGINEER NOTES")
print("="*70)

print("""

Never choose Boosting
only because it has
higher Accuracy.

Always evaluate

✔ Accuracy

✔ Precision

✔ Recall

✔ F1 Score

✔ ROC-AUC

✔ Training Time

✔ Prediction Time

✔ Memory Usage

✔ Explainability

✔ Business Cost

------------------------------------------------------------

Learning Rate

and

n_estimators

work together.

Smaller Learning Rate

↓

Need More Trees

Larger Learning Rate

↓

Need Fewer Trees

------------------------------------------------------------

If the model overfits,

don't immediately
reduce the number of trees.

Instead try

✔ Lower Learning Rate

✔ Smaller max_depth

✔ Regularization

✔ Early Stopping

------------------------------------------------------------

Feature Importance

shows contribution

NOT

cause-and-effect.

------------------------------------------------------------

In production,

always compare

Logistic Regression

Decision Tree

Random Forest

Gradient Boosting

XGBoost

LightGBM

CatBoost

The simplest model that
meets the business goal
is often the best choice.

""")

# =============================================================================
# 20. INDUSTRY APPLICATIONS
# =============================================================================

print("\n")
print("="*70)
print("REAL-WORLD APPLICATIONS")
print("="*70)

applications = [

"Fraud Detection",

"Credit Risk Scoring",

"Customer Churn Prediction",

"Disease Diagnosis",

"Insurance Risk Analysis",

"Recommendation Systems",

"Financial Forecasting",

"Marketing Analytics",

"Cybersecurity Threat Detection",

"Predictive Maintenance"

]

for app in applications:

    print("✔", app)

# =============================================================================
# 21. GITHUB REVISION NOTES
# =============================================================================

print("\n")
print("="*70)
print("GITHUB REVISION NOTES")
print("="*70)

print("""

Gradient Boosting

↓

Sequential Ensemble Learning

------------------------------------------------------------

Core Ideas

✔ Weak Learners

✔ Residual Learning

✔ Gradient Optimization

✔ Learning Rate

✔ Shrinkage

------------------------------------------------------------

Popular Frameworks

✔ Gradient Boosting

✔ XGBoost

✔ LightGBM

✔ CatBoost

------------------------------------------------------------

Gradient Boosting

↓

Simple

Sequential

Good Accuracy

------------------------------------------------------------

XGBoost

↓

Extreme Gradient Boosting

Regularization

Tree Pruning

Missing Value Handling

Parallel Processing

------------------------------------------------------------

LightGBM

↓

Light Gradient Boosting Machine

Histogram Learning

Leaf-wise Growth

Very Fast

Low Memory

------------------------------------------------------------

CatBoost

↓

Categorical Boosting

Best for

Categorical Features

Minimal Preprocessing

------------------------------------------------------------

Feature Scaling

↓

Not Required

------------------------------------------------------------

Most Important Hyperparameters

✔ learning_rate

✔ n_estimators

✔ max_depth

✔ subsample

✔ colsample_bytree

------------------------------------------------------------

Advantages

✔ High Accuracy

✔ Strong Generalization

✔ Excellent for Tabular Data

✔ Feature Importance

------------------------------------------------------------

Limitations

✔ Sensitive to Hyperparameters

✔ Longer Training

✔ Less Interpretable

""")

# =============================================================================
# 22. FINAL TAKEAWAYS
# =============================================================================

print("\n")
print("="*70)
print("FINAL TAKEAWAYS")
print("="*70)

takeaways = [

"Boosting trains models sequentially.",

"Each tree learns from previous errors.",

"Residual Learning is the heart of Gradient Boosting.",

"Learning Rate controls how much each tree contributes.",

"XGBoost adds regularization and pruning.",

"LightGBM focuses on speed and scalability.",

"CatBoost handles categorical features efficiently.",

"Boosting models often outperform Random Forest on tabular data.",

"Hyperparameter tuning is essential for best performance.",

"There is no universally best algorithm.",

"Model selection should always match the business problem."

]

for i, takeaway in enumerate(takeaways, start=1):

    print(f"{i}. {takeaway}")

# =============================================================================
# 23. WHAT I LEARNED TODAY
# =============================================================================

print("\n")
print("="*70)
print("WHAT I LEARNED TODAY")
print("="*70)

print("""

Today I learned

✔ Boosting

✔ Sequential Learning

✔ Weak Learners

✔ Residual Learning

✔ Learning Rate

✔ Shrinkage

✔ Gradient Boosting

✔ XGBoost

✔ LightGBM

✔ CatBoost

✔ Feature Importance

✔ Hyperparameter Tuning

✔ Industry Best Practices

✔ Model Benchmarking

✔ Real-world Applications

Today's biggest realization:

Random Forest builds
many independent trees.

Boosting builds
many cooperative trees.

Instead of voting,

every new tree learns
from previous mistakes.

This simple idea is why
boosting algorithms have become
the gold standard for
many structured data problems.

""")

# =============================================================================
# 24. END OF DAY 30
# =============================================================================

print("\n")
print("="*70)
print("END OF DAY 30")
print("="*70)

print("Next Topic : Cross Validation, GridSearchCV & Hyperparameter Tuning 🚀")