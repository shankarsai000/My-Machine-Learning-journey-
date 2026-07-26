# ==========================================
# DAY 26: ADVANCED MODEL EVALUATION METRICS
# ==========================================

import time
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    precision_recall_curve,
    auc,
    log_loss,
    brier_score_loss,
    matthews_corrcoef,
    cohen_kappa_score
)

# ------------------------------------------
# 1. LOAD DATASET
# ------------------------------------------

data = load_breast_cancer()

X = pd.DataFrame(
    data.data,
    columns=data.feature_names
)

y = data.target

print("="*50)
print("DATASET INFORMATION")
print("="*50)

print("Dataset Shape :", X.shape)
print("Target Classes:", data.target_names)

# ------------------------------------------
# 2. TRAIN TEST SPLIT
# ------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ------------------------------------------
# 3. FEATURE SCALING
# ------------------------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ------------------------------------------
# 4. TRAIN MODEL
# ------------------------------------------

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(X_train, y_train)

print("\nModel Trained Successfully!")

# ------------------------------------------
# 5. LATENCY & THROUGHPUT
# ------------------------------------------

start = time.perf_counter()

pred = model.predict(X_test)

end = time.perf_counter()

latency = (end - start) * 1000

throughput = len(X_test) / (end - start)

prob = model.predict_proba(X_test)[:,1]

# ------------------------------------------
# 6. ACCURACY
# ------------------------------------------

accuracy = accuracy_score(y_test,pred)

print("\nAccuracy")

print(round(accuracy,4))

# ------------------------------------------
# 7. PRECISION
# ------------------------------------------

precision = precision_score(y_test,pred)

print("\nPrecision")

print(round(precision,4))

# ------------------------------------------
# 8. RECALL
# ------------------------------------------

recall = recall_score(y_test,pred)

print("\nRecall")

print(round(recall,4))

# ------------------------------------------
# 9. F1 SCORE
# ------------------------------------------

f1 = f1_score(y_test,pred)

print("\nF1 Score")

print(round(f1,4))

# ------------------------------------------
# 10. ROC AUC
# ------------------------------------------

roc = roc_auc_score(y_test,prob)

print("\nROC-AUC")

print(round(roc,4))

# ------------------------------------------
# 11. PR AUC
# ------------------------------------------

precision_curve, recall_curve, thresholds = precision_recall_curve(
    y_test,
    prob
)

pr_auc = auc(
    recall_curve,
    precision_curve
)

print("\nPR-AUC")

print(round(pr_auc,4))

# ------------------------------------------
# 12. LOG LOSS
# ------------------------------------------

loss = log_loss(
    y_test,
    prob
)

print("\nLog Loss")

print(round(loss,4))

# ------------------------------------------
# 13. BRIER SCORE
# ------------------------------------------

brier = brier_score_loss(
    y_test,
    prob
)

print("\nBrier Score")

print(round(brier,4))

# ------------------------------------------
# 14. BALANCED ACCURACY
# ------------------------------------------

bal_acc = balanced_accuracy_score(
    y_test,
    pred
)

print("\nBalanced Accuracy")

print(round(bal_acc,4))

# ------------------------------------------
# 15. MATTHEWS CORRELATION COEFFICIENT
# ------------------------------------------

mcc = matthews_corrcoef(
    y_test,
    pred
)

print("\nMatthews Correlation Coefficient")

print(round(mcc,4))

# ------------------------------------------
# 16. COHEN KAPPA SCORE
# ------------------------------------------

kappa = cohen_kappa_score(
    y_test,
    pred
)

print("\nCohen Kappa")

print(round(kappa,4))

# ------------------------------------------
# 17. CONFUSION MATRIX
# ------------------------------------------

cm = confusion_matrix(
    y_test,
    pred
)

print("\nConfusion Matrix")

print(cm)

# ------------------------------------------
# 18. CLASSIFICATION REPORT
# ------------------------------------------

print("\nClassification Report")

print(
    classification_report(
        y_test,
        pred
    )
)

# ------------------------------------------
# 19. LATENCY
# ------------------------------------------

print("\nPrediction Latency")

print(round(latency,4),"ms")

# ------------------------------------------
# 20. THROUGHPUT
# ------------------------------------------

print("\nThroughput")

print(round(throughput,2),"predictions/sec")

# ------------------------------------------
# 21. METRIC GUIDE
# ------------------------------------------

print("\nMetric Guide")

print("Accuracy          -> Overall correctness")

print("Balanced Accuracy -> Handles imbalanced datasets")

print("Precision         -> Reduces False Positives")

print("Recall            -> Reduces False Negatives")

print("F1 Score          -> Balance Precision & Recall")

print("ROC-AUC           -> Overall class separation")

print("PR-AUC            -> Best for imbalanced datasets")

print("Log Loss          -> Confidence of predictions")

print("Brier Score       -> Probability calibration")

print("MCC               -> Balanced binary evaluation")

print("Cohen Kappa       -> Agreement beyond chance")

# ------------------------------------------
# 22. WHERE TO USE WHAT?
# ------------------------------------------

print("\nReal World Applications")

print("Cancer Detection      -> Recall")

print("Fraud Detection       -> Recall")

print("Spam Filter           -> Precision")

print("Medical Diagnosis     -> ROC-AUC")

print("Imbalanced Dataset    -> PR-AUC")

print("Probability Models    -> Log Loss")

print("Probability Quality   -> Brier Score")

print("General Classification-> F1 Score")

print("Highly Imbalanced ML  -> MCC")

# ------------------------------------------
# 23. TOP ML ENGINEER INSIGHT
# ------------------------------------------

print("\nTop ML Engineer Insight")

print("Never evaluate a model using only Accuracy.")

print("Choose evaluation metrics based on the")

print("business objective and the cost of errors.")

print("Production ML systems monitor both")

print("prediction quality and operational metrics")

print("such as latency and throughput.")


# =============================================================================
# DAY 26 : ADVANCED MODEL EVALUATION METRICS
# =============================================================================
#
# WHY THIS NOTEBOOK?
# -----------------------------------------------------------------------------
#
# Training a Machine Learning model is only half of the job.
#
# The real question is:
#
#     "How good is my model?"
#
# A model with very high Accuracy is NOT always a good model.
#
# Example:
#
# Suppose a dataset contains
#
#     990 Healthy Patients
#      10 Cancer Patients
#
# If the model predicts EVERY patient as Healthy,
#
# Accuracy becomes
#
#     990 / 1000 = 99%
#
# Yet the model completely fails because it detects
# ZERO cancer patients.
#
# Therefore,
#
# Accuracy alone should NEVER be used to evaluate
# classification models.
#
#
# =============================================================================
# GOAL OF THIS NOTEBOOK
# =============================================================================
#
# Learn how professional ML engineers evaluate
# classification models.
#
# Metrics covered:
#
# ✔ Accuracy
# ✔ Balanced Accuracy
# ✔ Precision
# ✔ Recall
# ✔ F1 Score
# ✔ ROC-AUC
# ✔ PR-AUC
# ✔ Log Loss
# ✔ Brier Score
# ✔ Matthews Correlation Coefficient (MCC)
# ✔ Cohen's Kappa
# ✔ Confusion Matrix
# ✔ Classification Report
#
# Production Metrics:
#
# ✔ Latency
# ✔ Throughput
#
#
# =============================================================================
# WHY DO WE NEED MULTIPLE METRICS?
# =============================================================================
#
# Every metric answers a different question.
#
# Accuracy
#
#     "Overall, how many predictions were correct?"
#
#
# Precision
#
#     "Out of everything predicted as Positive,
#      how many were actually Positive?"
#
#
# Recall
#
#     "Out of all actual Positive samples,
#      how many did we detect?"
#
#
# F1 Score
#
#     "How balanced are Precision and Recall?"
#
#
# ROC-AUC
#
#     "How well does the model separate
#      Positive and Negative classes?"
#
#
# PR-AUC
#
#     "How good is the model on imbalanced datasets?"
#
#
# Log Loss
#
#     "How confident are the probability predictions?"
#
#
# Brier Score
#
#     "How accurate are the predicted probabilities?"
#
#
# MCC
#
#     "Overall classification quality,
#      especially for imbalanced datasets."
#
#
# Cohen's Kappa
#
#     "How much better is the model than
#      random guessing?"
#
#
# =============================================================================
# WHY PRODUCTION ML IS DIFFERENT
# =============================================================================
#
# Most beginners stop after printing:
#
# Accuracy
#
# Professional ML Engineers ask:
#
#     Is the model reliable?
#
#     Is it calibrated?
#
#     Is it fast enough?
#
#     Can users trust it?
#
#     Does it generalize?
#
#     Is it fair?
#
#     Does it improve business KPIs?
#
#
# Therefore, real ML systems also monitor:
#
# • Latency
# • Throughput
# • Calibration
# • Drift
# • Fairness
# • Revenue Impact
# • Customer Satisfaction
# • Alert Fatigue
#
#
# =============================================================================
# WHICH METRIC SHOULD YOU USE?
# =============================================================================
#
# Cancer Detection
#
#     Recall
#
# Missing a patient is very expensive.
#
#
# Fraud Detection
#
#     Recall
#
# Missing fraud is costly.
#
#
# Spam Detection
#
#     Precision
#
# Avoid blocking genuine emails.
#
#
# Search Ranking
#
#     ROC-AUC
#
# Compare how well models rank results.
#
#
# Imbalanced Classification
#
#     PR-AUC
#
# Accuracy becomes misleading.
#
#
# General Classification
#
#     F1 Score
#
# Balance Precision and Recall.
#
#
# Probability Prediction
#
#     Log Loss
#     Brier Score
#
#
# =============================================================================
# PRODUCTION MINDSET
# =============================================================================
#
# Beginner:
#
#     "Which model has the highest Accuracy?"
#
#
# Professional:
#
#     "Which type of mistake costs the business the most?"
#
#
# Business Cost determines
#
# which evaluation metric matters.
#
#
# =============================================================================
# GOLDEN RULE
# =============================================================================
#
# NEVER optimize a model simply to increase Accuracy.
#
# Instead ask:
#
# 1. What problem am I solving?
#
# 2. What kind of mistakes are expensive?
#
# 3. Which metric represents that business objective?
#
# 4. Does improving this metric actually improve
#    the real-world system?
#
#
# =============================================================================
# WHAT I LEARNED TODAY
# =============================================================================
#
# ✔ Accuracy is NOT everything.
#
# ✔ Every metric tells a different story.
#
# ✔ Business objectives decide which metric matters.
#
# ✔ Production ML is much more than training models.
#
# ✔ Evaluation is one of the most important skills
#   of an ML Engineer.
#
# =============================================================================