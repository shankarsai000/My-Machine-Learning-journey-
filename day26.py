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