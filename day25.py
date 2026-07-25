# ==========================================
# DAY 25: LOGISTIC REGRESSION
# ==========================================

import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
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

print("Dataset Shape")
print(X.shape)

print("\nTarget Classes")
print(data.target_names)

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
    random_state=42,
    max_iter=1000
)

model.fit(X_train, y_train)

print("\nModel Trained Successfully!")

# ------------------------------------------
# 5. PREDICTIONS
# ------------------------------------------

predictions = model.predict(X_test)

probabilities = model.predict_proba(X_test)

print("\nFirst Five Predictions")

print(predictions[:5])

print("\nPrediction Probabilities")

print(probabilities[:5])

# ------------------------------------------
# 6. ACCURACY
# ------------------------------------------

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\nAccuracy")

print(round(accuracy,4))

# ------------------------------------------
# 7. CONFUSION MATRIX
# ------------------------------------------

cm = confusion_matrix(
    y_test,
    predictions
)

print("\nConfusion Matrix")

print(cm)

# ------------------------------------------
# 8. CLASSIFICATION REPORT
# ------------------------------------------

print("\nClassification Report")

print(
    classification_report(
        y_test,
        predictions
    )
)

# ------------------------------------------
# 9. MODEL COEFFICIENTS
# ------------------------------------------

coef = pd.DataFrame({

    "Feature": X.columns,

    "Coefficient":
    model.coef_[0]

})

print("\nTop 10 Important Features")

print(
    coef.sort_values(
        by="Coefficient",
        key=abs,
        ascending=False
    ).head(10)
)

# ------------------------------------------
# 10. NEW SAMPLE
# ------------------------------------------

new_sample = X.iloc[[0]]

new_scaled = scaler.transform(new_sample)

prediction = model.predict(new_scaled)

probability = model.predict_proba(new_scaled)

print("\nPrediction")

print(prediction)

print("\nProbability")

print(probability)

# ------------------------------------------
# 11. INDUSTRY INSIGHT
# ------------------------------------------

print("\nIndustry Insight")

print("Logistic Regression")

print("is one of the strongest")

print("baseline models")

print("for classification.")

# ------------------------------------------
# 12. FINAL TAKEAWAY
# ------------------------------------------

print("\nToday's Learning")

print("Regression predicts values.")

print("Logistic Regression")

print("predicts probabilities")

print("which become classes.")





# ==============================================================
# MACHINE LEARNING NOTES
# TOPIC: LOGISTIC REGRESSION (BINARY CLASSIFICATION)
# ==============================================================

# --------------------------------------------------------------
# 1. WHAT IS CLASSIFICATION?
# --------------------------------------------------------------
#
# Classification is a Supervised Machine Learning technique
# used to predict categorical outputs (classes).
#
# Unlike Regression, which predicts continuous numerical values,
# Classification predicts labels or categories.
#
# Examples
#
# ✔ Spam / Not Spam
# ✔ Cancer / No Cancer
# ✔ Fraud / Genuine
# ✔ Pass / Fail
# ✔ Customer Will Buy / Will Not Buy
#
# Classification predicts CLASSES.
#
# Regression predicts NUMBERS.
#
#
# --------------------------------------------------------------
# 2. WHAT IS LOGISTIC REGRESSION?
# --------------------------------------------------------------
#
# Logistic Regression is one of the simplest and most powerful
# Supervised Machine Learning algorithms used for Classification.
#
# Despite its name,
#
# Logistic Regression IS NOT a Regression algorithm.
#
# It is a Classification algorithm.
#
# Its objective is to predict the probability
# that an input belongs to a particular class.
#
#
# Example
#
# Patient has Cancer?
#
# Probability = 0.93
#
# Since probability > 0.5
#
# Prediction = Cancer
#
#
# --------------------------------------------------------------
# 3. WHY IS IT CALLED "LOGISTIC REGRESSION"?
# --------------------------------------------------------------
#
# It uses a regression equation internally
# but applies a Sigmoid Function
# to convert the output into probabilities.
#
#
# Linear Regression Output
#
# Can be
#
# -200
#
# 50
#
# 300
#
#
# Logistic Regression Output
#
# Always between
#
# 0 and 1
#
#
# Therefore
#
# Logistic Regression predicts probabilities
# instead of continuous numbers.
#
#
# --------------------------------------------------------------
# 4. SIGMOID FUNCTION
# --------------------------------------------------------------
#
# Logistic Regression uses the Sigmoid Function.
#
# Formula
#
#                 1
# P = -------------------------
#      1 + e^(- (wX + b))
#
#
# Properties
#
# ✔ Output always lies between 0 and 1
#
# ✔ Converts any real number
# into a probability
#
#
# Example
#
# Linear Output = 6
#
# Sigmoid Output
#
# 0.997
#
#
# Linear Output = -4
#
# Sigmoid Output
#
# 0.018
#
#
# Therefore
#
# Every prediction becomes
#
# A Probability
#
#
# --------------------------------------------------------------
# 5. DECISION BOUNDARY
# --------------------------------------------------------------
#
# After calculating probability,
#
# Logistic Regression converts
# probability into a class.
#
#
# Default Decision Boundary
#
# Probability >= 0.5
#
# Class = 1
#
#
# Probability < 0.5
#
# Class = 0
#
#
# Example
#
# Probability = 0.82
#
# Prediction = Positive
#
#
# Probability = 0.17
#
# Prediction = Negative
#
#
# --------------------------------------------------------------
# 6. HOW LOGISTIC REGRESSION LEARNS
# --------------------------------------------------------------
#
# During model.fit(X_train, y_train)
#
# the algorithm
#
# ✔ studies the training data
#
# ✔ learns feature importance
#
# ✔ calculates coefficients
#
# ✔ calculates intercept
#
# ✔ minimizes classification error
#
#
# Finally it learns
#
# Best coefficients
#
# and
#
# Best intercept
#
#
# --------------------------------------------------------------
# 7. MATHEMATICAL EQUATION
# --------------------------------------------------------------
#
# First,
#
# Linear Equation
#
# z =
#
# w1*x1
#
# +
#
# w2*x2
#
# +
#
# ...
#
# +
#
# b
#
#
# Then
#
# Sigmoid Function
#
#                1
# P = -----------------------
#     1 + e^(-z)
#
#
# Finally
#
# Probability
#
# becomes
#
# Class
#
#
# --------------------------------------------------------------
# 8. FEATURE COEFFICIENTS
# --------------------------------------------------------------
#
# Every feature receives
# one coefficient.
#
#
# Positive Coefficient
#
# Increasing that feature
# increases the probability
# of the positive class.
#
#
# Negative Coefficient
#
# Increasing that feature
# decreases the probability
# of the positive class.
#
#
# Larger Absolute Coefficient
#
# Greater influence
#
#
# --------------------------------------------------------------
# 9. INTERCEPT
# --------------------------------------------------------------
#
# Intercept is the starting point
# of the decision function.
#
# It shifts the decision boundary.
#
#
# --------------------------------------------------------------
# 10. FEATURE SCALING
# --------------------------------------------------------------
#
# Logistic Regression
#
# NEEDS Feature Scaling.
#
#
# Why?
#
# Because it uses optimization algorithms
# like Gradient Descent
# which converge faster
# when features are on similar scales.
#
#
# Best Practice
#
# Use StandardScaler.
#
#
# --------------------------------------------------------------
# 11. TRAINING PIPELINE
# --------------------------------------------------------------
#
# Load Dataset
#
# ↓
#
# Split Dataset
#
# ↓
#
# Feature Scaling
#
# ↓
#
# Train Logistic Regression
#
# ↓
#
# Learn Coefficients
#
# ↓
#
# Predict Probabilities
#
# ↓
#
# Convert Probability
#
# into Class
#
#
# --------------------------------------------------------------
# 12. PREDICTION METHODS
# --------------------------------------------------------------
#
# model.predict()
#
# Returns
#
# Final Class
#
#
# Example
#
# 0
#
# 1
#
# 1
#
#
# model.predict_proba()
#
# Returns
#
# Probability
# of every class.
#
#
# Example
#
# [[0.96 0.04]
#  [0.08 0.92]]
#
#
# Meaning
#
# First Sample
#
# 96%
#
# Class 0
#
# 4%
#
# Class 1
#
#
# Second Sample
#
# 8%
#
# Class 0
#
# 92%
#
# Class 1
#
#
# --------------------------------------------------------------
# 13. MODEL EVALUATION
# --------------------------------------------------------------
#
# Logistic Regression
# is evaluated using
#
# Classification Metrics.
#
#
# --------------------------------------------------------------
# 14. ACCURACY
# --------------------------------------------------------------
#
# Accuracy
#
# =
#
# Correct Predictions
#
# --------------------
#
# Total Predictions
#
#
# Example
#
# 98 correct
#
# out of
#
# 100
#
#
# Accuracy
#
# =
#
# 98%
#
#
# Best when
#
# classes are balanced.
#
#
# --------------------------------------------------------------
# 15. CONFUSION MATRIX
# --------------------------------------------------------------
#
# Confusion Matrix
#
# shows
#
# Correct
#
# and
#
# Wrong Predictions.
#
#
#                Predicted
#
#             0          1
#
# Actual
#
# 0         TN         FP
#
# 1         FN         TP
#
#
# TN
#
# True Negative
#
#
# FP
#
# False Positive
#
#
# FN
#
# False Negative
#
#
# TP
#
# True Positive
#
#
# --------------------------------------------------------------
# 16. PRECISION
# --------------------------------------------------------------
#
# Precision answers
#
# "Out of all predicted positives,
# how many were actually positive?"
#
#
# Formula
#
# Precision
#
# =
#
# TP
#
# ---------
#
# TP + FP
#
#
# High Precision
#
# Few False Positives.
#
#
# --------------------------------------------------------------
# 17. RECALL
# --------------------------------------------------------------
#
# Recall answers
#
# "Out of all actual positives,
# how many did we correctly detect?"
#
#
# Formula
#
# Recall
#
# =
#
# TP
#
# ---------
#
# TP + FN
#
#
# High Recall
#
# Few False Negatives.
#
#
# --------------------------------------------------------------
# 18. F1 SCORE
# --------------------------------------------------------------
#
# F1 Score
#
# balances
#
# Precision
#
# and
#
# Recall.
#
#
# Formula
#
# F1 =
#
# 2 ×
#
# Precision × Recall
#
# -------------------
#
# Precision + Recall
#
#
# Higher F1
#
# Better model.
#
#
# --------------------------------------------------------------
# 19. CLASSIFICATION REPORT
# --------------------------------------------------------------
#
# classification_report()
#
# prints
#
# ✔ Precision
#
# ✔ Recall
#
# ✔ F1 Score
#
# ✔ Support
#
#
# Support
#
# =
#
# Number of samples
#
# belonging
#
# to each class.
#
#
# --------------------------------------------------------------
# 20. HYPERPARAMETERS
# --------------------------------------------------------------
#
# max_iter
#
# Maximum optimization iterations.
#
#
# Higher value
#
# More time to converge.
#
#
# random_state
#
# Makes results reproducible.
#
#
# solver
#
# Optimization algorithm
#
# used internally.
#
#
# penalty
#
# Controls Regularization.
#
#
# --------------------------------------------------------------
# 21. REGULARIZATION
# --------------------------------------------------------------
#
# Logistic Regression
#
# also supports
#
# Regularization.
#
#
# L2
#
# Default
#
# Ridge Regularization.
#
#
# L1
#
# Feature Selection
#
#
# ElasticNet
#
# Combination
#
# of L1 and L2.
#
#
# Regularization
#
# reduces overfitting.
#
#
# --------------------------------------------------------------
# 22. ADVANTAGES
# --------------------------------------------------------------
#
# ✔ Simple
#
# ✔ Fast
#
# ✔ Easy to interpret
#
# ✔ Excellent baseline model
#
# ✔ Works well on small datasets
#
# ✔ Predicts probabilities
#
# ✔ Supports Regularization
#
#
# --------------------------------------------------------------
# 23. LIMITATIONS
# --------------------------------------------------------------
#
# ✘ Assumes roughly linear decision boundaries.
#
# ✘ Cannot capture very complex patterns
# without feature engineering.
#
# ✘ Sensitive to outliers.
#
# ✘ Requires Feature Scaling.
#
#
# --------------------------------------------------------------
# 24. REAL WORLD APPLICATIONS
# --------------------------------------------------------------
#
# ✔ Disease Detection
#
# ✔ Email Spam Detection
#
# ✔ Credit Card Fraud Detection
#
# ✔ Customer Churn Prediction
#
# ✔ Loan Approval
#
# ✔ Employee Attrition Prediction
#
# ✔ Marketing Response Prediction
#
#
# --------------------------------------------------------------
# 25. LINEAR vs LOGISTIC REGRESSION
# --------------------------------------------------------------
#
# Linear Regression
#
# Purpose
#
# Predict Numbers
#
#
# Output
#
# Continuous Values
#
#
# Examples
#
# House Price
#
# Salary
#
# Temperature
#
#
# Evaluation
#
# MAE
#
# MSE
#
# RMSE
#
# R²
#
#
# Logistic Regression
#
# Purpose
#
# Predict Classes
#
#
# Output
#
# Probability
#
# then
#
# Class
#
#
# Examples
#
# Spam
#
# Cancer
#
# Fraud
#
# Pass / Fail
#
#
# Evaluation
#
# Accuracy
#
# Precision
#
# Recall
#
# F1 Score
#
# Confusion Matrix
#
#
# --------------------------------------------------------------
# 26. INTERVIEW QUESTIONS
# --------------------------------------------------------------
#
# Q1. Is Logistic Regression a Regression algorithm?
#
# No.
#
# It is a Classification algorithm.
#
#
# Q2. Why is it called Logistic Regression?
#
# Because it uses a regression equation
# followed by a Sigmoid Function
# to predict probabilities.
#
#
# Q3. Why Feature Scaling?
#
# To improve optimization
# and faster convergence.
#
#
# Q4. Difference between predict()
# and predict_proba()?
#
# predict()
#
# Returns Final Class.
#
#
# predict_proba()
#
# Returns Class Probabilities.
#
#
# Q5. What does Sigmoid Function do?
#
# Converts any real number
# into a probability
# between 0 and 1.
#
#
# --------------------------------------------------------------
# 27. COMPLETE WORKFLOW
# --------------------------------------------------------------
#
# Load Dataset
#
# ↓
#
# Split Data
#
# ↓
#
# Feature Scaling
#
# ↓
#
# Train Logistic Regression
#
# ↓
#
# Learn Coefficients
#
# ↓
#
# Predict Probabilities
#
# ↓
#
# Apply Decision Boundary
#
# ↓
#
# Predict Class
#
# ↓
#
# Evaluate
#
# ↓
#
# Deploy Model
#
#
# --------------------------------------------------------------
# 28. FINAL TAKEAWAY
# --------------------------------------------------------------
#
# Logistic Regression is one of the most important
# Classification algorithms in Machine Learning.
#
# It predicts probabilities using the Sigmoid Function
# and converts them into classes using a decision boundary.
#
# It is simple, fast, interpretable, supports
# regularization, and is considered one of the best
# baseline models for binary classification problems.
#
# Every Machine Learning Engineer should understand
# Logistic Regression before moving to Decision Trees,
# Random Forests, Support Vector Machines,
# Gradient Boosting, or Deep Learning.
#
# ==============================================================