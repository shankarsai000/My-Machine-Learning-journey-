# ==========================================
# DAY 24: RIDGE, LASSO & ELASTICNET
# ==========================================
import pandas as pd
import numpy as np

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet
)
from sklearn.metrics import r2_score
# ------------------------------------------
# 1. LOAD DATASET
# ------------------------------------------
diabetes = load_diabetes()
X = pd.DataFrame(
    diabetes.data,
    columns=diabetes.feature_names
)

y = diabetes.target

print("Dataset Shape")
print(X.shape)


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
# 4. LINEAR REGRESSION
# ------------------------------------------
linear = LinearRegression()
linear.fit(X_train, y_train)
linear_pred = linear.predict(X_test)
print("\nLinear Regression R²")

print(round(
    r2_score(y_test, linear_pred),
    3
))
# ------------------------------------------
# 5. RIDGE REGRESSION
# ------------------------------------------
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
ridge_pred = ridge.predict(X_test)
print("\nRidge Regression R²")
print(round(
    r2_score(y_test, ridge_pred),
    3
))
# ------------------------------------------
# 6. LASSO REGRESSION
# ------------------------------------------
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
lasso_pred = lasso.predict(X_test)
print("\nLasso Regression R²")
print(round(
    r2_score(y_test, lasso_pred),
    3
))
# ------------------------------------------
# 7. ELASTICNET
# ------------------------------------------
elastic = ElasticNet(
    alpha=0.1,
    l1_ratio=0.5
)
elastic.fit(X_train, y_train)

elastic_pred = elastic.predict(X_test)

print("\nElasticNet R²")

print(round(
    r2_score(y_test, elastic_pred),
    3
))

# ------------------------------------------
# 8. COEFFICIENT COMPARISON
# ------------------------------------------

comparison = pd.DataFrame({
    "Feature": X.columns,
    "Linear": linear.coef_,
    "Ridge": ridge.coef_,
    "Lasso": lasso.coef_,
    "ElasticNet": elastic.coef_
})

print("\nModel Coefficients")

print(comparison)

# ------------------------------------------
# 9. LASSO FEATURE SELECTION
# ------------------------------------------

selected = comparison[
    comparison["Lasso"] != 0
]

print("\nFeatures Selected by Lasso")

print(selected["Feature"].tolist())

# ------------------------------------------
# 10. PREDICTION COMPARISON
# ------------------------------------------

results = pd.DataFrame({

    "Actual": y_test[:10],

    "Linear":
        linear_pred[:10].round(1),

    "Ridge":
        ridge_pred[:10].round(1),

    "Lasso":
        lasso_pred[:10].round(1),

    "Elastic":
        elastic_pred[:10].round(1)

})

print("\nPrediction Comparison")

print(results)

# ------------------------------------------
# 11. OVERFITTING IDEA
# ------------------------------------------

print("\nWithout Regularization")

print("Large weights")

print("Complex model")

print("Higher risk of overfitting")

# ------------------------------------------
# 12. RIDGE IDEA
# ------------------------------------------

print("\nRidge")

print("Keeps all features")

print("Shrinks coefficients")

# ------------------------------------------
# 13. LASSO IDEA
# ------------------------------------------

print("\nLasso")

print("Shrinks")

print("Removes unnecessary features")

# ------------------------------------------
# 14. ELASTICNET IDEA
# ------------------------------------------

print("\nElasticNet")

print("L1 + L2")

print("Balanced regularization")

# ------------------------------------------
# 15. TOP ML ENGINEER INSIGHT
# ------------------------------------------

print("\nTop ML Insight")

print("Regularization doesn't")

print("make models more powerful.")

print("It makes them generalize")

print("better on unseen data.")



#| Linear Regression         | Ridge                       | Lasso                            | ElasticNet                                                         |
#| ------------------------- | --------------------------- | -------------------------------- | ------------------------------------------------------------------ |
#| No regularization         | L2 regularization           | L1 regularization                | L1 + L2                                                            |
#| May overfit               | Reduces overfitting         | Reduces overfitting              | Reduces overfitting                                                |
#| Keeps all features        | Keeps all features          | Can remove features              | Can remove features                                                |
#| Coefficients can be large | Shrinks coefficients        | Shrinks and may set some to zero | Shrinks and may set some to zero                                   |
#| Good baseline model       | Good with multicollinearity | Good for feature selection       | Good when features are correlated and feature selection is desired |



# ==============================================================
# MACHINE LEARNING NOTES
# TOPIC: LINEAR REGRESSION, RIDGE, LASSO & ELASTICNET
# ==============================================================
#
# --------------------------------------------------------------
# 1. WHAT IS REGRESSION?
# --------------------------------------------------------------
#
# Regression is a Supervised Machine Learning technique used to
# predict continuous numerical values.
#
# Examples:
# ✔ House Price Prediction
# ✔ Salary Prediction
# ✔ Stock Price Prediction
# ✔ Temperature Prediction
# ✔ Sales Forecasting
#
# Regression predicts numbers.
#
# Classification predicts categories.
#
#
# --------------------------------------------------------------
# 2. WHAT IS LINEAR REGRESSION?
# --------------------------------------------------------------
#
# Linear Regression is the simplest regression algorithm.
#
# It learns the relationship between input features (X)
# and the target variable (y).
#
# The objective is to find the BEST FITTING STRAIGHT LINE.
#
# Single Feature Equation:
#
#       y = mx + c
#
# where
#
# y = prediction
# x = input feature
# m = slope
# c = intercept
#
#
# Multiple Linear Regression:
#
# y = w1*x1 + w2*x2 + w3*x3 + ... + wn*xn + b
#
# where
#
# w = coefficient (weight)
# x = feature
# b = intercept
#
#
# Example
#
# House Price =
# (Size × Weight1)
# + (Bedrooms × Weight2)
# + (Age × Weight3)
# + Intercept
#
#
# --------------------------------------------------------------
# 3. HOW DOES LINEAR REGRESSION LEARN?
# --------------------------------------------------------------
#
# During model.fit(X_train, y_train)
#
# the algorithm
#
# ✔ studies the training data
# ✔ finds relationships
# ✔ calculates coefficients
# ✔ minimizes prediction error
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
# 4. WHAT IS A COEFFICIENT?
# --------------------------------------------------------------
#
# Every feature receives one coefficient.
#
# Example
#
# Age = -3.5
# BMI = 25.6
# BP = 12.1
#
#
# Positive coefficient
#
# Feature increases
# Prediction increases
#
#
# Negative coefficient
#
# Feature increases
# Prediction decreases
#
#
# Larger coefficient
#
# Greater influence
#
#
# --------------------------------------------------------------
# 5. WHAT IS INTERCEPT?
# --------------------------------------------------------------
#
# Intercept is the starting point of the regression equation.
#
# It is the prediction when every feature is zero.
#
#
# --------------------------------------------------------------
# 6. HOW DOES PREDICTION HAPPEN?
# --------------------------------------------------------------
#
# Model simply substitutes feature values
# into the learned equation.
#
# Prediction =
#
# (Feature × Weight)
# +
# (Feature × Weight)
# +
# Intercept
#
#
# --------------------------------------------------------------
# 7. WHAT IS FEATURE SCALING?
# --------------------------------------------------------------
#
# Feature Scaling is the process of bringing every feature
# to a similar numerical scale.
#
# Example
#
# Age      = 25
# Salary   = 800000
#
# Salary values are much larger.
#
# Many ML algorithms may become biased toward
# features having larger numerical values.
#
# Scaling removes this issue.
#
#
# --------------------------------------------------------------
# 8. WHY FEATURE SCALING?
# --------------------------------------------------------------
#
# Without Scaling
#
# ✔ Large features dominate learning
# ✔ Optimization becomes slower
# ✔ Distance calculations become biased
#
#
# With Scaling
#
# ✔ Every feature contributes fairly
# ✔ Faster convergence
# ✔ Better optimization
#
#
# --------------------------------------------------------------
# 9. TYPES OF FEATURE SCALING
# --------------------------------------------------------------
#
# StandardScaler
#
# Mean becomes approximately 0
#
# Standard deviation becomes approximately 1
#
# Formula
#
# z = (x - mean) / standard deviation
#
#
# MinMaxScaler
#
# Converts values between
#
# 0 and 1
#
# Formula
#
# x = (x-min)/(max-min)
#
#
# --------------------------------------------------------------
# 10. IMPORTANT RULE OF SCALING
# --------------------------------------------------------------
#
# Never fit the scaler
# on the entire dataset.
#
# Correct Pipeline
#
# Split Data
#
# ↓
#
# Fit Scaler on Training Data
#
# ↓
#
# Transform Training Data
#
# ↓
#
# Transform Testing Data
#
#
# Never fit on testing data.
#
# Otherwise Data Leakage occurs.
#
#
# --------------------------------------------------------------
# 11. WHAT IS OVERFITTING?
# --------------------------------------------------------------
#
# Overfitting happens when
#
# the model memorizes
# training data
#
# instead of
#
# learning the actual pattern.
#
#
# Training Accuracy
#
# Very High
#
#
# Testing Accuracy
#
# Low
#
#
# Model performs poorly
# on unseen data.
#
#
# --------------------------------------------------------------
# 12. WHAT IS REGULARIZATION?
# --------------------------------------------------------------
#
# Regularization is a technique
# used to reduce overfitting.
#
# It discourages the model
# from learning unnecessarily
# large coefficients.
#
#
# Regularization improves
#
# Generalization
#
# rather than
#
# Training Accuracy.
#
#
# --------------------------------------------------------------
# 13. RIDGE REGRESSION
# --------------------------------------------------------------
#
# Ridge is Linear Regression
# with L2 Regularization.
#
# Idea
#
# Penalize large coefficients.
#
#
# Result
#
# ✔ Keeps every feature
#
# ✔ Shrinks coefficients
#
# ✔ Reduces overfitting
#
#
# Ridge DOES NOT remove features.
#
#
# Formula
#
# Cost Function
#
# Error
#
# +
#
# alpha × sum(weight²)
#
#
# Alpha
#
# controls
#
# Regularization Strength.
#
#
# Small Alpha
#
# Almost Linear Regression
#
#
# Large Alpha
#
# Strong Regularization
#
#
# --------------------------------------------------------------
# 14. LASSO REGRESSION
# --------------------------------------------------------------
#
# Lasso is Linear Regression
# with L1 Regularization.
#
#
# Lasso
#
# Shrinks coefficients
#
# AND
#
# can make coefficients
# exactly ZERO.
#
#
# Therefore
#
# Lasso performs
#
# Automatic Feature Selection.
#
#
# Formula
#
# Error
#
# +
#
# alpha × sum(|weight|)
#
#
# Best when
#
# many unnecessary
# features exist.
#
#
# --------------------------------------------------------------
# 15. ELASTICNET
# --------------------------------------------------------------
#
# ElasticNet combines
#
# Ridge
#
# and
#
# Lasso.
#
#
# It uses
#
# L1 + L2
#
#
# Advantages
#
# ✔ Shrinks coefficients
#
# ✔ Removes unnecessary features
#
# ✔ Handles correlated features better
#
#
# l1_ratio
#
# = 1
#
# Pure Lasso
#
#
# = 0
#
# Pure Ridge
#
#
# = 0.5
#
# Equal contribution
#
#
# --------------------------------------------------------------
# 16. WHAT IS ALPHA?
# --------------------------------------------------------------
#
# Alpha controls
#
# the amount of Regularization.
#
#
# Small Alpha
#
# Less penalty
#
#
# Large Alpha
#
# More penalty
#
#
# Alpha is a Hyperparameter.
#
#
# --------------------------------------------------------------
# 17. RIDGE vs LASSO vs ELASTICNET
# --------------------------------------------------------------
#
# Linear Regression
#
# ✔ No Regularization
#
# ✔ Can Overfit
#
#
# Ridge
#
# ✔ L2 Regularization
#
# ✔ Keeps all features
#
# ✔ Shrinks coefficients
#
#
# Lasso
#
# ✔ L1 Regularization
#
# ✔ Shrinks coefficients
#
# ✔ Removes features
#
#
# ElasticNet
#
# ✔ L1 + L2
#
# ✔ Balanced Regularization
#
# ✔ Feature Selection
#
#
# --------------------------------------------------------------
# 18. MODEL EVALUATION
# --------------------------------------------------------------
#
# MAE
#
# Mean Absolute Error
#
# Average prediction error.
#
# Lower is better.
#
#
# MSE
#
# Mean Squared Error
#
# Squares errors.
#
# Penalizes large mistakes.
#
# Lower is better.
#
#
# R² Score
#
# Explains how well
# the model fits the data.
#
#
# R² = 1
#
# Perfect Prediction
#
#
# R² = 0
#
# Same as predicting average
#
#
# R² < 0
#
# Worse than average prediction
#
#
# --------------------------------------------------------------
# 19. COMPLETE MACHINE LEARNING PIPELINE
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
# Train Model
#
# ↓
#
# Learn Coefficients
#
# ↓
#
# Predict
#
# ↓
#
# Evaluate
#
# ↓
#
# Improve using Regularization
#
#
# --------------------------------------------------------------
# 20. WHEN TO USE WHICH MODEL?
# --------------------------------------------------------------
#
# Linear Regression
#
# Use when
#
# ✔ Data has a simple linear relationship
# ✔ Overfitting is not a concern
#
#
# Ridge
#
# Use when
#
# ✔ Most features are useful
# ✔ Features are highly correlated
# ✔ Need better generalization
#
#
# Lasso
#
# Use when
#
# ✔ Many irrelevant features exist
# ✔ Feature selection is required
#
#
# ElasticNet
#
# Use when
#
# ✔ Dataset has many correlated features
# ✔ Some features should be removed
# ✔ Need the benefits of both Ridge and Lasso
#
#
# --------------------------------------------------------------
# 21. INTERVIEW QUESTIONS
# --------------------------------------------------------------
#
# Q1. What is Linear Regression?
#
# A supervised learning algorithm used for predicting
# continuous numerical values by learning the best linear
# relationship between features and the target.
#
#
# Q2. What is Regularization?
#
# A technique used to reduce overfitting by penalizing
# large model coefficients.
#
#
# Q3. Difference between Ridge and Lasso?
#
# Ridge shrinks coefficients but keeps every feature.
#
# Lasso shrinks coefficients and can remove
# unnecessary features by making coefficients zero.
#
#
# Q4. Why Feature Scaling?
#
# To bring all features to the same numerical scale,
# allowing scale-sensitive algorithms to learn fairly
# and converge faster.
#
#
# Q5. Why fit scaler only on training data?
#
# To prevent Data Leakage and ensure fair evaluation.
#
#
# --------------------------------------------------------------
# 22. FINAL TAKEAWAY
# --------------------------------------------------------------
#
# Linear Regression
# → Learns the best fitting line.
#
# Feature Scaling
# → Makes all features comparable.
#
# Overfitting
# → Model memorizes training data.
#
# Regularization
# → Prevents overfitting.
#
# Ridge
# → Shrinks coefficients (L2).
#
# Lasso
# → Shrinks + Removes features (L1).
#
# ElasticNet
# → Combines Ridge and Lasso (L1 + L2).
#
# The ultimate goal of every ML model is NOT to memorize
# the training data, but to GENERALIZE well and make
# accurate predictions on unseen data.
#
# ==============================================================