# =============================================================================
# DAY 27 : FEATURE SCALING & BIAS-VARIANCE TRADEOFF
# PART 1 : FEATURE SCALING
# =============================================================================

"""
Topics Covered

1. Why Feature Scaling?
2. StandardScaler
3. MinMaxScaler
4. RobustScaler
5. Normalizer
6. Compare all scalers
7. Algorithms requiring scaling
8. Data Leakage demonstration

"""

# =============================================================================
# IMPORT LIBRARIES
# =============================================================================

import pandas as pd
import numpy as np

from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    Normalizer
)

from sklearn.model_selection import train_test_split

# =============================================================================
# 1. CREATE DATASET
# =============================================================================

print("="*70)
print("ORIGINAL DATASET")
print("="*70)

data = pd.DataFrame({

    "Age":[18,22,25,35,45],

    "Salary":[25000,40000,65000,120000,950000],

    "Experience":[1,2,4,10,20]

})

print(data)

# =============================================================================
# WHY FEATURE SCALING?
# =============================================================================

print("\n")
print("="*70)
print("WHY FEATURE SCALING?")
print("="*70)

print("""
Notice the feature ranges

Age         : 18 → 45

Salary      : 25,000 → 950,000

Experience  : 1 → 20

Salary has much larger numerical values.

Many Machine Learning algorithms use mathematical optimization
or distance calculations.

Without scaling,

Salary dominates learning simply because its values
are much larger.

Feature Scaling solves this problem.
""")

# =============================================================================
# 2. OBSERVE FEATURE RANGES
# =============================================================================

print("\n")
print("="*70)
print("FEATURE RANGES")
print("="*70)

print(data.describe())

# =============================================================================
# 3. STANDARD SCALER
# =============================================================================

print("\n")
print("="*70)
print("STANDARD SCALER")
print("="*70)

standard = StandardScaler()

standard_scaled = pd.DataFrame(

    standard.fit_transform(data),

    columns=data.columns

)

print(standard_scaled)

print("""

Properties

✔ Mean ≈ 0

✔ Standard Deviation ≈ 1

Best For

• Logistic Regression

• Linear Regression

• Ridge

• Lasso

• ElasticNet

• PCA

• SVM

• Neural Networks

""")

# =============================================================================
# 4. MINMAX SCALER
# =============================================================================

print("\n")
print("="*70)
print("MINMAX SCALER")
print("="*70)

minmax = MinMaxScaler()

minmax_scaled = pd.DataFrame(

    minmax.fit_transform(data),

    columns=data.columns

)

print(minmax_scaled)

print("""

Properties

Minimum becomes 0

Maximum becomes 1

Best For

• Neural Networks

• Image Processing

• Deep Learning

""")

# =============================================================================
# 5. ROBUST SCALER
# =============================================================================

print("\n")
print("="*70)
print("ROBUST SCALER")
print("="*70)

robust = RobustScaler()

robust_scaled = pd.DataFrame(

    robust.fit_transform(data),

    columns=data.columns

)

print(robust_scaled)

print("""

Uses

Median

and

Interquartile Range (IQR)

instead of

Mean

Standard Deviation

Best when the dataset contains OUTLIERS.

""")

# =============================================================================
# 6. NORMALIZER
# =============================================================================

print("\n")
print("="*70)
print("NORMALIZER")
print("="*70)

normalizer = Normalizer()

normalized = pd.DataFrame(

    normalizer.fit_transform(data),

    columns=data.columns

)

print(normalized)

print("""

Important

Normalizer DOES NOT scale columns.

It scales EACH ROW independently.

Each row gets unit length.

Mostly used in

• NLP

• Recommendation Systems

• Cosine Similarity

""")

# =============================================================================
# 7. COMPARE ALL SCALERS
# =============================================================================

print("\n")
print("="*70)
print("SCALER COMPARISON")
print("="*70)

print("""

StandardScaler

✔ Mean = 0

✔ Std = 1

✔ Best for Gradient Descent algorithms


--------------------------------------------

MinMaxScaler

✔ Values between 0 and 1

✔ Preserves relative relationships


--------------------------------------------

RobustScaler

✔ Resistant to Outliers

✔ Uses Median + IQR


--------------------------------------------

Normalizer

✔ Scales each ROW

✔ Useful for Cosine Similarity

""")

# =============================================================================
# 8. WHICH ALGORITHMS REQUIRE SCALING?
# =============================================================================

print("\n")
print("="*70)
print("ALGORITHMS THAT REQUIRE SCALING")
print("="*70)

needs_scaling = [

    "Linear Regression",

    "Logistic Regression",

    "Ridge Regression",

    "Lasso Regression",

    "ElasticNet",

    "K-Nearest Neighbors (KNN)",

    "Support Vector Machine (SVM)",

    "K-Means Clustering",

    "Principal Component Analysis (PCA)",

    "Neural Networks"

]

for algorithm in needs_scaling:

    print("✔",algorithm)

print("\n")

print("="*70)
print("ALGORITHMS THAT DO NOT REQUIRE SCALING")
print("="*70)

no_scaling = [

    "Decision Tree",

    "Random Forest",

    "Extra Trees",

    "XGBoost",

    "LightGBM",

    "CatBoost"

]

for algorithm in no_scaling:

    print("✔",algorithm)

# =============================================================================
# WHY?
# =============================================================================

print("""

Tree-based algorithms split data using thresholds.

Example

Salary > 50000

or

Age < 30

They never calculate distances.

Therefore,

Feature Scaling is generally unnecessary.

""")

# =============================================================================
# 9. DATA LEAKAGE DEMONSTRATION
# =============================================================================

print("\n")
print("="*70)
print("DATA LEAKAGE")
print("="*70)

X = data.copy()

X_train, X_test = train_test_split(

    X,

    test_size=0.2,

    random_state=42

)

print("""

❌ WRONG PIPELINE

Scale Entire Dataset

↓

Train-Test Split

The scaler has already seen the testing data.

This leaks information into training.


--------------------------------------------


✅ CORRECT PIPELINE

Train-Test Split

↓

Fit Scaler ONLY on Training Data

↓

Transform Training Data

↓

Transform Testing Data

""")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

print("Training Shape :",X_train_scaled.shape)

print("Testing Shape  :",X_test_scaled.shape)

# =============================================================================
# TOP ML ENGINEER INSIGHT
# =============================================================================

print("\n")
print("="*70)
print("TOP ML ENGINEER INSIGHT")
print("="*70)

print("""

Feature Scaling is NOT performed to improve the data.

It is performed to help optimization algorithms
learn efficiently.

Always remember:

Scaling changes

the REPRESENTATION

NOT

the INFORMATION.

Never fit a scaler on the entire dataset.

That introduces DATA LEAKAGE and leads to
overly optimistic evaluation results.

""")

# =============================================================================
# DAY 27 : FEATURE SCALING & BIAS-VARIANCE TRADEOFF
# PART 2 : GENERALIZATION, BIAS & VARIANCE
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# =============================================================================
# 10. CREATE A NON-LINEAR DATASET
# =============================================================================

np.random.seed(42)

X = np.linspace(0,10,100)

noise = np.random.normal(0,0.7,100)

y = np.sin(X) + noise

X = X.reshape(-1,1)

X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42
)

print("="*70)
print("BIAS-VARIANCE DEMONSTRATION DATASET")
print("="*70)

print("Training Samples :",len(X_train))
print("Testing Samples  :",len(X_test))

# =============================================================================
# 11. UNDERFITTING
# =============================================================================

print("\n")
print("="*70)
print("UNDERFITTING")
print("="*70)

under_model = Pipeline([
    ("poly",PolynomialFeatures(degree=1)),
    ("linear",LinearRegression())
])

under_model.fit(X_train,y_train)

train_pred = under_model.predict(X_train)
test_pred = under_model.predict(X_test)

print("Training MSE :",round(mean_squared_error(y_train,train_pred),3))
print("Testing MSE  :",round(mean_squared_error(y_test,test_pred),3))

print("""

UNDERFITTING

Characteristics

✔ High Bias

✔ Low Variance

Model is TOO SIMPLE.

Cannot learn the real relationship.

Symptoms

• High Training Error

• High Testing Error

Solutions

✔ Increase model complexity

✔ Add better features

✔ Reduce excessive regularization

""")

# =============================================================================
# 12. GOOD FIT
# =============================================================================

print("\n")
print("="*70)
print("GOOD FIT")
print("="*70)

good_model = Pipeline([
    ("poly",PolynomialFeatures(degree=5)),
    ("linear",LinearRegression())
])

good_model.fit(X_train,y_train)

train_pred = good_model.predict(X_train)
test_pred = good_model.predict(X_test)

print("Training MSE :",round(mean_squared_error(y_train,train_pred),3))
print("Testing MSE  :",round(mean_squared_error(y_test,test_pred),3))

print("""

GOOD FIT

Balanced Bias

Balanced Variance

Model captures the true pattern

without memorizing noise.

Goal of Machine Learning

GENERALIZATION

""")

# =============================================================================
# 13. OVERFITTING
# =============================================================================

print("\n")
print("="*70)
print("OVERFITTING")
print("="*70)

over_model = Pipeline([
    ("poly",PolynomialFeatures(degree=20)),
    ("linear",LinearRegression())
])

over_model.fit(X_train,y_train)

train_pred = over_model.predict(X_train)
test_pred = over_model.predict(X_test)

print("Training MSE :",round(mean_squared_error(y_train,train_pred),3))
print("Testing MSE  :",round(mean_squared_error(y_test,test_pred),3))

print("""

OVERFITTING

Characteristics

✔ Low Bias

✔ High Variance

Model memorizes

training data

including noise.

Symptoms

Very Low Training Error

High Testing Error

Solutions

✔ More Data

✔ Regularization

✔ Simpler Model

✔ Feature Selection

✔ Cross Validation

""")

# =============================================================================
# 14. VISUAL COMPARISON
# =============================================================================

x_plot = np.linspace(0,10,300).reshape(-1,1)

models = [
    ("Underfitting",under_model),
    ("Good Fit",good_model),
    ("Overfitting",over_model)
]

for title,model in models:

    plt.figure(figsize=(6,4))

    plt.scatter(X,y,label="Data")

    plt.plot(
        x_plot,
        model.predict(x_plot),
        linewidth=2,
        label="Prediction"
    )

    plt.title(title)

    plt.xlabel("X")

    plt.ylabel("y")

    plt.legend()

    plt.show()

# =============================================================================
# 15. TRAINING vs TESTING ERROR
# =============================================================================

print("\n")
print("="*70)
print("TRAINING vs TESTING")
print("="*70)

print("""

Underfitting

Training Error

HIGH

Testing Error

HIGH

------------------------------

Good Fit

Training Error

LOW

Testing Error

LOW

------------------------------

Overfitting

Training Error

VERY LOW

Testing Error

HIGH

""")

# =============================================================================
# 16. WHAT IS BIAS?
# =============================================================================

print("\n")
print("="*70)
print("BIAS")
print("="*70)

print("""

Bias

=

Error caused

because

the model

is TOO SIMPLE.

Example

Trying to fit

a straight line

through

curved data.

High Bias

↓

Underfitting

""")

# =============================================================================
# 17. WHAT IS VARIANCE?
# =============================================================================

print("\n")
print("="*70)
print("VARIANCE")
print("="*70)

print("""

Variance

=

Error caused

because

the model

is TOO SENSITIVE

to training data.

It memorizes

noise

instead of

learning patterns.

High Variance

↓

Overfitting

""")

# =============================================================================
# 18. MODEL COMPLEXITY
# =============================================================================

print("\n")
print("="*70)
print("MODEL COMPLEXITY")
print("="*70)

complexity = [
    "Very Simple Model",
    "Moderately Complex Model",
    "Very Complex Model"
]

result = [
    "Underfitting",
    "Good Generalization",
    "Overfitting"
]

for c,r in zip(complexity,result):

    print(c," ---> ",r)

# =============================================================================
# 19. HOW REGULARIZATION HELPS
# =============================================================================

print("\n")
print("="*70)
print("REGULARIZATION")
print("="*70)

print("""

Ridge

↓

Shrinks coefficients

-----------------------------

Lasso

↓

Shrinks

+

Removes Features

-----------------------------

ElasticNet

↓

Combination

of

L1 + L2

Regularization

reduces

OVERFITTING

and improves

GENERALIZATION.

""")

# =============================================================================
# 20. DOES MORE DATA HELP?
# =============================================================================

print("\n")
print("="*70)
print("MORE DATA")
print("="*70)

print("""

More Data

usually

reduces

VARIANCE.

It helps

the model

generalize

better.

BUT

More Data

does NOT

automatically fix

High Bias.

If your model

is too simple,

collecting more data

will not solve

the problem.

""")

# =============================================================================
# 21. SENIOR ML ENGINEER THINKING
# =============================================================================

print("\n")
print("="*70)
print("SENIOR ML ENGINEER QUESTIONS")
print("="*70)

questions = [

"Is my model underfitting?",

"Is my model overfitting?",

"Is the problem caused by High Bias?",

"Is the problem caused by High Variance?",

"Do I have Data Leakage?",

"Would Regularization help?",

"Do I need more data?",

"Will this model generalize?"

]

for q in questions:

    print("✔",q)

# =============================================================================
# 22. GOLDEN RULE
# =============================================================================

print("\n")
print("="*70)
print("GOLDEN RULE")
print("="*70)

print("""

Machine Learning

is NOT about

maximizing

Training Accuracy.

It is about

GENERALIZATION.

The best model

is NOT

the one that

memorizes

training data.

The best model

is the one

that performs well

on data

it has NEVER seen.

""")

# =============================================================================
# 23. FINAL TAKEAWAYS
# =============================================================================

print("\n")
print("="*70)
print("TODAY'S LEARNING")
print("="*70)

takeaways = [

"Scaling improves optimization.",

"Some algorithms require scaling.",

"Tree-based models usually do not require scaling.",

"High Bias causes Underfitting.",

"High Variance causes Overfitting.",

"Regularization reduces Overfitting.",

"More data mainly reduces Variance.",

"Generalization is the ultimate goal of Machine Learning."

]

for t in takeaways:

    print("✔",t)






# =============================================================================
# DAY 27 : FEATURE SCALING & BIAS-VARIANCE TRADEOFF
# =============================================================================
#
# WHY THIS NOTEBOOK?
# -----------------------------------------------------------------------------
#
# A Machine Learning model should not only perform well on the training data,
# but also on completely unseen data.
#
# This ability is called:
#
#                         GENERALIZATION
#
# Generalization is one of the most important goals in Machine Learning.
#
# Two major concepts help us achieve better generalization:
#
# 1. Feature Scaling
# 2. Bias-Variance Tradeoff
#
# Feature Scaling helps optimization algorithms learn efficiently.
#
# Bias-Variance Tradeoff helps us understand WHY a model succeeds
# or fails on unseen data.
#
#
# =============================================================================
# PART 1 : FEATURE SCALING
# =============================================================================
#
# WHAT IS FEATURE SCALING?
# -----------------------------------------------------------------------------
#
# Feature Scaling is the process of transforming numerical features
# so they exist on approximately the same scale.
#
#
# Example
#
# Age          = 25
#
# Salary       = 800000
#
# Experience   = 2
#
#
# Salary has much larger values than Age or Experience.
#
# Many Machine Learning algorithms interpret larger numerical values
# as having greater importance.
#
# This creates biased learning.
#
#
# Feature Scaling removes this problem by bringing every feature
# into a comparable numerical range.
#
#
# =============================================================================
# WHY FEATURE SCALING IS IMPORTANT
# =============================================================================
#
# Imagine two students pulling a rope.
#
# Student A
#
# Strength = 10
#
#
# Student B
#
# Strength = 1000
#
#
# Student B completely dominates.
#
#
# Features behave similarly.
#
#
# Large numerical values dominate optimization.
#
#
# Scaling ensures every feature contributes fairly.
#
#
# =============================================================================
# WHAT HAPPENS WITHOUT SCALING?
# =============================================================================
#
# Problems
#
# ✔ Slow Gradient Descent
#
# ✔ Poor Optimization
#
# ✔ Distance Calculations become biased
#
# ✔ Some models perform poorly
#
#
# =============================================================================
# WHAT HAPPENS AFTER SCALING?
# =============================================================================
#
# Benefits
#
# ✔ Faster convergence
#
# ✔ Stable optimization
#
# ✔ Fair contribution from all features
#
# ✔ Better numerical stability
#
# ✔ Improved model performance
#
#
# =============================================================================
# TYPES OF FEATURE SCALING
# =============================================================================
#
# 1. StandardScaler
#
# Formula
#
# z = (x - mean) / standard deviation
#
#
# Result
#
# Mean ≈ 0
#
# Standard Deviation ≈ 1
#
#
# Best For
#
# ✔ Logistic Regression
#
# ✔ Linear Regression
#
# ✔ Ridge
#
# ✔ Lasso
#
# ✔ ElasticNet
#
# ✔ SVM
#
# ✔ PCA
#
# ✔ Neural Networks
#
#
# =============================================================================
#
# 2. MinMaxScaler
#
# Formula
#
# x = (x-min)/(max-min)
#
#
# Output Range
#
# 0 → 1
#
#
# Best For
#
# ✔ Deep Learning
#
# ✔ Image Processing
#
# ✔ Neural Networks
#
#
# Advantage
#
# Preserves relative relationships
# between values.
#
#
# =============================================================================
#
# 3. RobustScaler
#
# Uses
#
# Median
#
# and
#
# Interquartile Range (IQR)
#
#
# Instead of
#
# Mean
#
# Standard Deviation
#
#
# Best For
#
# Datasets containing
#
# many OUTLIERS.
#
#
# Example
#
# Salaries
#
# £25,000
#
# £30,000
#
# £28,000
#
# £35,000
#
# £10,000,000
#
#
# StandardScaler becomes distorted.
#
#
# RobustScaler remains stable.
#
#
# =============================================================================
#
# 4. Normalizer
#
# Different from every other scaler.
#
# StandardScaler
#
# scales FEATURES.
#
#
# Normalizer
#
# scales EACH ROW.
#
#
# Every sample
#
# gets unit length.
#
#
# Mostly used in
#
# ✔ NLP
#
# ✔ Recommendation Systems
#
# ✔ Cosine Similarity
#
#
# =============================================================================
# WHICH ALGORITHMS REQUIRE SCALING?
# =============================================================================
#
# REQUIRED
#
# ✔ Logistic Regression
#
# ✔ Linear Regression
#
# ✔ Ridge
#
# ✔ Lasso
#
# ✔ ElasticNet
#
# ✔ KNN
#
# ✔ K-Means
#
# ✔ SVM
#
# ✔ PCA
#
# ✔ Neural Networks
#
#
# NOT REQUIRED
#
# ✔ Decision Tree
#
# ✔ Random Forest
#
# ✔ XGBoost
#
# ✔ LightGBM
#
# ✔ CatBoost
#
#
# Why?
#
# Tree algorithms split using
# feature thresholds.
#
# They do NOT depend
# on feature magnitude.
#
#
# =============================================================================
# DATA LEAKAGE
# =============================================================================
#
# One of the most common beginner mistakes.
#
#
# WRONG
#
# Scale entire dataset
#
# ↓
#
# Train-Test Split
#
#
# Why?
#
# The scaler learns information
# from the testing data.
#
#
# The model indirectly
# "sees the future."
#
#
# This produces
#
# Unrealistically high performance.
#
#
# CORRECT
#
# Train-Test Split
#
# ↓
#
# Fit Scaler ONLY on Training Data
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
# Never fit
#
# on testing data.
#
#
# =============================================================================
# PART 2 : BIAS-VARIANCE TRADEOFF
# =============================================================================
#
# WHAT IS GENERALIZATION?
# -----------------------------------------------------------------------------
#
# Generalization means
#
# the model performs well
#
# on NEW,
#
# unseen,
#
# real-world data.
#
#
# This is the ultimate goal
# of Machine Learning.
#
#
# =============================================================================
# MODEL COMPLEXITY
# =============================================================================
#
# Every Machine Learning model
# has a certain complexity.
#
#
# Too Simple
#
# Cannot learn patterns.
#
#
# Too Complex
#
# Memorizes everything.
#
#
# Ideal
#
# Learns patterns
#
# without memorizing.
#
#
# =============================================================================
# UNDERFITTING
# =============================================================================
#
# Underfitting happens when
#
# the model is
#
# TOO SIMPLE.
#
#
# Characteristics
#
# High Bias
#
# Low Variance
#
#
# Training Accuracy
#
# Low
#
#
# Testing Accuracy
#
# Low
#
#
# Causes
#
# ✔ Simple model
#
# ✔ Too few features
#
# ✔ Too much regularization
#
#
# Solutions
#
# ✔ Increase complexity
#
# ✔ Add useful features
#
# ✔ Reduce regularization
#
#
# =============================================================================
# OVERFITTING
# =============================================================================
#
# Overfitting happens when
#
# the model memorizes
# the training data.
#
#
# Characteristics
#
# Low Bias
#
# High Variance
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
# Causes
#
# ✔ Model too complex
#
# ✔ Small dataset
#
# ✔ Too many features
#
#
# Solutions
#
# ✔ More training data
#
# ✔ Regularization
#
# ✔ Simpler model
#
# ✔ Feature Selection
#
# ✔ Cross Validation
#
#
# =============================================================================
# BIAS
# =============================================================================
#
# Bias is the error caused
# because the model
#
# is TOO SIMPLE.
#
#
# High Bias
#
# means
#
# the model cannot learn
# the true relationship.
#
#
# Think of
#
# drawing a straight line
#
# through curved data.
#
#
# =============================================================================
# VARIANCE
# =============================================================================
#
# Variance is the error caused
#
# because the model
#
# is TOO COMPLEX.
#
#
# High Variance
#
# means
#
# the model memorizes
# every detail,
#
# including noise.
#
#
# It performs well
#
# only on training data.
#
#
# =============================================================================
# BIAS-VARIANCE TRADEOFF
# =============================================================================
#
# Every ML model tries
#
# to balance
#
# Bias
#
# and
#
# Variance.
#
#
# Too much Bias
#
# ↓
#
# Underfitting
#
#
# Too much Variance
#
# ↓
#
# Overfitting
#
#
# Balanced Bias & Variance
#
# ↓
#
# Good Generalization
#
#
# =============================================================================
# LEARNING CURVES
# =============================================================================
#
# As Model Complexity increases:
#
# Training Error
#
# decreases.
#
#
# Validation Error
#
# first decreases
#
# then increases.
#
#
# The lowest Validation Error
#
# represents
#
# the BEST model.
#
#
# =============================================================================
# HOW REGULARIZATION HELPS
# =============================================================================
#
# Ridge
#
# Lasso
#
# ElasticNet
#
# all reduce
#
# OVERFITTING.
#
#
# They prevent
#
# extremely large coefficients.
#
#
# Result
#
# Better Generalization.
#
#
# =============================================================================
# DOES MORE DATA HELP?
# =============================================================================
#
# More Data
#
# usually reduces
#
# VARIANCE.
#
#
# It helps the model
#
# generalize better.
#
#
# More Data
#
# DOES NOT automatically
#
# fix High Bias.
#
#
# If the model is too simple,
#
# collecting more data
#
# will not solve
#
# the problem.
#
#
# =============================================================================
# SENIOR ML ENGINEER MINDSET
# =============================================================================
#
# Beginners ask:
#
# "Which algorithm should I use?"
#
#
# Experienced ML Engineers ask:
#
# "Why is my model failing?"
#
#
# Then diagnose:
#
# ✔ High Bias?
#
# ✔ High Variance?
#
# ✔ Data Leakage?
#
# ✔ Wrong Evaluation Metric?
#
# ✔ Need More Data?
#
# ✔ Need Regularization?
#
#
# =============================================================================
# GOLDEN RULE
# =============================================================================
#
# Machine Learning is NOT about
#
# maximizing Training Accuracy.
#
#
# It is about
#
# maximizing
#
# GENERALIZATION.
#
#
# The best model is NOT
#
# the one that memorizes.
#
#
# The best model is
#
# the one that performs well
#
# on data
#
# it has NEVER seen before.
#
#
# =============================================================================
# KEY TAKEAWAYS
# =============================================================================
#
# ✔ Scaling improves optimization.
#
# ✔ Some algorithms require scaling.
#
# ✔ Tree-based models generally do not.
#
# ✔ Data leakage can invalidate results.
#
# ✔ High Bias → Underfitting.
#
# ✔ High Variance → Overfitting.
#
# ✔ Regularization reduces overfitting.
#
# ✔ More data mostly reduces variance.
#
# ✔ Generalization is the ultimate goal of ML.
#
# =============================================================================