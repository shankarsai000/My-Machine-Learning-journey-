# =============================================================================
# DAY 31 : SUPPORT VECTOR MACHINE (SVM)
# PART 4.1 : LINEAR SVM
# =============================================================================

"""
Topics Covered

1. Load Dataset
2. Train-Test Split
3. Feature Scaling
4. Train Linear SVM
5. Predictions
6. Accuracy
7. Precision
8. Recall
9. F1 Score
10. ROC-AUC
11. Confusion Matrix
12. Classification Report
13. Support Vectors
14. New Sample Prediction
15. Top ML Engineer Insight

"""

# =============================================================================
# IMPORT LIBRARIES
# =============================================================================

import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC

from sklearn.metrics import (

    accuracy_score,

    precision_score,

    recall_score,

    f1_score,

    roc_auc_score,

    confusion_matrix,

    classification_report

)

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
# 3. FEATURE SCALING
# =============================================================================

print("\n")
print("="*70)
print("FEATURE SCALING")
print("="*70)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

print("Feature Scaling Completed!")

print("\nWhy Scaling?")

print("SVM is distance-based.")

print("Large feature values can dominate.")

print("Scaling gives every feature equal importance.")

# =============================================================================
# 4. TRAIN LINEAR SVM
# =============================================================================

print("\n")
print("="*70)
print("TRAINING LINEAR SVM")
print("="*70)

svm = SVC(

    kernel="linear",

    C=1.0,

    probability=True,

    random_state=42

)

svm.fit(

    X_train,

    y_train

)

print("Model Trained Successfully!")

# =============================================================================
# 5. PREDICTIONS
# =============================================================================

pred = svm.predict(

    X_test

)

prob = svm.predict_proba(

    X_test

)[:,1]

print("\nFirst Five Predictions")

print(pred[:5])

print("\nPrediction Probabilities")

print(np.round(prob[:5],4))

# =============================================================================
# 6. MODEL EVALUATION
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
# 7. CONFUSION MATRIX
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
# 8. CLASSIFICATION REPORT
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
# 9. SUPPORT VECTORS
# =============================================================================

print("\n")
print("="*70)
print("SUPPORT VECTORS")
print("="*70)

print("Number of Support Vectors")

print(svm.n_support_)

print("\nTotal Support Vectors")

print(sum(svm.n_support_))

# =============================================================================
# 10. MODEL INFORMATION
# =============================================================================

print("\n")
print("="*70)
print("MODEL INFORMATION")
print("="*70)

print("Kernel          :", svm.kernel)

print("Regularization C:", svm.C)

print("Support Vectors :", sum(svm.n_support_))

# =============================================================================
# 11. NEW SAMPLE PREDICTION
# =============================================================================

new_sample = X.iloc[[0]]

new_scaled = scaler.transform(

    new_sample

)

prediction = svm.predict(

    new_scaled

)

probability = svm.predict_proba(

    new_scaled

)

print("\n")
print("="*70)
print("NEW SAMPLE PREDICTION")
print("="*70)

print("Predicted Class :", prediction[0])

print("Prediction Probability")

print(np.round(probability,4))

# =============================================================================
# 12. MODEL SUMMARY
# =============================================================================

print("\n")
print("="*70)
print("MODEL SUMMARY")
print("="*70)

print("""

Algorithm

↓

Support Vector Machine

Kernel

↓

Linear

Learning Style

↓

Maximum Margin Classification

Decision Boundary

↓

Hyperplane

Important Points

↓

Support Vectors

Feature Scaling

↓

Required

""")

# =============================================================================
# 13. TOP ML ENGINEER INSIGHT
# =============================================================================

print("\n")
print("="*70)
print("TOP ML ENGINEER INSIGHT")
print("="*70)

print("""

Never train SVM
without feature scaling.

SVM relies on distance calculations.

Always tune

✔ Kernel

✔ C

✔ Gamma (for RBF)

Never rely only on Accuracy.

Also evaluate

✔ Precision

✔ Recall

✔ F1 Score

✔ ROC-AUC

Support Vectors define
the entire model.

The farther points usually
have little influence
on the decision boundary.

""")


# =============================================================================
# DAY 31 : SUPPORT VECTOR MACHINE (SVM)
# PART 4.2 : KERNEL COMPARISON
# =============================================================================

"""
Topics Covered

1. Linear Kernel
2. Polynomial Kernel
3. RBF Kernel
4. Sigmoid Kernel
5. Accuracy Comparison
6. Precision Comparison
7. Recall Comparison
8. F1 Score Comparison
9. ROC-AUC Comparison
10. Kernel Performance Summary
11. Best Kernel Selection
12. Top ML Engineer Insight

"""

# =============================================================================
# IMPORT LIBRARIES
# =============================================================================

from sklearn.svm import SVC

from sklearn.metrics import (

    accuracy_score,

    precision_score,

    recall_score,

    f1_score,

    roc_auc_score

)

import matplotlib.pyplot as plt

# =============================================================================
# DEFINE KERNELS
# =============================================================================

kernels = {

    "Linear": SVC(
        kernel="linear",
        probability=True,
        random_state=42
    ),

    "Polynomial": SVC(
        kernel="poly",
        degree=3,
        probability=True,
        random_state=42
    ),

    "RBF": SVC(
        kernel="rbf",
        probability=True,
        random_state=42
    ),

    "Sigmoid": SVC(
        kernel="sigmoid",
        probability=True,
        random_state=42
    )

}

# =============================================================================
# TRAIN & EVALUATE ALL KERNELS
# =============================================================================

results = []

for name, model in kernels.items():

    model.fit(
        X_train,
        y_train
    )

    pred = model.predict(X_test)

    prob = model.predict_proba(X_test)[:,1]

    results.append({

        "Kernel": name,

        "Accuracy":
            accuracy_score(y_test, pred),

        "Precision":
            precision_score(y_test, pred),

        "Recall":
            recall_score(y_test, pred),

        "F1 Score":
            f1_score(y_test, pred),

        "ROC-AUC":
            roc_auc_score(y_test, prob)

    })

# =============================================================================
# CREATE COMPARISON TABLE
# =============================================================================

comparison = pd.DataFrame(results)

comparison = comparison.round(4)

print("\n")
print("="*70)
print("KERNEL COMPARISON")
print("="*70)

print(comparison)

# =============================================================================
# BEST KERNEL
# =============================================================================

best = comparison.loc[
    comparison["Accuracy"].idxmax()
]

print("\n")
print("="*70)
print("BEST PERFORMING KERNEL")
print("="*70)

print("Kernel   :", best["Kernel"])

print("Accuracy :", best["Accuracy"])

# =============================================================================
# ACCURACY COMPARISON GRAPH
# =============================================================================

plt.figure(figsize=(8,5))

plt.bar(

    comparison["Kernel"],

    comparison["Accuracy"]

)

plt.title("Kernel Accuracy Comparison")

plt.ylabel("Accuracy")

plt.grid(True)

plt.show()

# =============================================================================
# PRECISION COMPARISON GRAPH
# =============================================================================

plt.figure(figsize=(8,5))

plt.bar(

    comparison["Kernel"],

    comparison["Precision"]

)

plt.title("Kernel Precision Comparison")

plt.ylabel("Precision")

plt.grid(True)

plt.show()

# =============================================================================
# RECALL COMPARISON GRAPH
# =============================================================================

plt.figure(figsize=(8,5))

plt.bar(

    comparison["Kernel"],

    comparison["Recall"]

)

plt.title("Kernel Recall Comparison")

plt.ylabel("Recall")

plt.grid(True)

plt.show()

# =============================================================================
# F1 SCORE COMPARISON GRAPH
# =============================================================================

plt.figure(figsize=(8,5))

plt.bar(

    comparison["Kernel"],

    comparison["F1 Score"]

)

plt.title("Kernel F1 Score Comparison")

plt.ylabel("F1 Score")

plt.grid(True)

plt.show()

# =============================================================================
# ROC-AUC COMPARISON GRAPH
# =============================================================================

plt.figure(figsize=(8,5))

plt.bar(

    comparison["Kernel"],

    comparison["ROC-AUC"]

)

plt.title("Kernel ROC-AUC Comparison")

plt.ylabel("ROC-AUC")

plt.grid(True)

plt.show()

# =============================================================================
# WHEN TO USE WHICH KERNEL
# =============================================================================

print("\n")
print("="*70)
print("WHEN TO USE WHICH KERNEL?")
print("="*70)

print("""

LINEAR

✔ Linearly separable data

✔ Text Classification

✔ High-dimensional datasets

--------------------------------------------------

POLYNOMIAL

✔ Curved relationships

✔ Engineering problems

✔ Scientific modelling

--------------------------------------------------

RBF

✔ Unknown relationships

✔ Complex decision boundaries

✔ Best default choice

--------------------------------------------------

SIGMOID

✔ Experimental

✔ Neural-network-inspired

✔ Rarely used in production

""")

# =============================================================================
# FEATURE SCALING REMINDER
# =============================================================================

print("\n")
print("="*70)
print("FEATURE SCALING")
print("="*70)

print("""

All kernels above were trained

AFTER

StandardScaler.

Without scaling,

distance calculations become biased.

Feature Scaling is mandatory
for almost every SVM model.

""")

# =============================================================================
# TOP ML ENGINEER INSIGHT
# =============================================================================

print("\n")
print("="*70)
print("TOP ML ENGINEER INSIGHT")
print("="*70)

print("""

Never assume

RBF

is always the best.

Always compare

✔ Linear

✔ Polynomial

✔ RBF

✔ Sigmoid

using

✔ Accuracy

✔ Precision

✔ Recall

✔ F1 Score

✔ ROC-AUC

The best kernel depends on

✔ Dataset size

✔ Data distribution

✔ Business objective

Experienced ML engineers

benchmark multiple kernels

before selecting the final model.

""")



# =============================================================================
# DAY 31 : SUPPORT VECTOR MACHINE (SVM)
# PART 4.3 : HYPERPARAMETER TUNING & MODEL COMPARISON
# =============================================================================

"""
Topics Covered

1. Effect of C
2. Effect of Gamma
3. Logistic Regression vs SVM
4. GridSearchCV
5. Best Parameters
6. Best Accuracy
7. Hyperparameter Summary
8. Top ML Engineer Insight

"""

# =============================================================================
# IMPORT LIBRARIES
# =============================================================================

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVC

from sklearn.metrics import accuracy_score

import pandas as pd

# =============================================================================
# 1. EFFECT OF C
# =============================================================================

print("\n")
print("="*70)
print("EFFECT OF PARAMETER C")
print("="*70)

C_values = [0.01, 0.1, 1, 10, 100]

c_results = []

for c in C_values:

    model = SVC(

        kernel="rbf",

        C=c,

        gamma="scale",

        probability=True,

        random_state=42

    )

    model.fit(

        X_train,

        y_train

    )

    pred = model.predict(

        X_test

    )

    c_results.append({

        "C": c,

        "Accuracy": round(

            accuracy_score(

                y_test,

                pred

            ),

            4

        )

    })

c_table = pd.DataFrame(c_results)

print(c_table)

# =============================================================================
# 2. EFFECT OF GAMMA
# =============================================================================

print("\n")
print("="*70)
print("EFFECT OF GAMMA")
print("="*70)

gamma_values = [

    0.001,

    0.01,

    0.1,

    1

]

gamma_results = []

for g in gamma_values:

    model = SVC(

        kernel="rbf",

        C=1,

        gamma=g,

        probability=True,

        random_state=42

    )

    model.fit(

        X_train,

        y_train

    )

    pred = model.predict(

        X_test

    )

    gamma_results.append({

        "Gamma": g,

        "Accuracy": round(

            accuracy_score(

                y_test,

                pred

            ),

            4

        )

    })

gamma_table = pd.DataFrame(

    gamma_results

)

print(gamma_table)

# =============================================================================
# 3. LOGISTIC REGRESSION VS SVM
# =============================================================================

print("\n")
print("="*70)
print("LOGISTIC REGRESSION VS SVM")
print("="*70)

log_model = LogisticRegression(

    max_iter=1000,

    random_state=42

)

log_model.fit(

    X_train,

    y_train

)

log_pred = log_model.predict(

    X_test

)

svm_model = SVC(

    kernel="rbf",

    probability=True,

    random_state=42

)

svm_model.fit(

    X_train,

    y_train

)

svm_pred = svm_model.predict(

    X_test

)

comparison = pd.DataFrame({

    "Model":[

        "Logistic Regression",

        "Support Vector Machine"

    ],

    "Accuracy":[

        round(

            accuracy_score(

                y_test,

                log_pred

            ),

            4

        ),

        round(

            accuracy_score(

                y_test,

                svm_pred

            ),

            4

        )

    ]

})

print(comparison)

# =============================================================================
# 4. GRID SEARCH CV
# =============================================================================

print("\n")
print("="*70)
print("GRID SEARCH CV")
print("="*70)

param_grid = {

    "C":[

        0.1,

        1,

        10

    ],

    "gamma":[

        0.01,

        0.1,

        1

    ],

    "kernel":[

        "linear",

        "rbf"

    ]

}

grid = GridSearchCV(

    estimator=SVC(

        probability=True,

        random_state=42

    ),

    param_grid=param_grid,

    cv=5,

    scoring="accuracy",

    n_jobs=-1

)

grid.fit(

    X_train,

    y_train

)

print("Best Parameters")

print(grid.best_params_)

print("\nBest Cross Validation Score")

print(round(

    grid.best_score_,

    4

))

# =============================================================================
# 5. BEST MODEL
# =============================================================================

best_model = grid.best_estimator_

best_pred = best_model.predict(

    X_test

)

best_accuracy = accuracy_score(

    y_test,

    best_pred

)

print("\n")
print("="*70)
print("BEST MODEL PERFORMANCE")
print("="*70)

print("Test Accuracy")

print(round(

    best_accuracy,

    4

))

# =============================================================================
# 6. HYPERPARAMETER SUMMARY
# =============================================================================

print("\n")
print("="*70)
print("HYPERPARAMETER SUMMARY")
print("="*70)

print("""

Parameter C

↓

Controls

Regularization

-------------------------------------------------

Large C

↓

Less Regularization

Fits Training Data More

Higher Overfitting Risk

-------------------------------------------------

Small C

↓

More Regularization

Allows Some Errors

Better Generalization

-------------------------------------------------

Gamma

↓

Controls

Influence of Each Point

-------------------------------------------------

Small Gamma

↓

Smooth Decision Boundary

Better Generalization

-------------------------------------------------

Large Gamma

↓

Complex Decision Boundary

Higher Overfitting Risk

""")

# =============================================================================
# 7. TOP ML ENGINEER INSIGHT
# =============================================================================

print("\n")
print("="*70)
print("TOP ML ENGINEER INSIGHT")
print("="*70)

print("""

Never tune

C

or

Gamma

by guessing.

Always use

✔ GridSearchCV

or

✔ RandomizedSearchCV

Evaluate using

✔ Cross Validation

instead of a

single Train-Test Split.

Production ML pipelines

never rely on

default hyperparameters.

Model tuning is often

the difference between

an average model

and a production-ready model.

""")

# =============================================================================
# DAY 31 : SUPPORT VECTOR MACHINE (SVM)
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

1. Train a Linear SVM.

2. Change

Kernel

Linear

Polynomial

RBF

Sigmoid

Compare Accuracy.

------------------------------------------------------------

Intermediate

3. Experiment with

C =

0.1

1

10

100

Observe

Training Accuracy

Testing Accuracy

Generalization

------------------------------------------------------------

4. Experiment with

Gamma =

0.001

0.01

0.1

1

Observe

Decision Boundary

Accuracy

Overfitting

------------------------------------------------------------

Advanced

5. Perform

GridSearchCV

for

Kernel

C

Gamma

------------------------------------------------------------

6. Compare

Logistic Regression

Decision Tree

Random Forest

SVM

using

Accuracy

Precision

Recall

F1 Score

ROC-AUC

""")

# =============================================================================
# 17. MINI CHALLENGE
# =============================================================================

print("\n")
print("="*70)
print("MINI CHALLENGE")
print("="*70)

print("""

Build a Cancer Detection System

Pipeline

✔ Data Cleaning

✔ Train-Test Split

✔ StandardScaler

✔ Linear SVM

✔ RBF SVM

✔ Kernel Comparison

✔ Hyperparameter Tuning

✔ ROC-AUC

✔ Confusion Matrix

✔ Classification Report

Bonus

Explain

WHY

one kernel performs
better than another.

""")

# =============================================================================
# 18. COMMON INTERVIEW QUESTIONS
# =============================================================================

print("\n")
print("="*70)
print("COMMON INTERVIEW QUESTIONS")
print("="*70)

questions = [

"What is SVM?",

"Why was SVM invented?",

"What is a Hyperplane?",

"What are Support Vectors?",

"What is Margin?",

"Why is SVM called the Maximum Margin Classifier?",

"What is Hard Margin SVM?",

"What is Soft Margin SVM?",

"What are Slack Variables?",

"What does parameter C control?",

"What happens when C is very large?",

"What happens when C is very small?",

"Why is feature scaling mandatory for SVM?",

"What is the Kernel Trick?",

"What is a Kernel Function?",

"Difference between Linear and RBF Kernel?",

"What does Gamma control?",

"What happens when Gamma is too large?",

"What happens when Gamma is too small?",

"When should you use Linear Kernel?",

"When should you use Polynomial Kernel?",

"When should you use RBF Kernel?",

"When should you use Sigmoid Kernel?",

"SVM vs Logistic Regression?",

"SVM vs Random Forest?",

"Advantages and limitations of SVM?",

"Why does SVM struggle with large datasets?"

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

Always Scale Features

↓

StandardScaler

↓

SVM

------------------------------------------------------------

Never tune only

C

Tune

✔ C

✔ Gamma

✔ Kernel

together.

------------------------------------------------------------

Large C

↓

Less Regularization

↓

Higher Overfitting Risk

------------------------------------------------------------

Small C

↓

More Regularization

↓

Better Generalization

------------------------------------------------------------

Large Gamma

↓

Very Complex Boundary

↓

High Variance

↓

Overfitting

------------------------------------------------------------

Small Gamma

↓

Smooth Boundary

↓

High Bias

↓

Underfitting

------------------------------------------------------------

Linear Kernel

↓

Fast

Simple

Large Datasets

------------------------------------------------------------

RBF Kernel

↓

Best Default

for Non-linear Data

------------------------------------------------------------

Polynomial Kernel

↓

Curved Relationships

------------------------------------------------------------

Sigmoid Kernel

↓

Rarely Used

------------------------------------------------------------

Support Vectors

are the ONLY training samples
that determine the decision boundary.

Removing distant points
usually changes nothing.

Removing Support Vectors
changes the model.

------------------------------------------------------------

Always use

Cross Validation

before deployment.

Never rely on a single
Train-Test Split.

""")

# =============================================================================
# 20. REAL-WORLD APPLICATIONS
# =============================================================================

print("\n")
print("="*70)
print("REAL-WORLD APPLICATIONS")
print("="*70)

applications = [

"Spam Detection",

"Email Classification",

"Face Recognition",

"Handwriting Recognition",

"Sentiment Analysis",

"Intrusion Detection",

"Gene Classification",

"Medical Diagnosis",

"Document Classification",

"Image Classification (Small Datasets)"

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

Support Vector Machine (SVM)

↓

Supervised Learning Algorithm

------------------------------------------------------------

Goal

↓

Find the Hyperplane

with Maximum Margin

------------------------------------------------------------

Core Concepts

✔ Hyperplane

✔ Support Vectors

✔ Margin

✔ Maximum Margin

✔ Hard Margin

✔ Soft Margin

✔ Slack Variables

✔ Kernel Trick

------------------------------------------------------------

Important Parameters

✔ C

✔ Gamma

✔ Kernel

------------------------------------------------------------

Common Kernels

✔ Linear

✔ Polynomial

✔ RBF (Gaussian)

✔ Sigmoid

------------------------------------------------------------

Feature Scaling

↓

Mandatory

(StandardScaler)

------------------------------------------------------------

Advantages

✔ Excellent for Small & Medium Datasets

✔ Strong Mathematical Foundation

✔ Handles High-dimensional Data

✔ Effective with Clear Margins

------------------------------------------------------------

Limitations

✔ Slow on Large Datasets

✔ Sensitive to Hyperparameters

✔ Requires Scaling

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

"SVM searches for the maximum-margin hyperplane.",

"Only support vectors determine the decision boundary.",

"Feature scaling is mandatory for SVM.",

"Soft Margin is preferred in real-world problems.",

"C controls the balance between margin and training errors.",

"Gamma controls the influence of individual data points.",

"Kernel Trick enables SVM to solve non-linear problems.",

"RBF is the most commonly used kernel.",

"GridSearchCV helps find optimal hyperparameters.",

"SVM performs best on small to medium-sized datasets.",

"Choose kernels based on data characteristics, not popularity."

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

✔ Hyperplane

✔ Support Vectors

✔ Margin

✔ Maximum Margin Classifier

✔ Hard Margin

✔ Soft Margin

✔ Slack Variables

✔ Parameter C

✔ Kernel Trick

✔ Linear Kernel

✔ Polynomial Kernel

✔ RBF Kernel

✔ Sigmoid Kernel

✔ Gamma

✔ GridSearchCV

✔ Production SVM Pipeline

Biggest Realization

SVM does NOT learn
from every data point.

It learns mainly from

Support Vectors.

These few critical points
completely define
the decision boundary.

That is what makes SVM
both elegant and mathematically powerful.

""")

# =============================================================================
# 24. END OF DAY 31
# =============================================================================

print("\n")
print("="*70)
print("END OF DAY 31")
print("="*70)

print("Next Topic : Naive Bayes - Learning Through Probability 🚀")


