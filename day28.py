# =============================================================================
# DAY 28 : DECISION TREE CLASSIFIER
# PART 4.1 : TRAINING A DECISION TREE
# =============================================================================

"""
Topics Covered

1. Load Breast Cancer Dataset
2. Explore Dataset
3. Train-Test Split
4. Train Decision Tree
5. Predictions
6. Accuracy
7. Confusion Matrix
8. Classification Report

"""

# =============================================================================
# IMPORT LIBRARIES
# =============================================================================

import pandas as pd
import numpy as np

from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

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

print("\nDataset Loaded Successfully!")

print("\nDataset Shape")

print(X.shape)

print("\nTarget Classes")

print(data.target_names)

# =============================================================================
# 2. EXPLORE DATASET
# =============================================================================

print("\n")
print("=" * 70)
print("DATASET INFORMATION")
print("=" * 70)

print("\nFirst Five Rows")

print(X.head())

print("\nFeature Names")

print(list(X.columns))

print("\nMissing Values")

print(X.isnull().sum().sum())

print("\nTarget Distribution")

print(pd.Series(y).value_counts())

print("\nTarget Meaning")

print("0 -> Malignant")

print("1 -> Benign")

# =============================================================================
# WHY DECISION TREES DO NOT REQUIRE SCALING
# =============================================================================

print("\n")
print("=" * 70)
print("WHY FEATURE SCALING IS NOT REQUIRED")
print("=" * 70)

print("""

Decision Trees split data using conditions like

Feature > Threshold

Example

Radius > 15.5

Area < 700

Scaling changes the numbers

but NOT their ordering.

Therefore,

Decision Trees generally do NOT require

StandardScaler

MinMaxScaler

RobustScaler

""")

# =============================================================================
# 3. TRAIN TEST SPLIT
# =============================================================================

print("\n")
print("=" * 70)
print("TRAIN TEST SPLIT")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42,

    stratify=y

)

print("Training Samples :", len(X_train))

print("Testing Samples  :", len(X_test))

# =============================================================================
# WHY STRATIFY?
# =============================================================================

print("""

Stratify maintains

the same class distribution

in both

Training

and

Testing

datasets.

This is a good practice

for classification problems.

""")

# =============================================================================
# 4. TRAIN DECISION TREE
# =============================================================================

print("\n")
print("=" * 70)
print("TRAINING DECISION TREE")
print("=" * 70)

model = DecisionTreeClassifier(

    criterion="gini",

    random_state=42

)

model.fit(

    X_train,

    y_train

)

print("Decision Tree Trained Successfully!")

# =============================================================================
# MODEL INFORMATION
# =============================================================================

print("\nTree Depth")

print(model.get_depth())

print("\nNumber of Leaf Nodes")

print(model.get_n_leaves())

# =============================================================================
# 5. MAKE PREDICTIONS
# =============================================================================

print("\n")
print("=" * 70)
print("PREDICTIONS")
print("=" * 70)

predictions = model.predict(X_test)

probabilities = model.predict_proba(X_test)

print("\nFirst 10 Predictions")

print(predictions[:10])

print("\nPrediction Probabilities")

print(probabilities[:5])

# =============================================================================
# 6. MODEL ACCURACY
# =============================================================================

accuracy = accuracy_score(

    y_test,

    predictions

)

print("\n")
print("=" * 70)
print("MODEL ACCURACY")
print("=" * 70)

print("Accuracy :", round(accuracy, 4))

# =============================================================================
# 7. CONFUSION MATRIX
# =============================================================================

cm = confusion_matrix(

    y_test,

    predictions

)

print("\n")
print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print(cm)

print("""

Interpretation

Top Left

True Negatives

Top Right

False Positives

Bottom Left

False Negatives

Bottom Right

True Positives

""")

# =============================================================================
# 8. CLASSIFICATION REPORT
# =============================================================================

print("\n")
print("=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(

    classification_report(

        y_test,

        predictions,

        target_names=data.target_names

    )

)

# =============================================================================
# PERFORMANCE SUMMARY
# =============================================================================

print("\n")
print("=" * 70)
print("MODEL SUMMARY")
print("=" * 70)

print("""

Algorithm

Decision Tree Classifier

Criterion

Gini Impurity

Task

Binary Classification

Feature Scaling

Not Required

Output

Class Prediction

Evaluation

Accuracy

Confusion Matrix

Classification Report

""")

# =============================================================================
# TOP ML ENGINEER INSIGHT
# =============================================================================

print("\n")
print("=" * 70)
print("TOP ML ENGINEER INSIGHT")
print("=" * 70)

print("""

Never judge a Decision Tree

only by Accuracy.

Also inspect

• Tree Depth

• Number of Leaves

• Overfitting

• Feature Importance

• Pruning

A tree with

100% Training Accuracy

may simply be

memorizing the training data.

""")
# =============================================================================
# DAY 28 : DECISION TREE CLASSIFIER
# PART 4.2 : VISUALIZATION & HYPERPARAMETER ANALYSIS
# =============================================================================

"""
Topics Covered

1. Decision Tree Visualization
2. Feature Importance
3. Effect of max_depth
4. Gini vs Entropy
5. Effect of min_samples_split
6. Accuracy Comparison Graphs

"""

import matplotlib.pyplot as plt

from sklearn.tree import plot_tree

# =============================================================================
# 9. DECISION TREE VISUALIZATION
# =============================================================================

print("\n")
print("="*70)
print("DECISION TREE VISUALIZATION")
print("="*70)

plt.figure(figsize=(22,12))

plot_tree(

    model,

    feature_names=X.columns,

    class_names=data.target_names,

    filled=True,

    rounded=True,

    fontsize=8

)

plt.title("Decision Tree (Gini Criterion)")

plt.show()

print("""

Visualization Guide

Each box represents a Node.

gini

↓

Impurity of that node.

samples

↓

Number of training samples.

value

↓

Samples belonging to each class.

class

↓

Predicted class.

The darker the colour,

the purer the node.

""")

# =============================================================================
# 10. FEATURE IMPORTANCE
# =============================================================================

print("\n")
print("="*70)
print("FEATURE IMPORTANCE")
print("="*70)

importance = pd.DataFrame({

    "Feature":X.columns,

    "Importance":model.feature_importances_

})

importance = importance.sort_values(

    by="Importance",

    ascending=False

)

print(importance.head(10))

plt.figure(figsize=(10,6))

plt.barh(

    importance["Feature"][:10],

    importance["Importance"][:10]

)

plt.gca().invert_yaxis()

plt.title("Top 10 Important Features")

plt.xlabel("Importance")

plt.show()

print("""

Feature Importance tells us

how useful each feature was

while building the Decision Tree.

Higher Importance

↓

Greater contribution

towards prediction.

""")

# =============================================================================
# 11. EFFECT OF MAX_DEPTH
# =============================================================================

print("\n")
print("="*70)
print("EFFECT OF max_depth")
print("="*70)

depths = [1,2,3,4,5,6,8,10,None]

train_scores = []

test_scores = []

for depth in depths:

    clf = DecisionTreeClassifier(

        max_depth=depth,

        random_state=42

    )

    clf.fit(

        X_train,

        y_train

    )

    train_scores.append(

        clf.score(X_train,y_train)

    )

    test_scores.append(

        clf.score(X_test,y_test)

    )

results = pd.DataFrame({

    "Max Depth":depths,

    "Training Accuracy":train_scores,

    "Testing Accuracy":test_scores

})

print(results)

plt.figure(figsize=(8,5))

plt.plot(

    range(len(depths)),

    train_scores,

    marker="o",

    label="Training"

)

plt.plot(

    range(len(depths)),

    test_scores,

    marker="s",

    label="Testing"

)

plt.xticks(

    range(len(depths)),

    [str(d) for d in depths]

)

plt.xlabel("max_depth")

plt.ylabel("Accuracy")

plt.title("Effect of Tree Depth")

plt.legend()

plt.grid(True)

plt.show()

print("""

Observation

Small Depth

↓

High Bias

↓

Underfitting

Large Depth

↓

High Variance

↓

Overfitting

Best depth

usually lies

between these extremes.

""")

# =============================================================================
# 12. GINI vs ENTROPY
# =============================================================================

print("\n")
print("="*70)
print("GINI vs ENTROPY")
print("="*70)

criteria = ["gini","entropy"]

comparison = []

for c in criteria:

    clf = DecisionTreeClassifier(

        criterion=c,

        random_state=42

    )

    clf.fit(

        X_train,

        y_train

    )

    comparison.append([

        c,

        clf.score(X_train,y_train),

        clf.score(X_test,y_test)

    ])

comparison = pd.DataFrame(

    comparison,

    columns=[

        "Criterion",

        "Training Accuracy",

        "Testing Accuracy"

    ]

)

print(comparison)

print("""

Gini

✔ Faster

✔ Default in Scikit-Learn

Entropy

✔ Information Theory

✔ Slightly slower

Performance

is usually

very similar.

""")

# =============================================================================
# 13. EFFECT OF min_samples_split
# =============================================================================

print("\n")
print("="*70)
print("EFFECT OF min_samples_split")
print("="*70)

splits = [2,5,10,20,30]

train_acc = []

test_acc = []

for s in splits:

    clf = DecisionTreeClassifier(

        min_samples_split=s,

        random_state=42

    )

    clf.fit(

        X_train,

        y_train

    )

    train_acc.append(

        clf.score(X_train,y_train)

    )

    test_acc.append(

        clf.score(X_test,y_test)

    )

split_results = pd.DataFrame({

    "min_samples_split":splits,

    "Training Accuracy":train_acc,

    "Testing Accuracy":test_acc

})

print(split_results)

plt.figure(figsize=(8,5))

plt.plot(

    splits,

    train_acc,

    marker="o",

    label="Training"

)

plt.plot(

    splits,

    test_acc,

    marker="s",

    label="Testing"

)

plt.xlabel("min_samples_split")

plt.ylabel("Accuracy")

plt.title("Effect of min_samples_split")

plt.legend()

plt.grid(True)

plt.show()

print("""

Small min_samples_split

↓

More Splits

↓

More Complex Tree

↓

Higher Overfitting Risk

Large min_samples_split

↓

Simpler Tree

↓

May Underfit

""")

# =============================================================================
# 14. HYPERPARAMETER SUMMARY
# =============================================================================

print("\n")
print("="*70)
print("IMPORTANT DECISION TREE HYPERPARAMETERS")
print("="*70)

summary = pd.DataFrame({

    "Hyperparameter":[

        "criterion",

        "max_depth",

        "min_samples_split",

        "min_samples_leaf",

        "ccp_alpha"

    ],

    "Purpose":[

        "Choose split criterion",

        "Control tree depth",

        "Minimum samples before split",

        "Minimum samples in each leaf",

        "Cost Complexity Pruning"

    ]

})

print(summary)

# =============================================================================
# TOP ML ENGINEER INSIGHT
# =============================================================================

print("\n")
print("="*70)
print("TOP ML ENGINEER INSIGHT")
print("="*70)

print("""

A Decision Tree can achieve

100% Training Accuracy

very easily.

That DOES NOT mean

it is a good model.

Always compare

Training Accuracy

and

Testing Accuracy.

A large gap

usually indicates

OVERFITTING.

Instead of blindly increasing

tree depth,

control complexity using

• max_depth

• min_samples_split

• min_samples_leaf

• Cost Complexity Pruning

These produce models

that GENERALIZE better.

""")

# =============================================================================
# DAY 28 : DECISION TREE
# PART 4.3 : COST COMPLEXITY PRUNING & DECISION TREE REGRESSOR
# =============================================================================

"""
Topics Covered

1. Cost Complexity Pruning (ccp_alpha)
2. Effect of Pruning
3. Decision Tree Regressor
4. Regression Metrics
5. Feature Importance (Regression)
6. Regression Visualization
"""

# =============================================================================
# IMPORT LIBRARIES
# =============================================================================

from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import load_diabetes

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# =============================================================================
# 15. COST COMPLEXITY PRUNING
# =============================================================================

print("\n")
print("="*70)
print("COST COMPLEXITY PRUNING")
print("="*70)

alphas = [0.0,0.001,0.005,0.01,0.02]

train_accuracy = []
test_accuracy = []
tree_depth = []

for alpha in alphas:

    clf = DecisionTreeClassifier(

        random_state=42,

        ccp_alpha=alpha

    )

    clf.fit(

        X_train,

        y_train

    )

    train_accuracy.append(

        clf.score(X_train,y_train)

    )

    test_accuracy.append(

        clf.score(X_test,y_test)

    )

    tree_depth.append(

        clf.get_depth()

    )

pruning_results = pd.DataFrame({

    "ccp_alpha":alphas,

    "Tree Depth":tree_depth,

    "Training Accuracy":train_accuracy,

    "Testing Accuracy":test_accuracy

})

print(pruning_results)

# =============================================================================
# VISUALIZE PRUNING
# =============================================================================

plt.figure(figsize=(8,5))

plt.plot(

    alphas,

    train_accuracy,

    marker="o",

    label="Training"

)

plt.plot(

    alphas,

    test_accuracy,

    marker="s",

    label="Testing"

)

plt.xlabel("ccp_alpha")

plt.ylabel("Accuracy")

plt.title("Cost Complexity Pruning")

plt.grid(True)

plt.legend()

plt.show()

print("""

Observation

Increasing ccp_alpha

↓

Removes unnecessary branches

↓

Smaller Tree

↓

Lower Overfitting

Too much pruning

↓

Underfitting

""")

# =============================================================================
# TREE DEPTH AFTER PRUNING
# =============================================================================

plt.figure(figsize=(8,5))

plt.plot(

    alphas,

    tree_depth,

    marker="o"

)

plt.xlabel("ccp_alpha")

plt.ylabel("Tree Depth")

plt.title("Tree Depth vs ccp_alpha")

plt.grid(True)

plt.show()

# =============================================================================
# 16. DECISION TREE REGRESSOR
# =============================================================================

print("\n")
print("="*70)
print("DECISION TREE REGRESSOR")
print("="*70)

diabetes = load_diabetes()

X_reg = pd.DataFrame(

    diabetes.data,

    columns=diabetes.feature_names

)

y_reg = diabetes.target

print("Dataset Shape :",X_reg.shape)

# =============================================================================
# TRAIN TEST SPLIT
# =============================================================================

X_train_reg,X_test_reg,y_train_reg,y_test_reg = train_test_split(

    X_reg,

    y_reg,

    test_size=0.20,

    random_state=42

)

# =============================================================================
# TRAIN REGRESSOR
# =============================================================================

regressor = DecisionTreeRegressor(

    random_state=42,

    max_depth=4

)

regressor.fit(

    X_train_reg,

    y_train_reg

)

print("\nDecision Tree Regressor Trained Successfully!")

# =============================================================================
# PREDICTIONS
# =============================================================================

predictions = regressor.predict(

    X_test_reg

)

print("\nFirst Five Predictions")

print(predictions[:5])

# =============================================================================
# REGRESSION METRICS
# =============================================================================

mae = mean_absolute_error(

    y_test_reg,

    predictions

)

mse = mean_squared_error(

    y_test_reg,

    predictions

)

rmse = np.sqrt(mse)

r2 = r2_score(

    y_test_reg,

    predictions

)

print("\n")
print("="*70)
print("REGRESSION METRICS")
print("="*70)

print("MAE  :",round(mae,2))

print("MSE  :",round(mse,2))

print("RMSE :",round(rmse,2))

print("R²   :",round(r2,3))

# =============================================================================
# ACTUAL VS PREDICTED
# =============================================================================

results = pd.DataFrame({

    "Actual":y_test_reg[:10],

    "Predicted":np.round(predictions[:10],2)

})

print("\nActual vs Predicted")

print(results)

# =============================================================================
# FEATURE IMPORTANCE (REGRESSION)
# =============================================================================

importance = pd.DataFrame({

    "Feature":X_reg.columns,

    "Importance":regressor.feature_importances_

})

importance = importance.sort_values(

    by="Importance",

    ascending=False

)

print("\nTop Features")

print(importance.head(10))

plt.figure(figsize=(10,6))

plt.barh(

    importance["Feature"][:10],

    importance["Importance"][:10]

)

plt.gca().invert_yaxis()

plt.xlabel("Importance")

plt.title("Decision Tree Regressor Feature Importance")

plt.show()

# =============================================================================
# REGRESSION TREE VISUALIZATION
# =============================================================================

plt.figure(figsize=(20,10))

plot_tree(

    regressor,

    feature_names=X_reg.columns,

    filled=True,

    rounded=True,

    fontsize=8

)

plt.title("Decision Tree Regressor")

plt.show()

# =============================================================================
# REGRESSION INSIGHT
# =============================================================================

print("\n")
print("="*70)
print("DECISION TREE REGRESSION")
print("="*70)

print("""

Classification Tree

↓

Predicts Categories

Examples

Spam

Fraud

Disease

-----------------------------------------

Regression Tree

↓

Predicts Continuous Values

Examples

House Price

Temperature

Salary

Stock Price

-----------------------------------------

Unlike Classification Trees,

Regression Trees split the data

to minimize prediction error

(MSE by default).

""")

# =============================================================================
# TOP ML ENGINEER INSIGHT
# =============================================================================

print("\n")
print("="*70)
print("TOP ML ENGINEER INSIGHT")
print("="*70)

print("""

Decision Trees are powerful,

but a single tree is rarely the best
production model.

Modern ML systems often use:

✔ Random Forest

✔ Extra Trees

✔ XGBoost

✔ LightGBM

✔ CatBoost

These ensemble methods combine
many trees to improve accuracy,
stability, and generalization.

Understanding a single Decision Tree
is essential because it is the
foundation of all these algorithms.

""")


# =============================================================================
# DAY 28 : DECISION TREE
# PART 4.4 : PRACTICE, INTERVIEW NOTES & FINAL SUMMARY
# =============================================================================

# =============================================================================
# 24. PRACTICE EXERCISES
# =============================================================================

print("\n")
print("="*70)
print("PRACTICE EXERCISES")
print("="*70)

print("""

Beginner

1. Train a Decision Tree using

   criterion = "entropy"

   Compare it with

   criterion = "gini"

------------------------------------------------------------

2. Change

   max_depth

   to

   2

   4

   6

   8

   None

Observe

• Training Accuracy

• Testing Accuracy

------------------------------------------------------------

3. Experiment with

   min_samples_split

   2

   5

   10

   20

Observe

how tree complexity changes.

------------------------------------------------------------

Intermediate

4. Train

DecisionTreeRegressor

using different

max_depth values.

Compare

MAE

RMSE

R² Score

------------------------------------------------------------

5. Try different

ccp_alpha

values.

Observe

Tree Depth

Training Accuracy

Testing Accuracy

------------------------------------------------------------

Advanced

6. Compare

Decision Tree

vs

Random Forest

on the same dataset.

Which performs better?

Why?

------------------------------------------------------------

7. Compare

Decision Tree

vs

Logistic Regression

When would you choose one
over the other?

------------------------------------------------------------

8. Download a dataset from Kaggle

Train

DecisionTreeClassifier

Perform

Hyperparameter Tuning

Evaluate

Feature Importance

Visualize the Tree

""")

# =============================================================================
# 25. MINI CHALLENGE
# =============================================================================

print("\n")
print("="*70)
print("MINI CHALLENGE")
print("="*70)

print("""

Challenge

Use the Titanic Dataset

Build

Decision Tree Classifier

Requirements

✔ Data Cleaning

✔ Handle Missing Values

✔ Feature Engineering

✔ Train-Test Split

✔ Decision Tree

✔ Accuracy

✔ Confusion Matrix

✔ Feature Importance

✔ Tree Visualization

Bonus

Compare with

Random Forest

""")

# =============================================================================
# 26. COMMON INTERVIEW QUESTIONS
# =============================================================================

print("\n")
print("="*70)
print("COMMON INTERVIEW QUESTIONS")
print("="*70)

questions = [

"What is a Decision Tree?",

"Why is it called a Tree?",

"Explain Entropy.",

"What is Information Gain?",

"What is Gini Impurity?",

"Why is Gini faster than Entropy?",

"Why do Decision Trees overfit?",

"What is Recursive Partitioning?",

"What is Pruning?",

"Difference between Pre-Pruning and Post-Pruning?",

"Why don't Decision Trees require Feature Scaling?",

"What is Cost Complexity Pruning?",

"What is ccp_alpha?",

"What is max_depth?",

"What is min_samples_split?",

"What is min_samples_leaf?",

"What is Feature Importance?",

"Decision Tree vs Random Forest?",

"Decision Tree vs Logistic Regression?",

"Advantages of Decision Trees?",

"Limitations of Decision Trees?",

"When should you use Decision Trees?"

]

for q in questions:

    print("✔",q)

# =============================================================================
# 27. SENIOR ML ENGINEER NOTES
# =============================================================================

print("\n")
print("="*70)
print("SENIOR ML ENGINEER NOTES")
print("="*70)

print("""

When analysing a Decision Tree,

never stop at Accuracy.

Always investigate:

✔ Tree Depth

Deep trees usually indicate
higher complexity.

------------------------------------------------------------

✔ Number of Leaf Nodes

Too many leaves often
mean overfitting.

------------------------------------------------------------

✔ Training vs Testing Accuracy

Large difference

↓

Overfitting

------------------------------------------------------------

✔ Feature Importance

Does the model rely on
reasonable features?

Are important features
consistent with domain knowledge?

------------------------------------------------------------

✔ Confusion Matrix

Which mistakes matter most?

False Positives?

False Negatives?

------------------------------------------------------------

✔ Business Cost

The best model is NOT

the one with

highest Accuracy.

It is the one that minimizes

business loss.

------------------------------------------------------------

✔ Hyperparameters

Always experiment with

max_depth

min_samples_split

min_samples_leaf

criterion

ccp_alpha

instead of accepting defaults.

------------------------------------------------------------

✔ Cross Validation

Never trust

a single train-test split.

Validate using

K-Fold Cross Validation.

------------------------------------------------------------

✔ Ensemble Methods

If a single Decision Tree
is unstable,

consider

Random Forest

Gradient Boosting

XGBoost

LightGBM

CatBoost

These usually generalize better.

""")

# =============================================================================
# 28. REAL-WORLD APPLICATIONS
# =============================================================================

print("\n")
print("="*70)
print("REAL-WORLD APPLICATIONS")
print("="*70)

applications = [

"Loan Approval",

"Medical Diagnosis",

"Fraud Detection",

"Credit Risk Analysis",

"Customer Churn Prediction",

"Insurance Claim Prediction",

"Marketing Campaign Analysis",

"Employee Attrition Prediction",

"Manufacturing Quality Control",

"Recommendation Systems (Tree Ensembles)"

]

for app in applications:

    print("✔",app)

# =============================================================================
# 29. GITHUB READY REVISION NOTES
# =============================================================================

print("\n")
print("="*70)
print("GITHUB REVISION NOTES")
print("="*70)

print("""

Decision Tree

↓

Supervised Learning Algorithm

------------------------------------------------------------

Works for

✔ Classification

✔ Regression

------------------------------------------------------------

Learns using

Recursive Partitioning

------------------------------------------------------------

Splits chosen using

✔ Gini Impurity

✔ Entropy

✔ Information Gain

------------------------------------------------------------

Advantages

✔ Easy to Understand

✔ Highly Interpretable

✔ Handles Non-linear Relationships

✔ No Feature Scaling Required

✔ Works with Numerical Data

------------------------------------------------------------

Disadvantages

✔ Easily Overfits

✔ Sensitive to Small Data Changes

✔ Lower Accuracy than Ensemble Methods

------------------------------------------------------------

Important Hyperparameters

✔ criterion

✔ max_depth

✔ min_samples_split

✔ min_samples_leaf

✔ ccp_alpha

------------------------------------------------------------

Prevents Overfitting

✔ Pruning

✔ More Data

✔ Cross Validation

✔ Hyperparameter Tuning

""")

# =============================================================================
# 30. FINAL TAKEAWAYS
# =============================================================================

print("\n")
print("="*70)
print("FINAL TAKEAWAYS")
print("="*70)

takeaways = [

"A Decision Tree learns a series of if-else rules.",

"It recursively splits the dataset into purer groups.",

"Entropy and Gini measure node impurity.",

"Information Gain selects the best split.",

"Decision Trees naturally handle non-linear relationships.",

"Feature Scaling is generally unnecessary.",

"Deep trees often overfit the training data.",

"Pruning controls model complexity and improves generalization.",

"Hyperparameters greatly influence model performance.",

"Feature Importance helps explain model decisions.",

"Decision Trees are the foundation of Random Forest, XGBoost, LightGBM and CatBoost.",

"Generalization is always more important than perfect training accuracy."

]

for i, takeaway in enumerate(takeaways, start=1):

    print(f"{i}. {takeaway}")

# =============================================================================
# 31. WHAT I LEARNED TODAY
# =============================================================================

print("\n")
print("="*70)
print("WHAT I LEARNED TODAY")
print("="*70)

print("""

Today I learned

✔ How Decision Trees think.

✔ How they choose the best split.

✔ Entropy

✔ Information Gain

✔ Gini Impurity

✔ Recursive Partitioning

✔ Classification Trees

✔ Regression Trees

✔ Tree Visualization

✔ Feature Importance

✔ Hyperparameter Tuning

✔ Cost Complexity Pruning

✔ Overfitting

✔ Generalization

✔ Real-world applications.

Decision Trees are one of the easiest models to understand,

yet they form the foundation of some of the world's most powerful Machine Learning algorithms.

""")

# =============================================================================
# END OF DAY 28
# =============================================================================

print("\n")
print("="*70)
print("END OF DAY 28 - DECISION TREES")
print("="*70)
print("Next Topic : Random Forest 🌲🌲🌲")

# =============================================================================
# SELF-REFLECTION
# =============================================================================

print("""

Questions to ask myself:

1. Can I explain this concept without looking at notes?

2. Why was this algorithm invented?

3. Where is it used in industry?

4. What assumptions does it make?

5. What are its limitations?

6. Which hyperparameters matter the most?

7. What would I use instead of this algorithm?

8. Can I implement a basic version from scratch?

9. Can I explain this in an interview?

10. Can I build a real project using it?

""")