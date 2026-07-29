# =============================================================================
# DAY 29 : RANDOM FOREST
# PART 4.1 : DECISION TREE vs RANDOM FOREST
# =============================================================================

"""
Topics Covered

1. Load Dataset
2. Train-Test Split
3. Decision Tree Classifier
4. Random Forest Classifier
5. Accuracy Comparison
6. Confusion Matrix
7. Classification Report
8. Prediction Probabilities

"""

# =============================================================================
# IMPORT LIBRARIES
# =============================================================================

import pandas as pd
import numpy as np

from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (

    accuracy_score,

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
# WHY NO FEATURE SCALING?
# =============================================================================

print("\nRandom Forest does NOT require feature scaling.")

print("Trees split using thresholds, not distances.")

# =============================================================================
# 3. DECISION TREE
# =============================================================================

print("\n")
print("="*70)
print("DECISION TREE")
print("="*70)

tree = DecisionTreeClassifier(

    random_state=42

)

tree.fit(

    X_train,

    y_train

)

tree_pred = tree.predict(

    X_test

)

tree_accuracy = accuracy_score(

    y_test,

    tree_pred

)

print("Decision Tree Accuracy :", round(tree_accuracy,4))

# =============================================================================
# 4. RANDOM FOREST
# =============================================================================

print("\n")
print("="*70)
print("RANDOM FOREST")
print("="*70)

forest = RandomForestClassifier(

    n_estimators=100,

    random_state=42

)

forest.fit(

    X_train,

    y_train

)

forest_pred = forest.predict(

    X_test

)

forest_prob = forest.predict_proba(

    X_test

)

forest_accuracy = accuracy_score(

    y_test,

    forest_pred

)

print("Random Forest Accuracy :", round(forest_accuracy,4))

# =============================================================================
# 5. ACCURACY COMPARISON
# =============================================================================

print("\n")
print("="*70)
print("MODEL COMPARISON")
print("="*70)

comparison = pd.DataFrame({

    "Model":[

        "Decision Tree",

        "Random Forest"

    ],

    "Accuracy":[

        tree_accuracy,

        forest_accuracy

    ]

})

print(comparison)

# =============================================================================
# BETTER MODEL
# =============================================================================

if forest_accuracy > tree_accuracy:

    print("\nRandom Forest performs better.")

elif tree_accuracy > forest_accuracy:

    print("\nDecision Tree performs better.")

else:

    print("\nBoth models have equal accuracy.")

# =============================================================================
# 6. CONFUSION MATRIX
# =============================================================================

print("\n")
print("="*70)
print("CONFUSION MATRIX")
print("="*70)

cm = confusion_matrix(

    y_test,

    forest_pred

)

print(cm)

print("""

Top Left

↓

True Negative

Top Right

↓

False Positive

Bottom Left

↓

False Negative

Bottom Right

↓

True Positive

""")

# =============================================================================
# 7. CLASSIFICATION REPORT
# =============================================================================

print("\n")
print("="*70)
print("CLASSIFICATION REPORT")
print("="*70)

print(

    classification_report(

        y_test,

        forest_pred,

        target_names=data.target_names

    )

)

# =============================================================================
# 8. FIRST FIVE PREDICTIONS
# =============================================================================

print("\n")
print("="*70)
print("FIRST FIVE PREDICTIONS")
print("="*70)

results = pd.DataFrame({

    "Actual":y_test[:5],

    "Predicted":forest_pred[:5]

})

print(results)

print("\nPrediction Probabilities")

print(forest_prob[:5])

# =============================================================================
# MODEL INFORMATION
# =============================================================================

print("\n")
print("="*70)
print("MODEL INFORMATION")
print("="*70)

print("Number of Trees :", forest.n_estimators)

print("Max Depth :", forest.max_depth)

print("Bootstrap :", forest.bootstrap)

# =============================================================================
# TOP ML ENGINEER INSIGHT
# =============================================================================

print("\n")
print("="*70)
print("TOP ML ENGINEER INSIGHT")
print("="*70)

print("""

Random Forest usually performs
better than a single Decision Tree
because it combines predictions
from many independent trees.

Instead of trusting one model,

it trusts the collective decision
of the entire forest.

This reduces variance and
improves generalization.

Always compare:

✔ Accuracy

✔ Precision

✔ Recall

✔ F1 Score

instead of relying on
Accuracy alone.

""")

# =============================================================================
# DAY 29 : RANDOM FOREST
# PART 4.2 : HYPERPARAMETER ANALYSIS & FEATURE IMPORTANCE
# =============================================================================

"""
Topics Covered

1. Effect of n_estimators
2. Effect of max_depth
3. Out-of-Bag (OOB) Score
4. Feature Importance
5. Accuracy Comparison Graphs
6. Hyperparameter Summary

"""

# =============================================================================
# IMPORT LIBRARIES
# =============================================================================

import matplotlib.pyplot as plt

# =============================================================================
# 9. EFFECT OF n_estimators
# =============================================================================

print("\n")
print("="*70)
print("EFFECT OF n_estimators")
print("="*70)

estimators = [1,5,10,25,50,100,200]

train_accuracy = []

test_accuracy = []

for n in estimators:

    model = RandomForestClassifier(

        n_estimators=n,

        random_state=42

    )

    model.fit(

        X_train,

        y_train

    )

    train_accuracy.append(

        model.score(

            X_train,

            y_train

        )

    )

    test_accuracy.append(

        model.score(

            X_test,

            y_test

        )

    )

results = pd.DataFrame({

    "n_estimators":estimators,

    "Training Accuracy":train_accuracy,

    "Testing Accuracy":test_accuracy

})

print(results)

plt.figure(figsize=(8,5))

plt.plot(

    estimators,

    train_accuracy,

    marker="o",

    label="Training"

)

plt.plot(

    estimators,

    test_accuracy,

    marker="s",

    label="Testing"

)

plt.xlabel("Number of Trees")

plt.ylabel("Accuracy")

plt.title("Effect of n_estimators")

plt.grid(True)

plt.legend()

plt.show()

print("""

Observation

More Trees

↓

Lower Variance

↓

Better Stability

↓

Longer Training Time

Beyond a certain point

accuracy improves very little.

""")

# =============================================================================
# 10. EFFECT OF max_depth
# =============================================================================

print("\n")
print("="*70)
print("EFFECT OF max_depth")
print("="*70)

depths = [2,4,6,8,10,None]

train_scores = []

test_scores = []

for depth in depths:

    model = RandomForestClassifier(

        n_estimators=100,

        max_depth=depth,

        random_state=42

    )

    model.fit(

        X_train,

        y_train

    )

    train_scores.append(

        model.score(

            X_train,

            y_train

        )

    )

    test_scores.append(

        model.score(

            X_test,

            y_test

        )

    )

depth_results = pd.DataFrame({

    "max_depth":depths,

    "Training Accuracy":train_scores,

    "Testing Accuracy":test_scores

})

print(depth_results)

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

plt.title("Effect of max_depth")

plt.grid(True)

plt.legend()

plt.show()

print("""

Small Depth

↓

Higher Bias

Lower Variance

Large Depth

↓

Lower Bias

Higher Variance

Random Forest controls
variance much better
than a single tree.

""")

# =============================================================================
# 11. OUT-OF-BAG SCORE
# =============================================================================

print("\n")
print("="*70)
print("OUT-OF-BAG SCORE")
print("="*70)

oob_model = RandomForestClassifier(

    n_estimators=200,

    oob_score=True,

    random_state=42

)

oob_model.fit(

    X_train,

    y_train

)

print("OOB Score :", round(

    oob_model.oob_score_,

    4

))

print("""

OOB Score

↓

Built-in validation

without creating a
separate validation set.

A good OOB Score
is usually close to
Testing Accuracy.

""")

# =============================================================================
# 12. FEATURE IMPORTANCE
# =============================================================================

print("\n")
print("="*70)
print("FEATURE IMPORTANCE")
print("="*70)

importance = pd.DataFrame({

    "Feature":X.columns,

    "Importance":oob_model.feature_importances_

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

plt.xlabel("Importance")

plt.title("Top 10 Important Features")

plt.grid(True)

plt.show()

print("""

Higher Importance

↓

Greater contribution

towards prediction.

Remember

Feature Importance
shows association,

not causation.

""")

# =============================================================================
# 13. FEATURE IMPORTANCE PERCENTAGE
# =============================================================================

print("\n")
print("="*70)
print("FEATURE IMPORTANCE (%)")
print("="*70)

importance["Percentage"] = (

    importance["Importance"] * 100

).round(2)

print(

    importance[

        ["Feature","Percentage"]

    ].head(10)

)

# =============================================================================
# 14. HYPERPARAMETER SUMMARY
# =============================================================================

print("\n")
print("="*70)
print("IMPORTANT HYPERPARAMETERS")
print("="*70)

summary = pd.DataFrame({

    "Hyperparameter":[

        "n_estimators",

        "max_depth",

        "max_features",

        "min_samples_split",

        "min_samples_leaf",

        "bootstrap",

        "oob_score"

    ],

    "Purpose":[

        "Number of Trees",

        "Maximum Tree Depth",

        "Random Features",

        "Minimum Samples Before Split",

        "Minimum Samples in Leaf",

        "Bootstrap Sampling",

        "Built-in Validation"

    ]

})

print(summary)

# =============================================================================
# 15. TOP ML ENGINEER INSIGHT
# =============================================================================

print("\n")
print("="*70)
print("TOP ML ENGINEER INSIGHT")
print("="*70)

print("""

Never increase

n_estimators

indefinitely.

After a certain point,

accuracy hardly changes,

while computation
continues to increase.

Always balance

✔ Accuracy

✔ Training Time

✔ Memory Usage

✔ Inference Speed

Also,

Feature Importance
helps explain

which variables
influenced predictions,

but should NOT be used
to claim cause-and-effect.

""")

# =============================================================================
# DAY 29 : RANDOM FOREST
# PART 4.3 : RANDOM FOREST REGRESSOR
# =============================================================================

"""
Topics Covered

1. Load Regression Dataset
2. Train-Test Split
3. Train Random Forest Regressor
4. Predictions
5. Regression Metrics
6. Actual vs Predicted
7. Feature Importance
8. Hyperparameter Experiment
9. Industry Insight

"""

# =============================================================================
# IMPORT LIBRARIES
# =============================================================================

from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (

    mean_absolute_error,

    mean_squared_error,

    r2_score

)

# =============================================================================
# 16. LOAD DATASET
# =============================================================================

print("\n")
print("="*70)
print("LOADING DIABETES DATASET")
print("="*70)

diabetes = load_diabetes()

X_reg = pd.DataFrame(

    diabetes.data,

    columns=diabetes.feature_names

)

y_reg = diabetes.target

print("Dataset Shape :", X_reg.shape)

# =============================================================================
# 17. TRAIN TEST SPLIT
# =============================================================================

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(

    X_reg,

    y_reg,

    test_size=0.20,

    random_state=42

)

print("Training Samples :", len(X_train_reg))

print("Testing Samples  :", len(X_test_reg))

# =============================================================================
# 18. TRAIN RANDOM FOREST REGRESSOR
# =============================================================================

print("\n")
print("="*70)
print("TRAINING RANDOM FOREST REGRESSOR")
print("="*70)

regressor = RandomForestRegressor(

    n_estimators=200,

    random_state=42

)

regressor.fit(

    X_train_reg,

    y_train_reg

)

print("Model Trained Successfully!")

# =============================================================================
# 19. MAKE PREDICTIONS
# =============================================================================

predictions = regressor.predict(

    X_test_reg

)

print("\nFirst Five Predictions")

print(np.round(predictions[:5],2))

# =============================================================================
# 20. MODEL EVALUATION
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

print("MAE  :", round(mae,2))

print("MSE  :", round(mse,2))

print("RMSE :", round(rmse,2))

print("R²   :", round(r2,3))

# =============================================================================
# 21. ACTUAL vs PREDICTED
# =============================================================================

results = pd.DataFrame({

    "Actual": y_test_reg[:10],

    "Predicted": np.round(predictions[:10],2)

})

print("\nActual vs Predicted")

print(results)

# =============================================================================
# 22. FEATURE IMPORTANCE
# =============================================================================

print("\n")
print("="*70)
print("FEATURE IMPORTANCE")
print("="*70)

importance = pd.DataFrame({

    "Feature": X_reg.columns,

    "Importance": regressor.feature_importances_

})

importance = importance.sort_values(

    by="Importance",

    ascending=False

)

print(importance)

plt.figure(figsize=(10,6))

plt.barh(

    importance["Feature"],

    importance["Importance"]

)

plt.gca().invert_yaxis()

plt.title("Random Forest Regressor Feature Importance")

plt.xlabel("Importance")

plt.grid(True)

plt.show()

# =============================================================================
# 23. EFFECT OF NUMBER OF TREES
# =============================================================================

print("\n")
print("="*70)
print("EFFECT OF n_estimators")
print("="*70)

estimators = [10,50,100,200,300]

scores = []

for n in estimators:

    model = RandomForestRegressor(

        n_estimators=n,

        random_state=42

    )

    model.fit(

        X_train_reg,

        y_train_reg

    )

    pred = model.predict(

        X_test_reg

    )

    scores.append(

        r2_score(

            y_test_reg,

            pred

        )

    )

comparison = pd.DataFrame({

    "Trees": estimators,

    "R² Score": scores

})

print(comparison)

plt.figure(figsize=(8,5))

plt.plot(

    estimators,

    scores,

    marker="o"

)

plt.xlabel("Number of Trees")

plt.ylabel("R² Score")

plt.title("Effect of n_estimators on Regression")

plt.grid(True)

plt.show()

# =============================================================================
# 24. PREDICTION VISUALIZATION
# =============================================================================

plt.figure(figsize=(8,6))

plt.scatter(

    y_test_reg,

    predictions

)

plt.xlabel("Actual Values")

plt.ylabel("Predicted Values")

plt.title("Actual vs Predicted")

plt.grid(True)

plt.show()

# =============================================================================
# 25. MODEL SUMMARY
# =============================================================================

print("\n")
print("="*70)
print("MODEL SUMMARY")
print("="*70)

print("""

Algorithm

↓

Random Forest Regressor

Task

↓

Regression

Prediction

↓

Average of all Trees

Evaluation Metrics

↓

MAE

MSE

RMSE

R² Score

Feature Scaling

↓

Not Required

""")

# =============================================================================
# TOP ML ENGINEER INSIGHT
# =============================================================================

print("\n")
print("="*70)
print("TOP ML ENGINEER INSIGHT")
print("="*70)

print("""

Random Forest is one of the
strongest baseline algorithms
for structured/tabular data.

However,

don't assume it is always
the best choice.

Always compare it with:

✔ Linear Regression

✔ Decision Tree

✔ Gradient Boosting

✔ XGBoost

✔ LightGBM

✔ CatBoost

Choose the model that provides
the best balance between

• Accuracy

• Interpretability

• Speed

• Memory Usage

• Business Requirements

""")

# =============================================================================
# DAY 29 : RANDOM FOREST
# PART 4.4 : PRACTICE, INTERVIEW NOTES & FINAL SUMMARY
# =============================================================================

# =============================================================================
# 26. PRACTICE EXERCISES
# =============================================================================

print("\n")
print("="*70)
print("PRACTICE EXERCISES")
print("="*70)

print("""

Beginner

1. Train a Random Forest using

   n_estimators = 10

   n_estimators = 50

   n_estimators = 100

Compare their accuracies.

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

Training Accuracy

Testing Accuracy

------------------------------------------------------------

3. Enable

oob_score=True

Compare

OOB Score

Testing Accuracy

------------------------------------------------------------

Intermediate

4. Compare

Decision Tree

vs

Random Forest

using

Accuracy

Precision

Recall

F1 Score

------------------------------------------------------------

5. Compare Feature Importance

Decision Tree

vs

Random Forest

Which features remain important?

------------------------------------------------------------

Advanced

6. Train

RandomForestRegressor

using

50

100

200

500

Trees

Compare

MAE

RMSE

R² Score

------------------------------------------------------------

7. Download a Kaggle Dataset

Build

Random Forest

Perform

Hyperparameter Tuning

Evaluate Performance

Explain Feature Importance

""")

# =============================================================================
# 27. MINI CHALLENGE
# =============================================================================

print("\n")
print("="*70)
print("MINI CHALLENGE")
print("="*70)

print("""

Dataset

Titanic Survival

Tasks

✔ Data Cleaning

✔ Handle Missing Values

✔ Train-Test Split

✔ Decision Tree

✔ Random Forest

✔ Accuracy Comparison

✔ Confusion Matrix

✔ Classification Report

✔ Feature Importance

Bonus

Tune

n_estimators

max_depth

max_features

Compare results.

""")

# =============================================================================
# 28. COMMON INTERVIEW QUESTIONS
# =============================================================================

print("\n")
print("="*70)
print("COMMON INTERVIEW QUESTIONS")
print("="*70)

questions = [

"What is Random Forest?",

"Why was Random Forest invented?",

"What is Ensemble Learning?",

"What is Bootstrap Sampling?",

"What is Bagging?",

"What is Random Feature Selection?",

"What is Majority Voting?",

"What is Averaging in Regression?",

"What is OOB Score?",

"Why does Random Forest reduce overfitting?",

"Why doesn't Random Forest require feature scaling?",

"What is Feature Importance?",

"Decision Tree vs Random Forest?",

"Random Forest vs Logistic Regression?",

"Random Forest vs XGBoost?",

"What is n_estimators?",

"What is max_features?",

"What is max_depth?",

"What is bootstrap=True?",

"Advantages of Random Forest?",

"Limitations of Random Forest?",

"When should you use Random Forest?"

]

for question in questions:

    print("✔", question)

# =============================================================================
# 29. SENIOR ML ENGINEER NOTES
# =============================================================================

print("\n")
print("="*70)
print("SENIOR ML ENGINEER NOTES")
print("="*70)

print("""

Don't judge a Random Forest
only by Accuracy.

Always analyse

✔ Precision

✔ Recall

✔ F1 Score

✔ ROC-AUC

✔ OOB Score

✔ Feature Importance

✔ Training Time

✔ Prediction Time

------------------------------------------------------------

Increasing

n_estimators

usually improves stability,

but after a point

performance improves only slightly

while computation increases.

------------------------------------------------------------

Always compare

Training Accuracy

Testing Accuracy

OOB Score

If all three are close,

the model is usually
generalizing well.

------------------------------------------------------------

Feature Importance

helps explain

which variables influenced predictions,

but it does NOT imply
causation.

------------------------------------------------------------

Random Forest is often

the first strong baseline

for structured datasets.

Always compare it against

Gradient Boosting

XGBoost

LightGBM

CatBoost

before deploying.

------------------------------------------------------------

Business value matters.

A model with

96% Accuracy

may be better than

98% Accuracy

if it is

faster

simpler

more interpretable

and cheaper to maintain.

""")

# =============================================================================
# 30. REAL-WORLD APPLICATIONS
# =============================================================================

print("\n")
print("="*70)
print("REAL-WORLD APPLICATIONS")
print("="*70)

applications = [

"Disease Prediction",

"Fraud Detection",

"Credit Risk Assessment",

"Insurance Risk Analysis",

"Customer Churn Prediction",

"Recommendation Systems",

"Manufacturing Quality Control",

"Agriculture",

"Financial Forecasting",

"Remote Sensing"

]

for app in applications:

    print("✔", app)

# =============================================================================
# 31. GITHUB REVISION NOTES
# =============================================================================

print("\n")
print("="*70)
print("GITHUB REVISION NOTES")
print("="*70)

print("""

Random Forest

↓

Ensemble Learning Algorithm

------------------------------------------------------------

Built Using

Many Decision Trees

------------------------------------------------------------

Core Ideas

✔ Bootstrap Sampling

✔ Bagging

✔ Random Feature Selection

✔ Majority Voting

✔ Averaging

------------------------------------------------------------

Reduces

Variance

------------------------------------------------------------

Improves

Generalization

------------------------------------------------------------

Works For

✔ Classification

✔ Regression

------------------------------------------------------------

Feature Scaling

Not Required

------------------------------------------------------------

Important Hyperparameters

✔ n_estimators

✔ max_depth

✔ max_features

✔ min_samples_split

✔ min_samples_leaf

✔ bootstrap

✔ oob_score

------------------------------------------------------------

Advantages

✔ Robust

✔ High Accuracy

✔ Less Overfitting

✔ Feature Importance

✔ Parallel Training

------------------------------------------------------------

Limitations

✔ Slower

✔ Larger Memory Usage

✔ Less Interpretable

""")

# =============================================================================
# 32. FINAL TAKEAWAYS
# =============================================================================

print("\n")
print("="*70)
print("FINAL TAKEAWAYS")
print("="*70)

takeaways = [

"Random Forest is an ensemble of many Decision Trees.",

"Bootstrap Sampling creates diverse training datasets.",

"Random Feature Selection increases tree diversity.",

"Bagging reduces variance and improves stability.",

"Classification uses Majority Voting.",

"Regression uses Averaging.",

"OOB Score provides built-in model validation.",

"Feature Importance improves model interpretability.",

"Random Forest usually generalizes better than a single Decision Tree.",

"It is one of the strongest baseline models for tabular data.",

"Hyperparameter tuning significantly impacts performance.",

"Always choose models based on business requirements, not just accuracy."

]

for i, takeaway in enumerate(takeaways, start=1):

    print(f"{i}. {takeaway}")

# =============================================================================
# 33. WHAT I LEARNED TODAY
# =============================================================================

print("\n")
print("="*70)
print("WHAT I LEARNED TODAY")
print("="*70)

print("""

Today I learned

✔ Why Decision Trees overfit

✔ Why Random Forest was invented

✔ Ensemble Learning

✔ Bootstrap Sampling

✔ Bagging

✔ Random Feature Selection

✔ Majority Voting

✔ Averaging

✔ OOB Score

✔ Feature Importance

✔ Random Forest Classifier

✔ Random Forest Regressor

✔ Hyperparameter Tuning

✔ Industry Best Practices

✔ Real-world Applications

Random Forest showed me that
great Machine Learning models
often come from combining
many simple models intelligently,
rather than making one model
increasingly complex.

""")

# =============================================================================
# END OF DAY 29
# =============================================================================

print("\n")
print("="*70)
print("END OF DAY 29 - RANDOM FOREST")
print("="*70)
print("Next Topic : Gradient Boosting, XGBoost & LightGBM 🚀")