# =============================================================================
# DAY 34 : HYPERPARAMETER TUNING
# PART 4.1 : GRIDSEARCHCV
# =============================================================================

"""
Topics Covered

1. Load Dataset
2. Train-Test Split
3. Build Baseline Model
4. Evaluate Baseline
5. Define Parameter Grid
6. GridSearchCV
7. Best Parameters
8. Best Cross Validation Score
9. Best Estimator
10. Test Accuracy
11. Top ML Engineer Insight

"""

# =============================================================================
# IMPORT LIBRARIES
# =============================================================================

import pandas as pd

from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import (

    train_test_split,

    GridSearchCV

)

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (

    accuracy_score,

    classification_report,

    confusion_matrix

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

print("Dataset Shape :", X.shape)

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
# 3. BASELINE MODEL
# =============================================================================

print("\n")
print("=" * 70)
print("BASELINE RANDOM FOREST")
print("=" * 70)

baseline = RandomForestClassifier(

    random_state=42

)

baseline.fit(

    X_train,

    y_train

)

baseline_pred = baseline.predict(

    X_test

)

baseline_accuracy = accuracy_score(

    y_test,

    baseline_pred

)

print("Baseline Accuracy :",

      round(baseline_accuracy,4))

# =============================================================================
# 4. PARAMETER GRID
# =============================================================================

print("\n")
print("=" * 70)
print("PARAMETER GRID")
print("=" * 70)

param_grid = {

    "n_estimators":[

        50,

        100,

        200

    ],

    "max_depth":[

        3,

        5,

        10

    ],

    "min_samples_split":[

        2,

        5,

        10

    ]

}

print(param_grid)

# =============================================================================
# 5. GRID SEARCH
# =============================================================================

print("\n")
print("=" * 70)
print("RUNNING GRID SEARCH")
print("=" * 70)

grid = GridSearchCV(

    estimator=RandomForestClassifier(

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

# =============================================================================
# 6. BEST RESULTS
# =============================================================================

print("\n")
print("=" * 70)
print("BEST GRID SEARCH RESULTS")
print("=" * 70)

print("Best Parameters")

print(grid.best_params_)

print("\nBest Cross Validation Score")

print(round(grid.best_score_,4))

print("\nBest Estimator")

print(grid.best_estimator_)

# =============================================================================
# 7. TEST SET PERFORMANCE
# =============================================================================

best_model = grid.best_estimator_

prediction = best_model.predict(

    X_test

)

accuracy = accuracy_score(

    y_test,

    prediction

)

print("\n")
print("=" * 70)
print("FINAL TEST PERFORMANCE")
print("=" * 70)

print("Test Accuracy :",

      round(accuracy,4))

print("\nConfusion Matrix")

print(

    confusion_matrix(

        y_test,

        prediction

    )

)

print("\nClassification Report")

print(

    classification_report(

        y_test,

        prediction

    )

)

# =============================================================================
# 8. INTERPRETATION
# =============================================================================

print("\n")
print("=" * 70)
print("INTERPRETATION")
print("=" * 70)

print("""

GridSearchCV

↓

Tests every parameter
combination.

↓

Uses Cross Validation.

↓

Chooses the model with
the best average score.

↓

Returns

Best Parameters

Best Estimator

Best Score

Automatically.

""")

# =============================================================================
# 9. TOP ML ENGINEER INSIGHT
# =============================================================================

print("\n")
print("=" * 70)
print("TOP ML ENGINEER INSIGHT")
print("=" * 70)

print("""

Never guess

hyperparameters.

Define

a search space.

Use

Cross Validation.

Select the model

that generalizes best,

not just the one that

performs well

on a single split.

""")

# =============================================================================
# DAY 34 : HYPERPARAMETER TUNING
# PART 4.2 : RANDOMIZEDSEARCHCV
# =============================================================================

"""
Topics Covered

1. Load Dataset
2. Train-Test Split
3. RandomizedSearchCV
4. Best Parameters
5. Best Cross Validation Score
6. Best Estimator
7. Runtime Comparison
8. Test Accuracy
9. Grid Search vs Random Search
10. Top ML Engineer Insight

"""

# =============================================================================
# IMPORT LIBRARIES
# =============================================================================

import time
import pandas as pd

from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import (

    train_test_split,

    RandomizedSearchCV

)

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (

    accuracy_score,

    classification_report,

    confusion_matrix

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

print("Dataset Shape :", X.shape)

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
# 3. PARAMETER DISTRIBUTIONS
# =============================================================================

print("\n")
print("="*70)
print("PARAMETER DISTRIBUTIONS")
print("="*70)

param_dist = {

    "n_estimators":[

        50,

        100,

        150,

        200,

        300,

        500

    ],

    "max_depth":[

        None,

        3,

        5,

        7,

        10,

        15,

        20

    ],

    "min_samples_split":[

        2,

        5,

        10,

        20

    ],

    "min_samples_leaf":[

        1,

        2,

        4,

        8

    ],

    "criterion":[

        "gini",

        "entropy"

    ]

}

print(param_dist)

# =============================================================================
# 4. RANDOMIZED SEARCH
# =============================================================================

print("\n")
print("="*70)
print("RUNNING RANDOMIZED SEARCH")
print("="*70)

start_time = time.time()

random_search = RandomizedSearchCV(

    estimator=RandomForestClassifier(

        random_state=42

    ),

    param_distributions=param_dist,

    n_iter=20,

    cv=5,

    scoring="accuracy",

    random_state=42,

    n_jobs=-1

)

random_search.fit(

    X_train,

    y_train

)

end_time = time.time()

runtime = end_time - start_time

# =============================================================================
# 5. BEST RESULTS
# =============================================================================

print("\n")
print("="*70)
print("BEST RANDOM SEARCH RESULTS")
print("="*70)

print("Best Parameters")

print(random_search.best_params_)

print("\nBest Cross Validation Score")

print(round(

    random_search.best_score_,

    4

))

print("\nBest Estimator")

print(random_search.best_estimator_)

print("\nRuntime")

print(round(runtime,2),"seconds")

# =============================================================================
# 6. TEST SET EVALUATION
# =============================================================================

best_model = random_search.best_estimator_

prediction = best_model.predict(

    X_test

)

accuracy = accuracy_score(

    y_test,

    prediction

)

print("\n")
print("="*70)
print("FINAL TEST PERFORMANCE")
print("="*70)

print("Accuracy :",

      round(accuracy,4))

print("\nConfusion Matrix")

print(

    confusion_matrix(

        y_test,

        prediction

    )

)

print("\nClassification Report")

print(

    classification_report(

        y_test,

        prediction

    )

)

# =============================================================================
# 7. GRID SEARCH vs RANDOM SEARCH
# =============================================================================

print("\n")
print("="*70)
print("GRID SEARCH vs RANDOM SEARCH")
print("="*70)

comparison = pd.DataFrame({

    "Method":[

        "GridSearchCV",

        "RandomizedSearchCV"

    ],

    "Search Strategy":[

        "Every Combination",

        "Random Sampling"

    ],

    "Speed":[

        "Slower",

        "Faster"

    ],

    "Best For":[

        "Small Search Space",

        "Large Search Space"

    ]

})

print(comparison)

# =============================================================================
# 8. WHEN TO USE WHAT?
# =============================================================================

print("\n")
print("="*70)
print("WHEN TO USE WHICH?")
print("="*70)

print("""

Use GridSearchCV

✔ Small datasets

✔ Few hyperparameters

✔ Fine tuning

--------------------------------------------

Use RandomizedSearchCV

✔ Large datasets

✔ XGBoost

✔ LightGBM

✔ Random Forest

✔ Large search spaces

✔ Faster experimentation

""")

# =============================================================================
# 9. INDUSTRY WORKFLOW
# =============================================================================

print("\n")
print("="*70)
print("INDUSTRY WORKFLOW")
print("="*70)

print("""

Randomized Search

↓

Find promising region

↓

Grid Search

↓

Fine tune around
the best parameters

↓

Deploy

""")

# =============================================================================
# 10. TOP ML ENGINEER INSIGHT
# =============================================================================

print("\n")
print("="*70)
print("TOP ML ENGINEER INSIGHT")
print("="*70)

print("""

Searching every
possible combination
is rarely practical
for real-world problems.

Professional ML engineers
often start with

RandomizedSearchCV

because it explores
large search spaces
efficiently.

After identifying
a promising region,

they switch to

GridSearchCV

for precise tuning.

This two-stage strategy
balances speed,
accuracy,
and computational cost.

""")

# =============================================================================
# DAY 34 : HYPERPARAMETER TUNING
# PART 4.3 : PIPELINE + GRIDSEARCHCV
# =============================================================================

"""
Topics Covered

1. Load Dataset
2. Train-Test Split
3. Build Pipeline
4. StandardScaler
5. Logistic Regression
6. Define Hyperparameter Grid
7. GridSearchCV with Pipeline
8. Best Parameters
9. Best Estimator
10. Best Cross Validation Score
11. Final Test Evaluation
12. Confusion Matrix
13. Classification Report
14. Pipeline Advantages
15. Top ML Engineer Insight

"""

# =============================================================================
# IMPORT LIBRARIES
# =============================================================================

import pandas as pd

from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import (

    train_test_split,

    GridSearchCV

)

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

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

print("Dataset Shape :", X.shape)

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
# 3. BUILD PIPELINE
# =============================================================================

print("\n")
print("="*70)
print("BUILDING PIPELINE")
print("="*70)

pipeline = Pipeline([

    ("scaler", StandardScaler()),

    ("classifier",

     LogisticRegression(

         max_iter=1000,

         random_state=42

     ))

])

print(pipeline)

# =============================================================================
# 4. PARAMETER GRID
# =============================================================================

print("\n")
print("="*70)
print("PARAMETER GRID")
print("="*70)

param_grid = {

    "classifier__C":[

        0.01,

        0.1,

        1,

        10,

        100

    ],

    "classifier__penalty":[

        "l2"

    ],

    "classifier__solver":[

        "lbfgs",

        "liblinear"

    ]

}

print(param_grid)

# =============================================================================
# 5. GRID SEARCH
# =============================================================================

print("\n")
print("="*70)
print("RUNNING GRID SEARCH")
print("="*70)

grid = GridSearchCV(

    estimator=pipeline,

    param_grid=param_grid,

    cv=5,

    scoring="accuracy",

    n_jobs=-1

)

grid.fit(

    X_train,

    y_train

)

# =============================================================================
# 6. BEST RESULTS
# =============================================================================

print("\n")
print("="*70)
print("BEST GRID SEARCH RESULTS")
print("="*70)

print("Best Parameters")

print(grid.best_params_)

print("\n")

print("Best Cross Validation Score")

print(round(

    grid.best_score_,

    4

))

print("\n")

print("Best Estimator")

print(grid.best_estimator_)

# =============================================================================
# 7. FINAL TEST EVALUATION
# =============================================================================

best_pipeline = grid.best_estimator_

prediction = best_pipeline.predict(

    X_test

)

accuracy = accuracy_score(

    y_test,

    prediction

)

print("\n")
print("="*70)
print("FINAL TEST PERFORMANCE")
print("="*70)

print("Accuracy :",

      round(accuracy,4))

print("\nConfusion Matrix")

print(

    confusion_matrix(

        y_test,

        prediction

    )

)

print("\nClassification Report")

print(

    classification_report(

        y_test,

        prediction

    )

)

# =============================================================================
# 8. PIPELINE WORKFLOW
# =============================================================================

print("\n")
print("="*70)
print("PIPELINE WORKFLOW")
print("="*70)

print("""

Raw Data

↓

StandardScaler

↓

Logistic Regression

↓

Prediction

Everything is handled
automatically by
the Pipeline.

""")

# =============================================================================
# 9. WHY PIPELINES?
# =============================================================================

print("\n")
print("="*70)
print("WHY PIPELINES?")
print("="*70)

print("""

✔ Cleaner Code

✔ Less Repetition

✔ Prevents Data Leakage

✔ Easy Cross Validation

✔ Easy Hyperparameter Tuning

✔ Easy Deployment

✔ Reproducible Workflow

""")

# =============================================================================
# 10. DATA LEAKAGE PREVENTION
# =============================================================================

print("\n")
print("="*70)
print("DATA LEAKAGE PREVENTION")
print("="*70)

print("""

Without Pipeline

Scale Entire Dataset

↓

Split Data

❌ Data Leakage

------------------------------------

With Pipeline

Split Data

↓

Scale Training Fold Only

↓

Train Model

↓

Evaluate Validation Fold

✔ Correct Evaluation

""")

# =============================================================================
# 11. PRODUCTION WORKFLOW
# =============================================================================

print("\n")
print("="*70)
print("PRODUCTION ML WORKFLOW")
print("="*70)

print("""

Raw Data

↓

Train/Test Split

↓

Pipeline

↓

Cross Validation

↓

GridSearchCV

↓

Best Model

↓

Evaluation

↓

Deployment

""")

# =============================================================================
# 12. TOP ML ENGINEER INSIGHT
# =============================================================================

print("\n")
print("="*70)
print("TOP ML ENGINEER INSIGHT")
print("="*70)

print("""

Professional ML Engineers
rarely use

Scaler

↓

Model

as separate steps.

Instead they combine
everything into a

Pipeline

and perform

Cross Validation

+

Grid Search

on the complete workflow.

This ensures

✔ Correct preprocessing

✔ No data leakage

✔ Reproducibility

✔ Production-ready code

""")
# =============================================================================
# DAY 34 : HYPERPARAMETER TUNING
# PART 4.4 : FINAL NOTES, INTERVIEW PREP & GITHUB SUMMARY
# =============================================================================

# =============================================================================
# 13. PRACTICE EXERCISES
# =============================================================================

print("\n")
print("=" * 70)
print("PRACTICE EXERCISES")
print("=" * 70)

print("""

Beginner

1. Train a Random Forest model.

2. Tune

   n_estimators

   max_depth

using GridSearchCV.

----------------------------------------------------------

Intermediate

3. Compare

Baseline Model

vs

GridSearchCV Model

----------------------------------------------------------

4. Compare

GridSearchCV

vs

RandomizedSearchCV

using

Accuracy

Runtime

----------------------------------------------------------

Advanced

5. Build a Pipeline

StandardScaler

↓

Logistic Regression

↓

GridSearchCV

----------------------------------------------------------

6. Tune

Decision Tree

Random Forest

Support Vector Machine

Compare

Best Parameters

Best Scores

Training Time

""")

# =============================================================================
# 14. MINI CHALLENGE
# =============================================================================

print("\n")
print("=" * 70)
print("MINI CHALLENGE")
print("=" * 70)

print("""

Build a complete

Machine Learning Pipeline

using

✔ Train-Test Split

✔ Pipeline

✔ StandardScaler

✔ Logistic Regression

✔ GridSearchCV

✔ Cross Validation

✔ Best Parameters

✔ Final Test Accuracy

Bonus

Compare with

RandomizedSearchCV

Explain

Which method
you would deploy
in production.

""")

# =============================================================================
# 15. COMMON INTERVIEW QUESTIONS
# =============================================================================

print("\n")
print("=" * 70)
print("COMMON INTERVIEW QUESTIONS")
print("=" * 70)

questions = [

"What is a parameter?",

"What is a hyperparameter?",

"Difference between parameters and hyperparameters?",

"What is GridSearchCV?",

"What is RandomizedSearchCV?",

"Difference between GridSearchCV and RandomizedSearchCV?",

"When should you use GridSearchCV?",

"When should you use RandomizedSearchCV?",

"What is best_params_?",

"What is best_score_?",

"What is best_estimator_?",

"Why is Cross Validation used inside GridSearchCV?",

"What is a Pipeline?",

"How do Pipelines prevent data leakage?",

"Why should preprocessing be inside the Pipeline?",

"What does classifier__C mean?",

"Why shouldn't you tune hyperparameters on the test set?",

"What is a search space?",

"Why is Random Search faster?",

"What workflow do production ML engineers follow?"

]

for q in questions:

    print("✔", q)

# =============================================================================
# 16. SENIOR ML ENGINEER NOTES
# =============================================================================

print("\n")
print("=" * 70)
print("SENIOR ML ENGINEER NOTES")
print("=" * 70)

print("""

Parameters

↓

Learned by the model.

Examples

Weights

Tree Splits

Leaf Values

--------------------------------------------------

Hyperparameters

↓

Chosen before training.

Examples

max_depth

n_estimators

learning_rate

C

gamma

--------------------------------------------------

GridSearchCV

↓

Checks

Every Combination

Best for

Small Search Spaces

--------------------------------------------------

RandomizedSearchCV

↓

Checks

Random Combinations

Best for

Large Search Spaces

--------------------------------------------------

Pipeline

↓

Combines

Preprocessing

+

Model

--------------------------------------------------

Pipeline prevents

Data Leakage

during

Cross Validation

--------------------------------------------------

Never tune

hyperparameters

using

the Test Set.

The Test Set

must remain untouched

until the final evaluation.

--------------------------------------------------

Production Workflow

Pipeline

↓

Cross Validation

↓

Hyperparameter Search

↓

Best Model

↓

Final Test Set

↓

Deployment

""")

# =============================================================================
# 17. REAL-WORLD APPLICATIONS
# =============================================================================

print("\n")
print("=" * 70)
print("REAL-WORLD APPLICATIONS")
print("=" * 70)

applications = [

"Fraud Detection",

"Healthcare AI",

"Customer Churn Prediction",

"Recommendation Systems",

"Credit Risk Assessment",

"Kaggle Competitions",

"Computer Vision",

"NLP Pipelines",

"Search Ranking",

"Production ML Systems"

]

for app in applications:

    print("✔", app)

# =============================================================================
# 18. GITHUB REVISION NOTES
# =============================================================================

print("\n")
print("=" * 70)
print("GITHUB REVISION NOTES")
print("=" * 70)

print("""

Hyperparameter Tuning

↓

Improves model performance
by searching for the best
hyperparameter values.

--------------------------------------------------

Search Methods

✔ GridSearchCV

✔ RandomizedSearchCV

--------------------------------------------------

GridSearchCV

↓

Exhaustive Search

Every Combination

--------------------------------------------------

RandomizedSearchCV

↓

Random Sampling

Faster

Scalable

--------------------------------------------------

Pipeline

↓

Preprocessing

↓

Model

↓

Prediction

--------------------------------------------------

Pipeline Benefits

✔ Cleaner Code

✔ Reusable Workflow

✔ Prevents Data Leakage

✔ Easy Deployment

✔ Easy Cross Validation

--------------------------------------------------

Important Attributes

best_params_

↓

Best Hyperparameter Values

-----------------------------------

best_score_

↓

Best Cross Validation Score

-----------------------------------

best_estimator_

↓

Best Trained Model

""")

# =============================================================================
# 19. FINAL TAKEAWAYS
# =============================================================================

print("\n")
print("=" * 70)
print("FINAL TAKEAWAYS")
print("=" * 70)

takeaways = [

"Parameters are learned automatically during training.",

"Hyperparameters are chosen before training.",

"GridSearchCV searches every parameter combination.",

"RandomizedSearchCV samples random combinations.",

"Random Search is faster for large search spaces.",

"Pipeline combines preprocessing and modeling.",

"Pipelines prevent data leakage.",

"Always combine hyperparameter tuning with Cross Validation.",

"Never tune using the test set.",

"Professional ML workflows rely on Pipelines and systematic tuning."

]

for i, item in enumerate(takeaways, start=1):

    print(f"{i}. {item}")

# =============================================================================
# 20. WHAT I LEARNED TODAY
# =============================================================================

print("\n")
print("=" * 70)
print("WHAT I LEARNED TODAY")
print("=" * 70)

print("""

Today I learned

✔ Parameters

✔ Hyperparameters

✔ Search Space

✔ GridSearchCV

✔ RandomizedSearchCV

✔ Best Parameters

✔ Best Estimator

✔ Best Cross Validation Score

✔ Pipeline

✔ Data Leakage Prevention

✔ Production ML Workflow

✔ Systematic Model Optimization

Biggest Realization

A good model

isn't built by guessing
hyperparameters.

It's built by

defining a search strategy,

validating with

Cross Validation,

and selecting the model

that generalizes best.

""")

# =============================================================================
# 21. TOP ML ENGINEER INSIGHT
# =============================================================================

print("\n")
print("=" * 70)
print("TOP ML ENGINEER INSIGHT")
print("=" * 70)

print("""

Beginners ask

"What hyperparameters
should I use?"

Professionals ask

"How should I search
for the best hyperparameters?"

Optimization is a
systematic process,
not trial and error.

A reproducible Pipeline
combined with Cross Validation
and Hyperparameter Tuning
is the foundation of
production-ready Machine Learning.

""")

# =============================================================================
# 22. END OF DAY 34
# =============================================================================

print("\n")
print("=" * 70)
print("END OF DAY 34")
print("=" * 70)

print("Next Topic : End-to-End Machine Learning Project 🚀")

