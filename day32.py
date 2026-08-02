# =============================================================================
# DAY 32 : NAIVE BAYES
# PART 4.1 : GAUSSIAN NAIVE BAYES
# =============================================================================

"""
Topics Covered

1. Load Dataset
2. Train-Test Split
3. Feature Scaling
4. Train Gaussian Naive Bayes
5. Predictions
6. Prediction Probabilities
7. Accuracy
8. Precision
9. Recall
10. F1 Score
11. ROC-AUC
12. Confusion Matrix
13. Classification Report
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

from sklearn.naive_bayes import GaussianNB

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

X_train = scaler.fit_transform(

    X_train

)

X_test = scaler.transform(

    X_test

)

print("Feature Scaling Completed!")

# =============================================================================
# 4. TRAIN GAUSSIAN NAIVE BAYES
# =============================================================================

print("\n")
print("="*70)
print("TRAINING GAUSSIAN NAIVE BAYES")
print("="*70)

gnb = GaussianNB()

gnb.fit(

    X_train,

    y_train

)

print("Model Trained Successfully!")

# =============================================================================
# 5. PREDICTIONS
# =============================================================================

pred = gnb.predict(

    X_test

)

prob = gnb.predict_proba(

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
# 9. CLASS PROBABILITIES
# =============================================================================

print("\n")
print("="*70)
print("CLASS PRIOR PROBABILITIES")
print("="*70)

print(pd.DataFrame({

    "Class":data.target_names,

    "Prior Probability":gnb.class_prior_

}))

# =============================================================================
# 10. NEW SAMPLE PREDICTION
# =============================================================================

new_sample = X.iloc[[0]]

new_scaled = scaler.transform(

    new_sample

)

prediction = gnb.predict(

    new_scaled

)

probability = gnb.predict_proba(

    new_scaled

)

print("\n")
print("="*70)
print("NEW SAMPLE PREDICTION")
print("="*70)

print("Predicted Class :", prediction[0])

print("Prediction Probabilities")

print(np.round(probability,4))

# =============================================================================
# 11. MODEL SUMMARY
# =============================================================================

print("\n")
print("="*70)
print("MODEL SUMMARY")
print("="*70)

print("""

Algorithm

↓

Gaussian Naive Bayes

Learning Style

↓

Probability Based

Uses

↓

Bayes Theorem

Feature Type

↓

Continuous Numerical Features

Examples

↓

Height

Weight

Temperature

Age

Prediction

↓

Highest Posterior Probability

""")

# =============================================================================
# 12. ADVANTAGES
# =============================================================================

print("\n")
print("="*70)
print("ADVANTAGES")
print("="*70)

advantages = [

"Very Fast Training",

"Very Fast Prediction",

"Works Well on Small Datasets",

"Easy to Interpret",

"Produces Probabilities",

"Strong Baseline Model"

]

for item in advantages:

    print("✔", item)

# =============================================================================
# 13. LIMITATIONS
# =============================================================================

print("\n")
print("="*70)
print("LIMITATIONS")
print("="*70)

limitations = [

"Assumes Feature Independence",

"Performance Drops for Correlated Features",

"Needs Gaussian Distribution Assumption",

"May Underperform Complex Models"

]

for item in limitations:

    print("✘", item)

# =============================================================================
# 14. TOP ML ENGINEER INSIGHT
# =============================================================================

print("\n")
print("="*70)
print("TOP ML ENGINEER INSIGHT")
print("="*70)

print("""

Gaussian Naive Bayes

is often used as

the FIRST baseline model

because

✔ Extremely Fast

✔ Easy to Train

✔ Requires Almost No Hyperparameter Tuning

✔ Produces Probabilities

If a simple model performs well,

there may be no need

to deploy

a much more complex model.

Always benchmark

Naive Bayes

against

Logistic Regression

Decision Tree

Random Forest

before choosing

a production model.

""")

# =============================================================================
# DAY 32 : NAIVE BAYES
# PART 4.2 : SPAM DETECTION USING MULTINOMIAL NAIVE BAYES
# =============================================================================

"""
Topics Covered

1. Create Spam Dataset
2. CountVectorizer
3. Train-Test Split
4. Train Multinomial Naive Bayes
5. Predictions
6. Prediction Probabilities
7. Accuracy
8. Confusion Matrix
9. Classification Report
10. New Email Prediction
11. Most Frequent Words
12. Top ML Engineer Insight

"""

# =============================================================================
# IMPORT LIBRARIES
# =============================================================================

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import (

    accuracy_score,

    confusion_matrix,

    classification_report

)

# =============================================================================
# 1. CREATE SAMPLE DATASET
# =============================================================================

print("="*70)
print("CREATING SPAM DATASET")
print("="*70)

emails = [

    "Congratulations you won a free iPhone",

    "Claim your free lottery prize now",

    "Win money click here",

    "Limited time offer buy now",

    "Meeting is scheduled for tomorrow",

    "Please submit your assignment",

    "Let's have lunch today",

    "Project deadline is next week",

    "Exclusive cash reward waiting",

    "Your account has been updated",

    "Free vacation tickets available",

    "Join the team meeting"

]

labels = [

    1,

    1,

    1,

    1,

    0,

    0,

    0,

    0,

    1,

    0,

    1,

    0

]

dataset = pd.DataFrame({

    "Email": emails,

    "Spam": labels

})

print(dataset)

# =============================================================================
# 2. TEXT TO NUMBERS
# =============================================================================

print("\n")
print("="*70)
print("COUNTVECTORIZER")
print("="*70)

vectorizer = CountVectorizer()

X = vectorizer.fit_transform(

    dataset["Email"]

)

y = dataset["Spam"]

print("Vocabulary Size :", len(vectorizer.vocabulary_))

print("\nFirst 15 Words")

print(list(vectorizer.vocabulary_.keys())[:15])

# =============================================================================
# 3. TRAIN TEST SPLIT
# =============================================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.25,

    random_state=42,

    stratify=y

)

# =============================================================================
# 4. TRAIN MODEL
# =============================================================================

print("\n")
print("="*70)
print("TRAINING MULTINOMIAL NAIVE BAYES")
print("="*70)

model = MultinomialNB()

model.fit(

    X_train,

    y_train

)

print("Model Trained Successfully!")

# =============================================================================
# 5. PREDICTIONS
# =============================================================================

pred = model.predict(

    X_test

)

prob = model.predict_proba(

    X_test

)

print("\nPredictions")

print(pred)

print("\nPrediction Probabilities")

print(np.round(prob,4))

# =============================================================================
# 6. MODEL EVALUATION
# =============================================================================

accuracy = accuracy_score(

    y_test,

    pred

)

print("\n")
print("="*70)
print("MODEL EVALUATION")
print("="*70)

print("Accuracy :", round(accuracy,4))

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

        target_names=[

            "Not Spam",

            "Spam"

        ]

    )

)

# =============================================================================
# 9. NEW EMAIL PREDICTION
# =============================================================================

new_email = [

    "Congratulations you have won free cash prize"

]

new_vector = vectorizer.transform(

    new_email

)

prediction = model.predict(

    new_vector

)

probability = model.predict_proba(

    new_vector

)

print("\n")
print("="*70)
print("NEW EMAIL PREDICTION")
print("="*70)

print("Email")

print(new_email[0])

print("\nPrediction")

print(

    "Spam"

    if prediction[0] == 1

    else "Not Spam"

)

print("\nProbability")

print(np.round(probability,4))

# =============================================================================
# 10. WORD FREQUENCY
# =============================================================================

print("\n")
print("="*70)
print("VOCABULARY")
print("="*70)

vocab = pd.DataFrame({

    "Word":

    vectorizer.get_feature_names_out()

})

print(vocab.head(20))

# =============================================================================
# 11. MODEL SUMMARY
# =============================================================================

print("\n")
print("="*70)
print("MODEL SUMMARY")
print("="*70)

print("""

Algorithm

↓

Multinomial Naive Bayes

Best For

↓

Text Classification

Spam Detection

News Classification

Sentiment Analysis

Input

↓

Word Counts

Output

↓

Probability

↓

Predicted Class

""")

# =============================================================================
# 12. TOP ML ENGINEER INSIGHT
# =============================================================================

print("\n")
print("="*70)
print("TOP ML ENGINEER INSIGHT")
print("="*70)

print("""

Multinomial Naive Bayes

is one of the strongest

baseline algorithms

for NLP.

Before training

large Transformer models

(BERT, RoBERTa, Llama),

many ML engineers first build

a Multinomial Naive Bayes model.

Why?

✔ Trains in seconds

✔ Very little memory

✔ Easy to interpret

✔ Strong baseline

If Naive Bayes already performs well,

a more complex model

may not justify

its additional cost.

""")

# =============================================================================
# DAY 32 : NAIVE BAYES
# PART 4.3 : MODEL COMPARISON & BENCHMARKING
# =============================================================================

"""
Topics Covered

1. Logistic Regression
2. Gaussian Naive Bayes
3. Training Time Comparison
4. Prediction Time Comparison
5. Accuracy Comparison
6. Precision Comparison
7. Recall Comparison
8. F1 Score Comparison
9. ROC-AUC Comparison
10. Probability Comparison
11. Model Benchmarking
12. Top ML Engineer Insight

"""

# =============================================================================
# IMPORT LIBRARIES
# =============================================================================

import time
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import (

    accuracy_score,

    precision_score,

    recall_score,

    f1_score,

    roc_auc_score

)

# =============================================================================
# 1. TRAIN LOGISTIC REGRESSION
# =============================================================================

print("="*70)
print("TRAINING LOGISTIC REGRESSION")
print("="*70)

start = time.time()

log_model = LogisticRegression(

    max_iter=1000,

    random_state=42

)

log_model.fit(

    X_train,

    y_train

)

log_training_time = time.time() - start

# =============================================================================
# 2. TRAIN GAUSSIAN NAIVE BAYES
# =============================================================================

print("\n")
print("="*70)
print("TRAINING GAUSSIAN NAIVE BAYES")
print("="*70)

start = time.time()

nb_model = GaussianNB()

nb_model.fit(

    X_train,

    y_train

)

nb_training_time = time.time() - start

# =============================================================================
# 3. PREDICTIONS
# =============================================================================

log_pred = log_model.predict(

    X_test

)

nb_pred = nb_model.predict(

    X_test

)

log_prob = log_model.predict_proba(

    X_test

)[:,1]

nb_prob = nb_model.predict_proba(

    X_test

)[:,1]

# =============================================================================
# 4. TRAINING TIME
# =============================================================================

training = pd.DataFrame({

    "Model":[

        "Logistic Regression",

        "Gaussian Naive Bayes"

    ],

    "Training Time (sec)":[

        round(log_training_time,6),

        round(nb_training_time,6)

    ]

})

print("\n")
print("="*70)
print("TRAINING TIME")
print("="*70)

print(training)

# =============================================================================
# 5. PREDICTION TIME
# =============================================================================

start = time.time()

log_model.predict(X_test)

log_prediction_time = time.time() - start

start = time.time()

nb_model.predict(X_test)

nb_prediction_time = time.time() - start

prediction = pd.DataFrame({

    "Model":[

        "Logistic Regression",

        "Gaussian Naive Bayes"

    ],

    "Prediction Time (sec)":[

        round(log_prediction_time,6),

        round(nb_prediction_time,6)

    ]

})

print("\nPrediction Time")

print(prediction)

# =============================================================================
# 6. PERFORMANCE COMPARISON
# =============================================================================

comparison = pd.DataFrame({

    "Model":[

        "Logistic Regression",

        "Gaussian Naive Bayes"

    ],

    "Accuracy":[

        accuracy_score(y_test, log_pred),

        accuracy_score(y_test, nb_pred)

    ],

    "Precision":[

        precision_score(y_test, log_pred),

        precision_score(y_test, nb_pred)

    ],

    "Recall":[

        recall_score(y_test, log_pred),

        recall_score(y_test, nb_pred)

    ],

    "F1 Score":[

        f1_score(y_test, log_pred),

        f1_score(y_test, nb_pred)

    ],

    "ROC-AUC":[

        roc_auc_score(y_test, log_prob),

        roc_auc_score(y_test, nb_prob)

    ]

})

comparison = comparison.round(4)

print("\n")
print("="*70)
print("MODEL COMPARISON")
print("="*70)

print(comparison)

# =============================================================================
# 7. ACCURACY COMPARISON GRAPH
# =============================================================================

plt.figure(figsize=(8,5))

plt.bar(

    comparison["Model"],

    comparison["Accuracy"]

)

plt.title("Accuracy Comparison")

plt.ylabel("Accuracy")

plt.grid(True)

plt.show()

# =============================================================================
# 8. ROC-AUC COMPARISON GRAPH
# =============================================================================

plt.figure(figsize=(8,5))

plt.bar(

    comparison["Model"],

    comparison["ROC-AUC"]

)

plt.title("ROC-AUC Comparison")

plt.ylabel("ROC-AUC")

plt.grid(True)

plt.show()

# =============================================================================
# 9. SAMPLE PROBABILITY COMPARISON
# =============================================================================

probability = pd.DataFrame({

    "Actual": y_test[:5].values,

    "Logistic Probability":

        log_prob[:5].round(4),

    "Naive Bayes Probability":

        nb_prob[:5].round(4)

})

print("\n")
print("="*70)
print("PREDICTION PROBABILITIES")
print("="*70)

print(probability)

# =============================================================================
# 10. MODEL BENCHMARK
# =============================================================================

print("\n")
print("="*70)
print("MODEL BENCHMARK")
print("="*70)

print("""

Logistic Regression

↓

Learns Linear Decision Boundary

Needs Optimization

Higher Training Time

Better for Correlated Features

------------------------------------------------

Naive Bayes

↓

Learns Probabilities

No Optimization

Extremely Fast

Best for NLP

Strong Baseline Model

""")

# =============================================================================
# 11. WHEN TO USE WHICH?
# =============================================================================

print("\n")
print("="*70)
print("WHEN TO USE WHICH MODEL?")
print("="*70)

print("""

Choose Logistic Regression

✔ Correlated Features

✔ Better Probability Calibration

✔ Higher Predictive Performance

------------------------------------------------

Choose Naive Bayes

✔ NLP

✔ Spam Detection

✔ Small Datasets

✔ Fast Training

✔ Fast Prediction

✔ Limited Computing Resources

""")

# =============================================================================
# 12. TOP ML ENGINEER INSIGHT
# =============================================================================

print("\n")
print("="*70)
print("TOP ML ENGINEER INSIGHT")
print("="*70)

print("""

Production ML is not about
using the most complex model.

It is about choosing the model
that provides the best balance of

✔ Accuracy

✔ Speed

✔ Memory Usage

✔ Scalability

✔ Interpretability

✔ Business Value

Always benchmark

Naive Bayes

against

Logistic Regression

before moving to
larger deep learning models.

A simple model that achieves
95% accuracy in milliseconds
may be more valuable than a
complex model achieving 96%.

""")

# =============================================================================
# DAY 32 : NAIVE BAYES
# PART 4.4 : FINAL NOTES, INTERVIEW PREP & GITHUB SUMMARY
# =============================================================================

# =============================================================================
# 13. PRACTICE EXERCISES
# =============================================================================

print("\n")
print("="*70)
print("PRACTICE EXERCISES")
print("="*70)

print("""

Beginner

1. Train Gaussian Naive Bayes
   on the Breast Cancer Dataset.

------------------------------------------------------------

2. Train Multinomial Naive Bayes
   on a Text Dataset.

------------------------------------------------------------

Intermediate

3. Compare

Gaussian NB

vs

Logistic Regression

using

Accuracy

Precision

Recall

F1 Score

ROC-AUC

------------------------------------------------------------

4. Compare

Training Time

Prediction Time

Memory Usage

------------------------------------------------------------

Advanced

5. Download a real Spam Dataset

Compare

Multinomial NB

TF-IDF + Logistic Regression

Support Vector Machine

------------------------------------------------------------

6. Compare

CountVectorizer

vs

TF-IDF Vectorizer

Observe

Accuracy

Training Time

Prediction Time

""")

# =============================================================================
# 14. MINI CHALLENGE
# =============================================================================

print("\n")
print("="*70)
print("MINI CHALLENGE")
print("="*70)

print("""

Build an Email Spam Detector

Pipeline

✔ Load Dataset

✔ Text Cleaning

✔ CountVectorizer

✔ Train-Test Split

✔ Multinomial Naive Bayes

✔ Predictions

✔ Prediction Probabilities

✔ Confusion Matrix

✔ Classification Report

✔ ROC-AUC

Bonus

Compare with

Logistic Regression

Support Vector Machine

Explain WHY
one model performs better.

""")

# =============================================================================
# 15. COMMON INTERVIEW QUESTIONS
# =============================================================================

print("\n")
print("="*70)
print("COMMON INTERVIEW QUESTIONS")
print("="*70)

questions = [

"What is Naive Bayes?",

"Why is it called Naive Bayes?",

"What is Bayes Theorem?",

"What is Prior Probability?",

"What is Posterior Probability?",

"What is Likelihood?",

"What is Conditional Probability?",

"What is the Independence Assumption?",

"Why does Naive Bayes work despite unrealistic assumptions?",

"What is Gaussian Naive Bayes?",

"What is Multinomial Naive Bayes?",

"What is Bernoulli Naive Bayes?",

"What is Complement Naive Bayes?",

"When should you use Gaussian NB?",

"When should you use Multinomial NB?",

"When should you use Bernoulli NB?",

"Difference between Gaussian and Multinomial NB?",

"Naive Bayes vs Logistic Regression?",

"Why is Naive Bayes so fast?",

"Why is Naive Bayes popular in NLP?",

"What are the limitations of Naive Bayes?",

"Can Naive Bayes handle correlated features?",

"Does Naive Bayes require feature scaling?",

"Why is Naive Bayes considered a strong baseline model?"

]

for q in questions:

    print("✔", q)

# =============================================================================
# 16. SENIOR ML ENGINEER NOTES
# =============================================================================

print("\n")
print("="*70)
print("SENIOR ML ENGINEER NOTES")
print("="*70)

print("""

Naive Bayes predicts

Probabilities

NOT

Decision Boundaries.

--------------------------------------------------

Gaussian NB

↓

Continuous Features

Examples

Age

Height

Weight

Temperature

--------------------------------------------------

Multinomial NB

↓

Word Counts

Emails

Documents

News Articles

--------------------------------------------------

Bernoulli NB

↓

Binary Features

Yes / No

Present / Absent

--------------------------------------------------

Complement NB

↓

Imbalanced Text Classification

--------------------------------------------------

Naive Bayes

usually requires

almost NO

hyperparameter tuning.

--------------------------------------------------

Naive Bayes

trains extremely fast.

Prediction is also

very fast.

--------------------------------------------------

Feature Scaling

↓

Usually NOT required.

--------------------------------------------------

Naive Bayes often becomes

the FIRST baseline

for NLP tasks.

Always compare

Naive Bayes

with

Logistic Regression

before moving to

Deep Learning.

""")

# =============================================================================
# 17. REAL-WORLD APPLICATIONS
# =============================================================================

print("\n")
print("="*70)
print("REAL-WORLD APPLICATIONS")
print("="*70)

applications = [

"Spam Detection",

"Email Filtering",

"Sentiment Analysis",

"News Classification",

"Language Detection",

"Document Categorization",

"Search Ranking",

"Chatbot Intent Classification",

"Recommendation Systems (Simple Baselines)",

"Medical Diagnosis"

]

for app in applications:

    print("✔", app)

# =============================================================================
# 18. GITHUB REVISION NOTES
# =============================================================================

print("\n")
print("="*70)
print("GITHUB REVISION NOTES")
print("="*70)

print("""

Naive Bayes

↓

Probability-based
Classification Algorithm

--------------------------------------------------

Based On

↓

Bayes Theorem

--------------------------------------------------

Core Concepts

✔ Prior Probability

✔ Likelihood

✔ Posterior Probability

✔ Conditional Probability

✔ Independence Assumption

--------------------------------------------------

Types

✔ Gaussian NB

✔ Multinomial NB

✔ Bernoulli NB

✔ Complement NB

--------------------------------------------------

Best Use Cases

Gaussian

↓

Continuous Data

Multinomial

↓

Text Classification

Bernoulli

↓

Binary Features

Complement

↓

Imbalanced Text

--------------------------------------------------

Advantages

✔ Very Fast

✔ Simple

✔ Excellent NLP Baseline

✔ Low Memory Usage

✔ Works Well with Small Datasets

--------------------------------------------------

Limitations

✔ Strong Independence Assumption

✔ Sensitive to Correlated Features

✔ Lower Accuracy on Complex Problems

✔ Gaussian Assumption for Continuous Data

""")

# =============================================================================
# 19. FINAL TAKEAWAYS
# =============================================================================

print("\n")
print("="*70)
print("FINAL TAKEAWAYS")
print("="*70)

takeaways = [

"Naive Bayes is a probability-based classifier.",

"It is built on Bayes Theorem.",

"It assumes feature independence.",

"Despite the assumption, it performs surprisingly well.",

"Gaussian NB works with continuous features.",

"Multinomial NB is excellent for NLP.",

"Bernoulli NB handles binary features.",

"Complement NB improves performance on imbalanced text.",

"Naive Bayes trains and predicts extremely fast.",

"It is often the first baseline model for text classification.",

"Simple models can sometimes outperform complex ones."

]

for i, item in enumerate(takeaways, start=1):

    print(f"{i}. {item}")

# =============================================================================
# 20. WHAT I LEARNED TODAY
# =============================================================================

print("\n")
print("="*70)
print("WHAT I LEARNED TODAY")
print("="*70)

print("""

Today I learned

✔ Bayes Theorem

✔ Prior Probability

✔ Posterior Probability

✔ Likelihood

✔ Conditional Probability

✔ Independence Assumption

✔ Gaussian Naive Bayes

✔ Multinomial Naive Bayes

✔ Bernoulli Naive Bayes

✔ Complement Naive Bayes

✔ Spam Detection

✔ CountVectorizer

✔ Probability Prediction

✔ Model Benchmarking

✔ Production Best Practices

Biggest Realization

Naive Bayes

doesn't memorize patterns.

Instead,

it calculates

the probability

of every class

and selects

the one with

the highest probability.

Sometimes,

a simple probabilistic model

can compete with

much more complex algorithms.

""")

# =============================================================================
# 21. END OF DAY 32
# =============================================================================

print("\n")
print("="*70)
print("END OF DAY 32")
print("="*70)

print("Next Topic : Cross Validation & Stratified K-Fold Cross Validation 🚀")
