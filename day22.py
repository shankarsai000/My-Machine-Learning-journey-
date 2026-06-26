# ==========================================
# DAY 22: ML WORKFLOW & TRAIN-TEST SPLIT
# ==========================================
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
# ------------------------------------------
# 1. LOAD DATASET
# ------------------------------------------
iris = load_iris()

X = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)
y = iris.target
print("First Five Rows")
print(X.head())
# ------------------------------------------
# 2. FEATURES & TARGET
# ------------------------------------------
print("\nFeatures")
print(list(X.columns))

print("\nTarget")
print("Flower Species")
# ------------------------------------------
# 3. DATASET SHAPE
# ------------------------------------------
# print("\nDataset Shape")
print(X.shape)
# ------------------------------------------
# 4. TRAIN TEST SPLIT
# ------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    shuffle=True
)
print("\nTraining Samples")
print(len(X_train))
print("\nTesting Samples")
print(len(X_test))
# ------------------------------------------
# 5. WHY RANDOM STATE?
# ------------------------------------------
print("\nRandom State")
print("Keeps results reproducible.")
# ------------------------------------------
# 6. CHECK SPLIT
# ------------------------------------------
print("\nTraining Features")
print(X_train.head())
print("\nTesting Features")
print(X_test.head())
# ------------------------------------------
# 7. DATA LEAKAGE (WRONG WAY)
# ------------------------------------------
print("\nWrong Workflow")
print("""
Load Data
↓

Scale Entire Dataset ❌

↓

Split Data

(Model already saw future data)
""")
# ------------------------------------------
# 8. CORRECT WORKFLOW
# ------------------------------------------
print("\nCorrect Workflow")
print("""
Load Data

↓

Train-Test Split

↓

Fit Scaler on Train

↓

Transform Test

↓

Train Model
""")
# ------------------------------------------
# 9. ML PIPELINE
# ------------------------------------------

steps = [
    "Problem Framing",
    "Collect Data",
    "EDA",
    "Cleaning",
    "Feature Engineering",
    "Train-Test Split",
    "Training",
    "Evaluation",
    "Deployment"
]

print("\nML Workflow")

for step in steps:
    print("✔", step)
# ------------------------------------------
# 10. FEATURES VS TARGET
# ------------------------------------------
print("\nFeatures = Inputs")
print(X.columns.tolist())
print("\nTarget = Output")
print("Flower Species")
# ------------------------------------------
# 11. WHY SPLIT DATA?
# ------------------------------------------
print("\nReason")
print("Train → Learn")
print("Test → Evaluate")
print("Avoid Overfitting")
# ------------------------------------------
# 12. INDUSTRY INSIGHT
# ------------------------------------------
print("\nTop ML Engineer Insight")
print("A great model trained on bad data")
print("is still a bad model.")
# ------------------------------------------
# 13. FINAL TAKEAWAY
# ------------------------------------------
print("\nToday's Learning")
print("Every ML project starts")
print("with understanding the problem,")
print("not choosing the algorithm.")