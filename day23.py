# ==========================================
# DAY 23: FEATURE SCALING & LINEAR REGRESSION
# ==========================================
import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
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

print("\nFeatures")
print(X.columns.tolist())
# ------------------------------------------
# 2. TRAIN TEST SPLIT
# ------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))
# ------------------------------------------
# 3. STANDARD SCALER
# ------------------------------------------

standard = StandardScaler()

X_train_std = standard.fit_transform(X_train)
X_test_std = standard.transform(X_test)

print("\nStandard Scaled Data")
print(X_train_std[:3])

# Mean ≈ 0
# Standard Deviation ≈ 1
# ------------------------------------------
# 4. MINMAX SCALER
# ------------------------------------------
minmax = MinMaxScaler()
X_train_mm = minmax.fit_transform(X_train)
X_test_mm = minmax.transform(X_test)

print("\nMinMax Scaled Data")
print(X_train_mm[:3])
# Values between 0 and 1
# ------------------------------------------
# 5. TRAIN LINEAR REGRESSION
# ------------------------------------------
model = LinearRegression()
model.fit(X_train_std, y_train)
print("\nModel Trained Successfully!")
# ------------------------------------------
# 6. MODEL COEFFICIENTS
# ------------------------------------------
print("\nFeature Coefficients")

for feature, coef in zip(
        X.columns,
        model.coef_):
    print(feature, ":", round(coef,2))

# ------------------------------------------
# 7. INTERCEPT
# ------------------------------------------
print("\nIntercept")
print(round(model.intercept_,2))
# ------------------------------------------
# 8. PREDICTIONS
# ------------------------------------------
predictions = model.predict(X_test_std)
print("\nFirst Five Predictions")
print(predictions[:5])
# ------------------------------------------
# 9. MODEL EVALUATION
# ------------------------------------------
mae = mean_absolute_error(
    y_test,
    predictions
)
mse = mean_squared_error(
    y_test,
    predictions
)
r2 = r2_score(
    y_test,
    predictions
)
print("\nEvaluation Metrics")

print("MAE :", round(mae,2))
print("MSE :", round(mse,2))
print("R²  :", round(r2,3))
# ------------------------------------------
# 10. ACTUAL VS PREDICTED
# ------------------------------------------

results = pd.DataFrame({
    "Actual": y_test[:10],
    "Predicted": predictions[:10].round(2)
})
print("\nActual vs Predicted")
print(results)
# ------------------------------------------
# 11. NEW PATIENT PREDICTION
# ------------------------------------------
new_patient = X.iloc[[0]]

new_patient_scaled = standard.transform(new_patient)

prediction = model.predict(
    new_patient_scaled
)

print("\nPrediction for New Patient")

print(round(prediction[0],2))
# ------------------------------------------
# 12. WHY SCALING MATTERS
# ------------------------------------------
print("\nScaling Insight")
print("Without scaling:")
print("Large features dominate learning.")
print("\nWith scaling:")
print("Every feature contributes fairly.")

# ------------------------------------------
# 13. INDUSTRY INSIGHT
# ------------------------------------------
print("\nTop ML Engineer Insight")
print("Never fit the scaler")
print("on the entire dataset.")
print("Fit only on training data.")
# ------------------------------------------
# 14. MODELS THAT NEED SCALING
# ------------------------------------------
print("\nNeeds Scaling")

models = [
    "Logistic Regression",
    "KNN",
    "SVM",
    "Neural Networks",
    "PCA"
]

for model_name in models:
    print("✔", model_name)
# ------------------------------------------
# 15. FINAL TAKEAWAY
# ------------------------------------------
print("\nToday's Learning")
print("Scaling improves learning.")
print("Linear Regression learns")
print("the relationship between")
print("features and target.")