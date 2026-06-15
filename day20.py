# ==========================================
# DAY 20: GRADIENT DESCENT & LINEAR REGRESSION
# ==========================================

import numpy as np

# ------------------------------------------
# 1. DATASET
# ------------------------------------------

# Hours Studied
X = np.array([1,2,3,4,5])

# Marks Obtained
y = np.array([20,40,60,80,100])

print("Dataset")

for h,m in zip(X,y):
    print(f"Hours={h}, Marks={m}")


# ------------------------------------------
# 2. LINEAR REGRESSION EQUATION
# ------------------------------------------
# y = mx + b

m = 0
b = 0

print("\nInitial Parameters")
print("m =",m)
print("b =",b)


# ------------------------------------------
# 3. PREDICTION FUNCTION
# ------------------------------------------

def predict(x,m,b):
    return m*x+b

print("\nPrediction Example")
print(
    predict(5,m,b)
)


# ------------------------------------------
# 4. LOSS FUNCTION
# ------------------------------------------
# Mean Squared Error

predictions = predict(X,m,b)

loss = np.mean(
    (y-predictions)**2
)

print("\nInitial Loss")
print(loss)


# ------------------------------------------
# 5. GRADIENT DESCENT IDEA
# ------------------------------------------

learning_rate = 0.01

print("\nLearning Rate")
print(learning_rate)

# controls step size


# ------------------------------------------
# 6. TRAINING LOOP
# ------------------------------------------

epochs = 1000

for epoch in range(epochs):

    predictions = m*X+b

    dm = (-2/len(X)) * np.sum(
        X*(y-predictions)
    )

    db = (-2/len(X)) * np.sum(
        y-predictions
    )

    m = m-learning_rate*dm

    b = b-learning_rate*db


# ------------------------------------------
# 7. LEARNED PARAMETERS
# ------------------------------------------

print("\nLearned Parameters")

print("Slope (m) =", round(m,2))
print("Intercept (b) =", round(b,2))


# ------------------------------------------
# 8. FINAL PREDICTIONS
# ------------------------------------------

predictions = m*X+b

print("\nPredictions")

print(predictions)


# ------------------------------------------
# 9. FINAL LOSS
# ------------------------------------------

final_loss = np.mean(
    (y-predictions)**2
)

print("\nFinal Loss")

print(round(final_loss,5))


# ------------------------------------------
# 10. FUTURE PREDICTION
# ------------------------------------------

hours = 6

future_marks = m*hours+b

print("\nPrediction for 6 Hours Study")

print(round(future_marks,2))


# ------------------------------------------
# 11. UNDERSTANDING SLOPE
# ------------------------------------------

print("\nSlope Interpretation")

print(
    "Slope tells how much marks increase"
)

print(
    "when study hours increase by 1"
)


# ------------------------------------------
# 12. UNDERSTANDING INTERCEPT
# ------------------------------------------

print("\nIntercept Interpretation")

print(
    "Starting value of model"
)


# ------------------------------------------
# 13. WHY GRADIENT DESCENT WORKS
# ------------------------------------------

print("\nGradient Descent Insight")

print(
    "Find error"
)

print(
    "Calculate gradient"
)

print(
    "Update parameters"
)

print(
    "Repeat"
)

# ------------------------------------------
# 14. REAL ML CONNECTION
# ------------------------------------------

print("\nUsed In")

print("Linear Regression")
print("Logistic Regression")
print("Neural Networks")
print("Deep Learning")


# ------------------------------------------
# 15. FINAL TAKEAWAY
# ------------------------------------------

print("\nToday's Learning")

print(
    "Gradient Descent teaches models how to learn."
)

print(
    "Linear Regression is the first learning model."
)