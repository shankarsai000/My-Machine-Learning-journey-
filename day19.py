# ==========================================
# DAY 19: CALCULUS INTUITION FOR ML
# ==========================================
import numpy as np
# ------------------------------------------
# 1. FUNCTION
# ------------------------------------------
# A function maps input -> output

def f(x):
    return x**2

print("Function Values")

for x in range(6):
    print(f"x={x}, y={f(x)}")

# ------------------------------------------
# 2. RATE OF CHANGE
# ------------------------------------------
# How much output changes
# when input changes

x1 = 2
x2 = 3

change_x = x2 - x1

change_y = f(x2) - f(x1)

print("\nChange in X:", change_x)
print("Change in Y:", change_y)

# slope = change_y/change_x
# ------------------------------------------
# 3. SLOPE
# ------------------------------------------

slope = change_y / change_x

print("\nSlope:")
print(slope)

# Calculus studies slopes
# ------------------------------------------
# 4. SIMPLE LINEAR FUNCTION
# ------------------------------------------

def line(x):
    return 2*x + 1

print("\nLinear Function")

for x in range(5):
    print(line(x))

# slope always = 2
# ------------------------------------------
# 5. NUMERICAL DERIVATIVE
# ------------------------------------------
# Approximate derivative

def derivative(func, x, h=0.0001):
    return (func(x+h) - func(x))/h

print("\nDerivative at x=2")

print(
    derivative(f, 2)
)

# Expected ≈ 4
# ------------------------------------------
# 6. DERIVATIVE AT DIFFERENT POINTS
# ------------------------------------------

print("\nDerivatives")

for x in [1,2,3,4]:
    print(
        x,
        derivative(f,x)
    )
# ------------------------------------------
# 7. LOSS FUNCTION IDEA
# ------------------------------------------

actual = 100

prediction = 130

loss = (actual-prediction)**2

print("\nLoss")

print(loss)

# ML tries to minimize this
# ------------------------------------------
# 8. BETTER PREDICTION
# ------------------------------------------

prediction2 = 110

loss2 = (actual-prediction2)**2

print("\nNew Loss")

print(loss2)

# Smaller loss = better model
# ------------------------------------------
# 9. COST LANDSCAPE
# ------------------------------------------

weights = np.arange(
    -5,
    6
)

print("\nWeight vs Cost")

for w in weights:

    cost = (w-2)**2

    print(
        f"Weight={w}, Cost={cost}"
    )

# Minimum cost at w=2
# ------------------------------------------
# 10. OPTIMIZATION IDEA
# ------------------------------------------

print("\nOptimization Insight")

print(
    "Goal: Find minimum cost"
)

print(
    "Calculus tells direction"
)

print(
    "Gradient Descent follows that direction"
)
# ------------------------------------------
# 11. MOUNTAIN ANALOGY
# ------------------------------------------

print("\nMountain Analogy")

print(
    "Top of mountain = High Error"
)

print(
    "Bottom of valley = Low Error"
)

print(
    "Gradient tells where to walk"
)
# ------------------------------------------
# 12. ML CONNECTION
# ------------------------------------------

print("\nML Connection")

print(
    "Derivative = Slope"
)

print(
    "Slope = Direction"
)

print(
    "Direction = Learning"
)

print(
    "Learning = Better Predictions"
)
# ------------------------------------------
# 13. REAL EXAMPLE
# ------------------------------------------

hours = np.array(
    [1,2,3,4,5]
)

marks = np.array(
    [20,40,60,80,100]
)

print("\nDataset")

for h,m in zip(hours,marks):
    print(
        h,m
    )
# Linear Regression learns
# relationship using calculus
# ------------------------------------------
# 14. WHY CALCULUS MATTERS
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
    "Calculus teaches machines how to improve."
)