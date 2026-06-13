# ==========================================
# DAY 15: VECTORS FOR MACHINE LEARNING
# ==========================================

import numpy as np

# ------------------------------------------
# 1. WHAT IS A VECTOR?
# ------------------------------------------
# A vector is simply a collection of numbers.
# In ML, one row of data is usually a vector.

student = np.array([5, 90, 8])

print("Student Vector:")
print(student)

# Meaning:
# [Hours Studied, Attendance %, Assignments]
# [5, 90, 8]


# ------------------------------------------
# 2. VECTOR ADDITION
# ------------------------------------------
v1 = np.array([1, 2])
v2 = np.array([3, 4])

print("\nVector Addition:")
print(v1 + v2)

# [1,2] + [3,4]
# = [4,6]


# ------------------------------------------
# 3. VECTOR SUBTRACTION
# ------------------------------------------
print("\nVector Subtraction:")
print(v2 - v1)

# [3,4] - [1,2]
# = [2,2]


# ------------------------------------------
# 4. SCALAR MULTIPLICATION
# ------------------------------------------
print("\nScalar Multiplication:")

vector = np.array([2, 3])

print(3 * vector)

# 3 × [2,3]
# = [6,9]


# ------------------------------------------
# 5. VECTOR MAGNITUDE
# ------------------------------------------
# Magnitude = Length of vector

v = np.array([3, 4])

magnitude = np.linalg.norm(v)

print("\nVector Magnitude:")
print(magnitude)

# √(3² + 4²)
# = √25
# = 5


# ------------------------------------------
# 6. DOT PRODUCT 
# ------------------------------------------
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print("\nDot Product:")
print(np.dot(a, b))

# (1×4)+(2×5)+(3×6)
# = 32


# ------------------------------------------
# 7. REAL ML EXAMPLE
# ------------------------------------------
# Features of a student

features = np.array([5, 90, 8])

# Model weights

weights = np.array([2, 0.1, 1])

prediction = np.dot(features, weights)

print("\nML Prediction Score:")
print(prediction)

# (5×2) + (90×0.1) + (8×1)
# = 10 + 9 + 8
# = 27


# ------------------------------------------
# 8. WHY VECTORS MATTER IN ML
# ------------------------------------------
print("\nML Insight:")
print("Every row in a dataset is a vector.")
print("Machine Learning models learn from vectors.")
print("Predictions are often made using vector operations.")