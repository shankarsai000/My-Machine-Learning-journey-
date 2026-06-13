# ==========================================
# DAY 16: MATRICES & DOT PRODUCT FOR ML
# ==========================================

import numpy as np

# ------------------------------------------
# 1. WHAT IS A MATRIX?
# ------------------------------------------
# A matrix is a collection of vectors.

A = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

print("Matrix A:")
print(A)


# ------------------------------------------
# 2. MATRIX SHAPE
# ------------------------------------------
print("\nMatrix Shape:")
print(A.shape)

# 2 rows × 3 columns


# ------------------------------------------
# 3. MATRIX ADDITION
# ------------------------------------------
B = np.array([
    [10, 20, 30],
    [40, 50, 60]
])

print("\nMatrix Addition:")
print(A + B)

# Element-wise addition


# ------------------------------------------
# 4. MATRIX SUBTRACTION
# ------------------------------------------
print("\nMatrix Subtraction:")
print(B - A)


# ------------------------------------------
# 5. SCALAR MULTIPLICATION
# ------------------------------------------
print("\nScalar Multiplication:")
print(2 * A)

# Multiply every element by 2


# ------------------------------------------
# 6. MATRIX TRANSPOSE
# ------------------------------------------
print("\nTranspose of A:")
print(A.T)

# Rows become columns


# ------------------------------------------
# 7. MATRIX MULTIPLICATION
# ------------------------------------------
M1 = np.array([
    [1, 2],
    [3, 4]
])

M2 = np.array([
    [5, 6],
    [7, 8]
])

print("\nMatrix Multiplication:")
print(M1 @ M2)

# Core operation behind Neural Networks


# ------------------------------------------
# 8. DOT PRODUCT
# ------------------------------------------
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

dot_product = np.dot(v1, v2)

print("\nDot Product:")
print(dot_product)

# (1×4)+(2×5)+(3×6)=32


# ------------------------------------------
# 9. REAL ML EXAMPLE
# ------------------------------------------
# Features of a student

features = np.array([
    5,    # Hours Studied
    90,   # Attendance
    8     # Assignments
])

# Model Weights

weights = np.array([
    2,
    0.1,
    1
])

prediction = np.dot(features, weights)

print("\nPrediction Score:")
print(prediction)

# (5×2)+(90×0.1)+(8×1)
# = 10+9+8
# = 27


# ------------------------------------------
# 10. DATASET AS MATRIX
# ------------------------------------------
students = np.array([
    [5, 90, 8],
    [3, 80, 6],
    [7, 95, 9]
])

print("\nStudent Dataset Matrix:")
print(students)

# Each row = one student
# Each row = one vector


# ------------------------------------------
# 11. MULTIPLE PREDICTIONS
# ------------------------------------------
scores = students @ weights

print("\nPredictions for All Students:")
print(scores)

# Matrix × Vector


# ------------------------------------------
# 12. WHY MATRICES MATTER
# ------------------------------------------
print("\nML Insight:")
print("Datasets are Matrices")
print("Rows are Vectors")
print("Predictions are Dot Products")
print("Neural Networks are Matrix Operations")