# ==========================================
# DAY 8: NumPy Foundations (ML Perspective)
# ==========================================

import numpy as np

# ----------- 1. PYTHON LIST vs NUMPY ARRAY -----------
a = [1, 2, 3]
b = [4, 5, 6]

print("List + List (concatenation ❌):", a + b)

# NumPy arrays
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print("Array + Array (element-wise ✅):", a + b)
print("Multiplication:", a * b)
print("Subtraction:", a - b)
print("Dot Product (@):", a @ b)   # core ML operation


# ----------- 2. ARRAY DIMENSIONS (1D, 2D, 3D) -----------
a = np.array([1, 2, 3])  # 1D
b = np.array([[1, 2, 3], [4, 5, 6]])  # 2D (matrix)
c = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])  # 3D

print("1D:", a)
print("2D:\n", b)
print("3D:\n", c)


# ----------- 3. ARRAY PROPERTIES (IMPORTANT IN ML) -----------
arr = np.array([[1, 2, 3], [4, 5, 6]])

print("Shape:", arr.shape)      # (rows, cols)
print("Dimensions:", arr.ndim)  # number of axes
print("Data Type:", arr.dtype)


# ----------- 4. SPECIAL ARRAYS (INITIALIZATION) -----------
print("Zeros:\n", np.zeros((3, 3)))
print("Ones:\n", np.ones((3, 3)))
print("Identity Matrix:\n", np.eye(4))


# ----------- 5. VECTOR OPERATIONS (NO LOOPS 🔥) -----------
a = np.array([1, 2, 3])

print("Add scalar:", a + 2)
print("Multiply scalar:", a * 3)
print("Power:", a ** 2)


# ----------- 6. INDEXING & SLICING -----------
arr = np.array([10, 20, 30, 40])

print("First element:", arr[0])
print("Slice [1:3]:", arr[1:3])
print("Last element:", arr[-1])

# 2D indexing
mat = np.array([[1, 2, 3],
                [4, 5, 6]])

print("Element at row1,col1:", mat[1, 1])


# ----------- 7. RESHAPING (DATA TRANSFORMATION) -----------
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8])

print("Reshape (2x4):\n", arr.reshape(2, 4))


# ----------- 8. STATISTICS (VERY IMPORTANT IN ML) -----------
arr = np.array([1,2,3,4,5,6,7,8,9,10])

print("Max:", arr.max())
print("Min:", arr.min())
print("Sum:", arr.sum())
print("Mean:", arr.mean())
print("Std Dev:", arr.std())
print("Variance:", arr.var())
print("Index of max:", arr.argmax())


# ----------- 9. MATRIX MULTIPLICATION (CORE ML 🔥) -----------
a = np.array([[1, 2],
              [3, 4]])

b = np.array([[5, 6],
              [7, 8]])

print("Matrix Dot Product:\n", np.dot(a, b))


# ----------- 10. RANDOM DATA (SIMULATING DATASETS) -----------
print("Random float:\n", np.random.rand(2, 2))
print("Random integers:\n", np.random.randint(1, 20, (2, 2)))
print("Random dataset:\n", np.random.randint(100, 1000, (3, 3)))


# ----------- 11. MINI ML-LIKE EXAMPLE 🔥 -----------

# Features (input data)
X = np.array([1, 2, 3, 4])

# True outputs
y = np.array([2, 4, 6, 8])

# Initialize weight
w = 1

# Training loop
for _ in range(5):
    y_pred = X * w              # prediction
    error = y - y_pred          # error
    w += 0.1 * np.mean(error)   # update weight

print("Trained Weight:", w)
print("Final Prediction:", X * w)