# ==========================================
# NUMPY MASTER CELL (0 → ADVANCED + ML)
# ==========================================

import numpy as np

# ----------- 1. LIST vs NUMPY -----------
a = [1, 2, 3]
b = [4, 5, 6]
print("List:", a + b)  # concatenation ❌

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print("NumPy:", a + b)  # element-wise ✅


# ----------- 2. ARRAY CREATION -----------
arr1 = np.array([1, 2, 3])
arr2 = np.array([[1, 2, 3], [4, 5, 6]])
arr3 = np.array([[[1,2],[3,4]],[[5,6],[7,8]]])
print("1D:", arr1)
print("2D:\n", arr2)
print("3D:\n", arr3)


# ----------- 3. PROPERTIES -----------
print("Shape:", arr2.shape)
print("Dimensions:", arr2.ndim)
print("Type:", arr2.dtype)


# ----------- 4. SPECIAL ARRAYS -----------
print("Zeros:\n", np.zeros((2,2)))
print("Ones:\n", np.ones((2,2)))
print("Identity:\n", np.eye(3))


# ----------- 5. VECTOR OPERATIONS -----------
a = np.array([1,2,3])
print("Add:", a + 2)
print("Multiply:", a * 3)
print("Power:", a ** 2)


# ----------- 6. INDEXING & SLICING -----------
arr = np.array([10,20,30,40])
print(arr[0], arr[1:3], arr[-1])

mat = np.array([[1,2,3],[4,5,6]])
print("2D index:", mat[1,1])


# ----------- 7. RESHAPING -----------
arr = np.arange(1,9)
print("Reshape 2x4:\n", arr.reshape(2,4))


# ----------- 8. STATISTICS -----------
arr = np.arange(1,11)
print("Mean:", arr.mean(), "Std:", arr.std(), "Max:", arr.max())


# ----------- 9. MATRIX MULTIPLICATION -----------
a = np.array([[1,2],[3,4]])
b = np.array([[5,6],[7,8]])
print("Dot:\n", np.dot(a,b))


# ----------- 10. RANDOM DATA -----------
print("Random:\n", np.random.randint(1,10,(2,2)))


# ----------- 11. BROADCASTING -----------
a = np.array([1,2,3])
b = np.array([[10],[20],[30]])
print("Broadcast:\n", a + b)


# ----------- 12. AXIS OPERATIONS -----------
arr = np.array([[1,2,3],[4,5,6]])
print("Sum axis0:", np.sum(arr,axis=0))
print("Sum axis1:", np.sum(arr,axis=1))


# ----------- 13. BOOLEAN MASKING -----------
data = np.array([10,20,30,40,50])
print("Filtered:", data[data > 25])


# ----------- 14. ADVANCED INDEXING -----------
arr = np.array([100,200,300,400])
print("Selected:", arr[[0,2]])


# ----------- 15. FULL ML PREPROCESS PIPELINE 🔥 -----------

# Raw dataset (age, salary)
X = np.array([
    [25,50000],
    [30,60000],
    [22,250000],   # outlier
    [35,80000]
])

# 1. Remove outliers
X = X[X[:,1] < 100000]

# 2. Normalize
X = (X - np.mean(X,axis=0)) / np.std(X,axis=0)

# 3. Add bias
bias = np.ones((X.shape[0],1))
X = np.hstack((bias, X))

# 4. Prediction (simple model)
w = np.array([1,2,3])
y_pred = X @ w

print("Final ML Output:\n", y_pred)