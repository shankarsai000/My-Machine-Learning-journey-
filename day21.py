# ==========================================
# DAY 21: EIGENVALUES, EIGENVECTORS & PCA
# ==========================================
import numpy as np
from sklearn.decomposition import PCA
# ------------------------------------------
# 1. MATRIX
# ------------------------------------------
A = np.array([
    [4, 2],
    [1, 3]
])
print("Matrix A")
print(A)
# ------------------------------------------
# 2. EIGENVALUES & EIGENVECTORS
# ------------------------------------------
eigenvalues, eigenvectors = np.linalg.eig(A)

print("\nEigenvalues")
print(eigenvalues)

print("\nEigenvectors")
print(eigenvectors)
# ------------------------------------------
# 3. VERIFY EQUATION
# ------------------------------------------
# A*v = λ*v

v = eigenvectors[:,0]
lam = eigenvalues[0]

left = A @ v
right = lam * v

print("\nA*v")
print(left)

print("\nλ*v")
print(right)
# ------------------------------------------
# 4. DATASET
# ------------------------------------------

X = np.array([
    [2, 3],
    [3, 5],
    [4, 7],
    [5, 9],
    [6, 11]
])

print("\nDataset")
print(X)
# ------------------------------------------
# 5. MEAN CENTERING
# ------------------------------------------
X_centered = X - np.mean(X, axis=0)

print("\nCentered Data")
print(X_centered)
# ------------------------------------------
# 6. COVARIANCE MATRIX
# ------------------------------------------

cov_matrix = np.cov(
    X_centered.T
)

print("\nCovariance Matrix")
print(cov_matrix)
# ------------------------------------------
# 7. EIGEN DECOMPOSITION
# ------------------------------------------

vals, vecs = np.linalg.eig(
    cov_matrix
)

print("\nEigenvalues")
print(vals)

print("\nEigenvectors")
print(vecs)
# ------------------------------------------
# 8. MOST IMPORTANT DIRECTION
# ------------------------------------------

principal_vector = vecs[:, np.argmax(vals)]

print("\nPrincipal Direction")
print(principal_vector)
# PCA chooses this direction
# ------------------------------------------
# 9. PCA USING SKLEARN
# ------------------------------------------

pca = PCA(n_components=1)

X_reduced = pca.fit_transform(X)

print("\nReduced Data")
print(X_reduced)
# ------------------------------------------
# 10. EXPLAINED VARIANCE
# ------------------------------------------

print("\nExplained Variance")

print(
    pca.explained_variance_ratio_
)
# Information retained
# ------------------------------------------
# 11. IMAGE COMPRESSION IDEA
# ------------------------------------------

image_pixels = np.random.randint(
    0,
    255,
    (100,100)
)

print("\nImage Shape")
print(image_pixels.shape)

# PCA compresses this
# ------------------------------------------
# 12. RECOMMENDATION SYSTEM IDEA
# ------------------------------------------

users_movies = np.array([
    [5,4,0,1],
    [4,5,1,0],
    [1,0,5,4]
])

print("\nRecommendation Matrix")
print(users_movies)

# Eigenvectors reveal patterns
# ------------------------------------------
# 13. WHY PCA WORKS
# ------------------------------------------

print("\nPCA Insight")

print(
    "Find directions with maximum information"
)

print(
    "Discard less useful directions"
)
# ------------------------------------------
# 14. ML APPLICATIONS
# ------------------------------------------

print("\nUsed In")

print("Face Recognition")
print("Image Compression")
print("Recommendation Systems")
print("Dimensionality Reduction")
print("Feature Engineering")
# ------------------------------------------
# 15. FINAL TAKEAWAY
# ------------------------------------------
print("\nToday's Learning")

print(
    "Eigenvectors reveal important directions."
)

print(
    "Eigenvalues measure importance."
)

print(
    "PCA uses both to compress data."
)