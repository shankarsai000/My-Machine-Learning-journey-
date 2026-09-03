#---------------------------------------day38------------------------------------#
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

X, y = make_blobs(
    n_samples=500,
    n_features=5,
    centers=3,
    cluster_std=2.0,
    random_state=42
)

X.shape

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("Original shape:", X.shape)
print("Scaled shape:", X_scaled.shape)


pca = PCA()

X_pca = pca.fit_transform(X_scaled)

print("Original shape:", X_scaled.shape)
print("PCA shape:", X_pca.shape)

explained_variance = pca.explained_variance_ratio_

print(explained_variance)

cumulative_variance = np.cumsum(
    pca.explained_variance_ratio_
)

print(cumulative_variance)

plt.figure(figsize=(8, 5))

plt.plot(
    range(1, len(explained_variance) + 1),
    cumulative_variance,
    marker="o"
)

plt.axhline(
    y=0.95,
    linestyle="--",
    label="95% variance"
)

plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Explained Variance")
plt.legend()
plt.grid(True)

plt.show()


pca_95 = PCA(n_components=0.95)

X_reduced = pca_95.fit_transform(X_scaled)

print("Original shape:", X_scaled.shape)
print("Reduced shape:", X_reduced.shape)
print(
    "Components selected:",
    pca_95.n_components_
)

pca_2d = PCA(n_components=2)

X_2d = pca_2d.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))

plt.scatter(
    X_2d[:, 0],
    X_2d[:, 1],
    c=y,
    alpha=0.7
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA: 5D → 2D")

plt.show()

# ============================================================
# DAY 38/60 — PCA (PRINCIPAL COMPONENT ANALYSIS)
# REVISION NOTES
# ============================================================

# ------------------------------------------------------------
# 1. WHAT IS PCA?
# ------------------------------------------------------------

# PCA = Principal Component Analysis
#
# PCA is an UNSUPERVISED dimensionality reduction technique.
#
# Main goal:
# Reduce the number of features while preserving as much
# important information (variance) as possible.
#
# Example:
#
# 100 features
#      ↓ PCA
# 10 principal components
#
# Instead of working with 100 original features,
# we work with 10 new features.


# ------------------------------------------------------------
# 2. WHY DO WE NEED PCA?
# ------------------------------------------------------------

# High-dimensional datasets can cause:
#
# - High computational cost
# - High memory usage
# - Difficult visualization
# - Redundant/correlated features
# - Noise
# - Curse of dimensionality
#
# PCA helps create a smaller representation of the data.


# ------------------------------------------------------------
# 3. CORE INTUITION
# ------------------------------------------------------------

# PCA finds new directions (axes) where the data has
# the greatest amount of variance.
#
# Think of PCA as ROTATING the coordinate system.
#
# Original:
#
#       Y
#       ↑
#       |      •
#       |    •
#       |  •
#       | •
#       +--------------→ X
#
# PCA finds a new axis along the direction where the
# data varies the most.
#
# PC1 → maximum variance
# PC2 → second-highest variance
# PC3 → third-highest variance
# ...


# ------------------------------------------------------------
# 4. PRINCIPAL COMPONENTS
# ------------------------------------------------------------

# Principal Components are NEW FEATURES created from
# combinations of the original features.
#
# Example:
#
# Original:
# Age
# Income
# Savings
#
# PCA:
# PC1
# PC2
# PC3
#
# PC1 is NOT simply one original feature.
#
# It is a linear combination:
#
# PC1 = w1*Age + w2*Income + w3*Savings
#
# where w1, w2, w3 are weights/loadings.


# ------------------------------------------------------------
# 5. PCA DOES NOT SELECT EXISTING FEATURES
# ------------------------------------------------------------

# Feature Selection:
#
# Original:
# Age, Income, Savings, Debt
#
# Select:
# Income, Debt
#
# The original features remain unchanged.
#
#
# PCA:
#
# Age, Income, Savings, Debt
#          ↓
#         PCA
#          ↓
# PC1, PC2
#
# PCA creates NEW features.


# ------------------------------------------------------------
# 6. VARIANCE
# ------------------------------------------------------------

# PCA assumes that directions with higher variance
# contain more useful information about the structure
# of the dataset.
#
# PC1 → maximum variance
# PC2 → maximum remaining variance
# PC3 → next maximum variance
#
# Therefore:
#
# PC1 > PC2 > PC3 > ...


# ------------------------------------------------------------
# 7. COVARIANCE
# ------------------------------------------------------------

# PCA looks at how features vary together.
#
# Positive covariance:
#
# X ↑ → Y ↑
#
# Negative covariance:
#
# X ↑ → Y ↓
#
# The covariance matrix captures these relationships.
#
# Example:
#
#         X1       X2
# X1    Var(X1)  Cov(X1,X2)
# X2    Cov(X2,X1) Var(X2)


# ------------------------------------------------------------
# 8. EIGENVECTORS AND EIGENVALUES
# ------------------------------------------------------------

# PCA uses eigenvectors and eigenvalues of the
# covariance structure.
#
# Eigenvectors:
# → Give the principal directions/axes.
#
# Eigenvalues:
# → Tell us how much variance exists along each direction.
#
#
# Think:
#
# Eigenvector → WHERE to look
# Eigenvalue  → HOW IMPORTANT that direction is
#
#
# Largest eigenvalue:
# → PC1
#
# Second largest:
# → PC2
#
# and so on.


# ------------------------------------------------------------
# 9. EXPLAINED VARIANCE
# ------------------------------------------------------------

# Explained variance tells us how much information
# each principal component captures.
#
# Example:
#
# PC1 → 60%
# PC2 → 25%
# PC3 → 10%
# PC4 → 5%
#
# Total = 100%
#
# Therefore:
#
# PC1                  → 60%
# PC1 + PC2            → 85%
# PC1 + PC2 + PC3      → 95%
# PC1 + PC2 + PC3+PC4  → 100%


# ------------------------------------------------------------
# 10. EXPLAINED VARIANCE RATIO
# ------------------------------------------------------------

# Formula:
#
# Explained Variance Ratio =
#
#             eigenvalue
#     -------------------------
#       sum of all eigenvalues
#
#
# In mathematical notation:
#
# EVR_i = λ_i / Σλ_j
#
# λ (lambda) = eigenvalue
#
# Example:
#
# Eigenvalues:
# [6, 2.5, 1, 0.5]
#
# Total = 10
#
# PC1 = 6 / 10   = 60%
# PC2 = 2.5 / 10 = 25%
# PC3 = 1 / 10   = 10%
# PC4 = 0.5 / 10 = 5%


# ------------------------------------------------------------
# 11. CUMULATIVE EXPLAINED VARIANCE
# ------------------------------------------------------------

# Cumulative explained variance tells us how much
# total variance is preserved when combining components.
#
# Example:
#
# PC1 = 60%
# PC2 = 25%
# PC3 = 10%
# PC4 = 5%
#
# Cumulative:
#
# PC1           → 60%
# PC1 + PC2     → 85%
# PC1 + PC2+3   → 95%
# PC1 + PC2+3+4 → 100%


# ------------------------------------------------------------
# 12. HOW DO WE CHOOSE NUMBER OF COMPONENTS?
# ------------------------------------------------------------

# Common methods:
#
# 1. Explained variance threshold
# 2. Scree plot
# 3. Downstream model performance
# 4. Domain requirements
#
# Example:
#
# PCA(n_components=0.95)
#
# Means:
# Keep enough components to preserve at least
# approximately 95% of the variance.


# ------------------------------------------------------------
# 13. SCREE PLOT
# ------------------------------------------------------------

# Scree plot shows how much variance is explained
# by each principal component.
#
# We look for an "elbow".
#
# Example:
#
# Variance
#    |
# 60 | ●
#    |  \
# 40 |   \
#    |    ●
# 20 |      ●
#    |        ●
#    +----------------
#       PC1 PC2 PC3 PC4
#
# After the elbow, additional components may contribute
# relatively little information.


# ------------------------------------------------------------
# 14. VERY IMPORTANT — SCALE BEFORE PCA
# ------------------------------------------------------------

# PCA is variance-based.
#
# Suppose:
#
# Age       → 18 - 80
# Income    → 20,000 - 2,000,000
#
# Income has a much larger numerical scale.
#
# It could dominate PCA.
#
# Therefore, usually:
#
# Raw Data
#    ↓
# StandardScaler
#    ↓
# PCA
#
#
# Example:
#
# from sklearn.preprocessing import StandardScaler
#
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)


# ------------------------------------------------------------
# 15. PCA WORKFLOW
# ------------------------------------------------------------

# Typical workflow:
#
# Raw Data
#    ↓
# Train/Test Split
#    ↓
# Feature Scaling
#    ↓
# PCA
#    ↓
# ML Model
#    ↓
# Evaluation
#
# IMPORTANT:
# PCA should be fitted using training data only
# when building a supervised ML system.


# ------------------------------------------------------------
# 16. PCA WITH SCIKIT-LEARN
# ------------------------------------------------------------

# from sklearn.decomposition import PCA
#
# pca = PCA(n_components=2)
#
# X_pca = pca.fit_transform(X_scaled)
#
# This converts:
#
# Original:
# (n_samples, n_features)
#
# Into:
# (n_samples, 2)


# ------------------------------------------------------------
# 17. KEEPING 95% VARIANCE
# ------------------------------------------------------------

# pca = PCA(n_components=0.95)
#
# X_reduced = pca.fit_transform(X_scaled)
#
# This automatically selects the number of components
# needed to preserve at least 95% variance.


# ------------------------------------------------------------
# 18. IMPORTANT PCA ATTRIBUTES
# ------------------------------------------------------------

# pca.components_
#
# → Directions/loadings of principal components.
#
#
# pca.explained_variance_
#
# → Amount of variance captured by each component.
#
#
# pca.explained_variance_ratio_
#
# → Proportion of total variance captured by
#   each component.
#
#
# pca.n_components_
#
# → Number of components selected.


# ------------------------------------------------------------
# 19. CUMULATIVE VARIANCE IN PYTHON
# ------------------------------------------------------------

# import numpy as np
#
# cumulative_variance = np.cumsum(
#     pca.explained_variance_ratio_
# )
#
# Example:
#
# [0.60, 0.25, 0.10, 0.05]
#
# becomes:
#
# [0.60, 0.85, 0.95, 1.00]


# ------------------------------------------------------------
# 20. PCA FOR VISUALIZATION
# ------------------------------------------------------------

# High-dimensional data can be reduced to 2 dimensions:
#
# 100 features
#     ↓
# PCA(n_components=2)
#     ↓
# 2 features
#
# Then:
#
# plt.scatter(
#     X_pca[:, 0],
#     X_pca[:, 1]
# )
#
# This allows us to visualize high-dimensional data
# in 2D.


# ------------------------------------------------------------
# 21. PCA VS CLUSTERING
# ------------------------------------------------------------

# PCA DOES NOT perform clustering.
#
# PCA:
# → Dimensionality reduction
#
# K-Means:
# → Centroid-based clustering
#
# DBSCAN:
# → Density-based clustering
#
# Hierarchical:
# → Hierarchy-based clustering
#
#
# They can also be combined:
#
# High-dimensional data
#        ↓
#       PCA
#        ↓
# Reduced dimensions
#        ↓
#     K-Means
#        ↓
#     Clusters


# ------------------------------------------------------------
# 22. PCA VS K-MEANS / DBSCAN / HIERARCHICAL
# ------------------------------------------------------------

# K-Means:
# → Finds centroids
# → Requires number of clusters
#
# DBSCAN:
# → Finds dense regions
# → Handles noise
# → Does not require number of clusters beforehand
#
# Hierarchical:
# → Builds hierarchy of clusters
# → Dendrogram
#
# PCA:
# → Finds important directions
# → Reduces dimensions
# → Does NOT create clusters


# ------------------------------------------------------------
# 23. ADVANTAGES OF PCA
# ------------------------------------------------------------

# + Reduces dimensionality
# + Reduces computational cost
# + Can remove redundant information
# + Helps visualization
# + Can reduce noise
# + Useful before clustering/modeling
# + Creates compact representations


# ------------------------------------------------------------
# 24. DISADVANTAGES / LIMITATIONS
# ------------------------------------------------------------

# - Components can be difficult to interpret
# - Information can be lost
# - Sensitive to feature scaling
# - Mainly captures linear structure
# - High variance does not always mean high predictive value
# - Can become expensive for extremely high-dimensional data
# - Not always beneficial for every dataset


# ------------------------------------------------------------
# 25. IMPORTANT ASSUMPTION
# ------------------------------------------------------------

# PCA is fundamentally a LINEAR dimensionality reduction
# technique.
#
# If the data has complex nonlinear structure:
#
# PCA
#  ↓
# may not capture it well.
#
# This motivates techniques such as:
#
# t-SNE
# UMAP
#
# which are the next topics in the roadmap.


# ------------------------------------------------------------
# 26. COMMON BEGINNER MISTAKES
# ------------------------------------------------------------

# Mistake 1:
# Applying PCA without scaling when features have
# very different scales.
#
#
# Mistake 2:
# Thinking PC1 is the first original feature.
#
# PC1 is a combination of original features.
#
#
# Mistake 3:
# Always choosing 2 components.
#
# 2 components are useful for visualization,
# but may lose too much information.
#
#
# Mistake 4:
# Assuming PCA always improves model performance.
#
# It can improve, hurt, or have little effect.
#
#
# Mistake 5:
# Fitting PCA on the entire dataset before train/test split.
#
# This can cause data leakage.
#
# Correct:
#
# pca.fit(X_train)
# X_train_pca = pca.transform(X_train)
# X_test_pca = pca.transform(X_test)


# ------------------------------------------------------------
# 27. DATA LEAKAGE WITH PCA
# ------------------------------------------------------------

# WRONG:
#
# X_scaled = scaler.fit_transform(X)
# X_pca = pca.fit_transform(X_scaled)
# train_test_split(X_pca, y)
#
# PCA has already learned from the entire dataset.
#
#
# BETTER:
#
# Split first:
#
# X_train, X_test, y_train, y_test = train_test_split(...)
#
# scaler.fit(X_train)
# X_train_scaled = scaler.transform(X_train)
# X_test_scaled = scaler.transform(X_test)
#
# pca.fit(X_train_scaled)
# X_train_pca = pca.transform(X_train_scaled)
# X_test_pca = pca.transform(X_test_scaled)
#
# Later:
# Put scaling + PCA inside a Pipeline.


# ------------------------------------------------------------
# 28. PCA MATHEMATICAL SUMMARY
# ------------------------------------------------------------

# Given feature matrix:
#
# X ∈ R^(n × d)
#
# PCA finds directions:
#
# w1, w2, w3, ...
#
# such that:
#
# PC1 = Xw1
# PC2 = Xw2
# PC3 = Xw3
#
# with maximum variance captured in descending order.
#
#
# Eigenvectors → principal directions
# Eigenvalues  → variance of those directions


# ------------------------------------------------------------
# 29. PCA AND SVD
# ------------------------------------------------------------

# PCA can be computed using:
#
# Eigenvalue decomposition
# OR
# Singular Value Decomposition (SVD)
#
# In practical ML libraries, SVD-based implementations
# are commonly used.
#
# Scikit-Learn's PCA provides different SVD solvers,
# including randomized SVD for suitable large datasets.


# ------------------------------------------------------------
# 30. REAL-WORLD APPLICATIONS
# ------------------------------------------------------------

# PCA can be used for:
#
# - High-dimensional feature reduction
# - Data visualization
# - Image compression
# - Customer analytics
# - Genomics
# - Document representations
# - Exploratory data analysis
# - Preprocessing before clustering
#
# Example:
#
# 1000 features
#       ↓
#      PCA
#       ↓
# 50 components
#       ↓
# ML model


# ------------------------------------------------------------
# 31. INTERVIEW QUICK ANSWERS
# ------------------------------------------------------------

# Q: What is PCA?
#
# A:
# PCA is an unsupervised dimensionality reduction technique
# that transforms original features into orthogonal principal
# components ordered by the amount of variance they explain.
#
#
# Q: What is PC1?
#
# A:
# The direction that captures the maximum variance.
#
#
# Q: What do eigenvectors represent?
#
# A:
# Principal directions/axes of the data.
#
#
# Q: What do eigenvalues represent?
#
# A:
# Variance associated with those directions.
#
#
# Q: Why standardize before PCA?
#
# A:
# Because PCA depends on variance and large-scale features
# can dominate the principal components.
#
#
# Q: PCA vs feature selection?
#
# A:
# Feature selection keeps original features.
# PCA creates new features as linear combinations.
#
#
# Q: Is PCA supervised?
#
# A:
# No. Standard PCA does not use target labels.
#
#
# Q: How do you choose n_components?
#
# A:
# Using explained variance, scree plot, downstream performance,
# and domain requirements.


# ------------------------------------------------------------
# 32. ONE-MINUTE REVISION
# ------------------------------------------------------------

# PCA = Dimensionality Reduction
#
# PCA finds directions of maximum variance.
#
# PC1 = maximum variance
# PC2 = second maximum variance
# PC3 = third maximum variance
#
# Eigenvectors = directions
# Eigenvalues = importance/variance
#
# Explained variance = information captured
#
# Scree plot = helps choose components
#
# Scale features before PCA when scales differ.
#
# PCA creates NEW features.
#
# PCA ≠ clustering.
#
# PCA is mainly LINEAR.
#
# Always avoid fitting PCA on test data.


# ------------------------------------------------------------
# 33. MEMORY TRICK
# ------------------------------------------------------------

# Remember:
#
#       "PCA = Preserve the most variance with fewer axes"
#
#
# P → Principal
# C → Components
# A → Analysis
#
#
# PC1 → Most important direction
# PC2 → Next important direction
# PC3 → Next...
#
# Fewer dimensions + maximum retained variance
# = The essence of PCA.