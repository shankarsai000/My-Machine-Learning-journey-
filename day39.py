# ============================================================
# DAY 39/60 — t-SNE & UMAP
# Complete One-Cell Experiment
# MNIST: PCA vs t-SNE vs UMAP
# ============================================================

# If UMAP is not installed, uncomment the next line:
# %pip install umap-learn

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# ============================================================
# 1. LOAD MNIST DATASET
# ============================================================

print("=" * 60)
print("DAY 39 — t-SNE & UMAP")
print("=" * 60)

print("\nLoading MNIST dataset...")

mnist = fetch_openml(
    "mnist_784",
    version=1,
    as_frame=False
)

X = mnist.data.astype(np.float32)
y = mnist.target.astype(int)

print(f"Original dataset shape: {X.shape}")
print(f"Number of samples: {X.shape[0]:,}")
print(f"Number of features: {X.shape[1]}")


# ============================================================
# 2. NORMALIZE PIXEL VALUES
# ============================================================

# MNIST pixel values range from 0 to 255.
# Convert them to the range 0 to 1.

X = X / 255.0

print("\nPixel values normalized.")
print(f"Minimum value: {X.min():.2f}")
print(f"Maximum value: {X.max():.2f}")


# ============================================================
# 3. USE A SUBSET
# ============================================================

# t-SNE can be computationally expensive.
# Therefore, we use 5,000 samples for this experiment.

N_SAMPLES = 5000

X_small = X[:N_SAMPLES]
y_small = y[:N_SAMPLES]

print(f"\nUsing subset: {X_small.shape}")


# ============================================================
# 4. PCA — LINEAR DIMENSIONALITY REDUCTION
# ============================================================

print("\n" + "=" * 60)
print("1. PCA")
print("=" * 60)

# Reduce 784 dimensions to 2 dimensions.

pca = PCA(
    n_components=2,
    random_state=42
)

X_pca = pca.fit_transform(X_small)

pca_variance = pca.explained_variance_ratio_
pca_total_variance = pca_variance.sum()

print(f"PCA output shape: {X_pca.shape}")

print("\nPCA explained variance:")
for i, variance in enumerate(pca_variance, start=1):
    print(f"PC{i}: {variance:.2%}")

print(f"\nTotal variance retained: {pca_total_variance:.2%}")


# ============================================================
# 5. t-SNE — NONLINEAR DIMENSIONALITY REDUCTION
# ============================================================

print("\n" + "=" * 60)
print("2. t-SNE")
print("=" * 60)

# First reduce 784 → 50 dimensions using PCA.
# This makes t-SNE faster and removes some redundancy.

pca_50 = PCA(
    n_components=50,
    random_state=42
)

X_pca_50 = pca_50.fit_transform(X_small)

print(f"PCA preprocessing shape: {X_pca_50.shape}")

# Apply t-SNE.

tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate="auto",
    init="pca",
    random_state=42
)

X_tsne = tsne.fit_transform(X_pca_50)

print(f"t-SNE output shape: {X_tsne.shape}")
print("t-SNE completed.")


# ============================================================
# 6. UMAP — NONLINEAR DIMENSIONALITY REDUCTION
# ============================================================

print("\n" + "=" * 60)
print("3. UMAP")
print("=" * 60)

try:
    import umap

    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        n_components=2,
        random_state=42
    )

    X_umap = reducer.fit_transform(X_small)

    print(f"UMAP output shape: {X_umap.shape}")
    print("UMAP completed.")

    umap_available = True

except ImportError:
    print("\nUMAP is not installed.")
    print("Run:")
    print("%pip install umap-learn")
    print("Then run this cell again.")

    umap_available = False


# ============================================================
# 7. VISUALIZE ORIGINAL SAMPLE
# ============================================================

print("\n" + "=" * 60)
print("Example MNIST Digits")
print("=" * 60)

fig, axes = plt.subplots(2, 5, figsize=(12, 5))

for i, ax in enumerate(axes.ravel()):
    ax.imshow(
        X_small[i].reshape(28, 28),
        cmap="gray"
    )
    ax.set_title(f"Digit: {y_small[i]}")
    ax.axis("off")

plt.suptitle("Sample MNIST Images")
plt.tight_layout()
plt.show()


# ============================================================
# 8. PCA VISUALIZATION
# ============================================================

plt.figure(figsize=(10, 8))

scatter = plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=y_small,
    s=8,
    alpha=0.7
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("MNIST — PCA (784 → 2 Dimensions)")
plt.colorbar(
    scatter,
    label="Digit"
)
plt.grid(alpha=0.2)

plt.show()


# ============================================================
# 9. t-SNE VISUALIZATION
# ============================================================

plt.figure(figsize=(10, 8))

scatter = plt.scatter(
    X_tsne[:, 0],
    X_tsne[:, 1],
    c=y_small,
    s=8,
    alpha=0.7
)

plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.title("MNIST — t-SNE (784 → 50 → 2)")
plt.colorbar(
    scatter,
    label="Digit"
)
plt.grid(alpha=0.2)

plt.show()


# ============================================================
# 10. UMAP VISUALIZATION
# ============================================================

if umap_available:

    plt.figure(figsize=(10, 8))

    scatter = plt.scatter(
        X_umap[:, 0],
        X_umap[:, 1],
        c=y_small,
        s=8,
        alpha=0.7
    )

    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.title("MNIST — UMAP (784 → 2)")
    plt.colorbar(
        scatter,
        label="Digit"
    )
    plt.grid(alpha=0.2)

    plt.show()


# ============================================================
# 11. COMPARISON
# ============================================================

print("\n" + "=" * 60)
print("COMPARISON")
print("=" * 60)

print("""
PCA:
    - Linear dimensionality reduction
    - Finds directions of maximum variance
    - Fast
    - Useful for preprocessing
    - Easy to transform new data

t-SNE:
    - Nonlinear dimensionality reduction
    - Focuses heavily on local neighborhoods
    - Mainly used for visualization
    - Can be computationally expensive
    - Sensitive to parameters such as perplexity

UMAP:
    - Nonlinear dimensionality reduction
    - Preserves neighborhood/manifold structure
    - Usually faster than t-SNE on larger datasets
    - Useful for visualization
    - n_neighbors controls local vs broader structure
""")


# ============================================================
# 12. FINAL SHAPES
# ============================================================

print("=" * 60)
print("DIMENSIONALITY REDUCTION SUMMARY")
print("=" * 60)

print(f"""
Original:
    {X_small.shape[1]} dimensions

PCA:
    {X_pca.shape[1]} dimensions

t-SNE:
    {X_tsne.shape[1]} dimensions

UMAP:
    {X_umap.shape[1] if umap_available else "Not available"} dimensions
""")


# ============================================================
# 13. IMPORTANT LESSONS
# ============================================================

print("=" * 60)
print("DAY 39 REVISION")
print("=" * 60)

print("""
1. PCA is LINEAR.
2. t-SNE is NONLINEAR.
3. UMAP is NONLINEAR.
4. PCA focuses on VARIANCE.
5. t-SNE focuses strongly on LOCAL NEIGHBORHOODS.
6. UMAP models LOCAL + broader MANIFOLD STRUCTURE.
7. t-SNE is mainly used for VISUALIZATION.
8. UMAP is often more scalable than t-SNE.
9. A beautiful 2D plot does NOT automatically mean
   the representation is better.
10. Labels are NOT used by PCA, t-SNE, or UMAP here.
    Labels are only used to color the visualization.
11. PCA can be useful before t-SNE:
       784 → 50 → 2
12. Do not blindly interpret distances in a t-SNE plot
    as exact distances from the original feature space.
""")


# ============================================================
# 14. INTERVIEW QUICK CHECK
# ============================================================

print("=" * 60)
print("INTERVIEW QUICK CHECK")
print("=" * 60)

print("""
Q1. What is t-SNE?
→ A nonlinear dimensionality reduction technique mainly
  used for visualizing high-dimensional data.

Q2. What does t-SNE preserve?
→ Primarily local neighborhood relationships.

Q3. What is perplexity?
→ A parameter controlling the effective neighborhood size.

Q4. What is UMAP?
→ Uniform Manifold Approximation and Projection.

Q5. PCA vs t-SNE?
→ PCA is linear and variance-based.
  t-SNE is nonlinear and neighborhood-focused.

Q6. Why use PCA before t-SNE?
→ To reduce dimensionality and redundancy before the
  computationally expensive nonlinear embedding.

Q7. Are labels required?
→ No. These are unsupervised techniques.

Q8. Can t-SNE plots be interpreted as exact geometry?
→ No. The 2D visualization can distort global distances
  and relationships.
""")


# ============================================================
# 15. DAY 39 COMPLETE
# ============================================================

print("=" * 60)
print("✅ DAY 39 COMPLETE")
print("=" * 60)

print("""
Next:
    DAY 40 → Scikit-Learn Pipelines

Upcoming workflow:

    Preprocessing
         ↓
    Feature Engineering
         ↓
    PCA / Transformation
         ↓
    ML Model
         ↓
    Evaluation

The next step is turning individual ML steps into
a clean, reusable and leakage-safe pipeline.
""")

# t-SNE = nonlinear dimensionality reduction
# UMAP = nonlinear dimensionality reduction
#
# Main purpose:
# → Visualization of high-dimensional data
#
# PCA:
# → Linear
# → Variance-based
#
# t-SNE:
# → Local neighborhood focused
# → Uses Student's t-distribution
# → Perplexity controls neighborhood scale
# → Mainly visualization
#
# UMAP:
# → Manifold/neighborhood based
# → n_neighbors controls neighborhood scale
# → min_dist controls compactness
# → Generally scales better than t-SNE
#
# Both:
# → Usually produce 2D/3D visualizations
# → Don't require labels
# → Can reveal hidden structure
# → Should NOT be treated as perfect representations
#
# Useful workflow:
#
# High-dimensional data
#       ↓
#      PCA
#       ↓
#   30–50 dimensions
#       ↓
#    t-SNE/UMAP
#       ↓
#      2D
#       ↓
# Visualization