#--------------------------day 37--------------------------#

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

from scipy.cluster.hierarchy import dendrogram, linkage

X, _ = make_blobs(
    n_samples=300,
    centers=3,
    cluster_std=1.2,
    random_state=42
)

plt.figure(figsize=(8, 5))

plt.scatter(
    X[:, 0],
    X[:, 1],
    alpha=0.7
)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Raw Data")

plt.grid(alpha=0.2)
plt.show()

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

linkage_matrix = linkage(
    X_scaled,
    method="ward"
)
plt.figure(figsize=(12, 6))

dendrogram(
    linkage_matrix,
    truncate_mode="lastp",
    p=20
)

plt.xlabel("Samples / Clusters")
plt.ylabel("Distance")
plt.title("Hierarchical Clustering Dendrogram")

plt.show()

model = AgglomerativeClustering(
    n_clusters=3,
    linkage="ward"
)

labels = model.fit_predict(X_scaled)


plt.figure(figsize=(8, 5))

plt.scatter(
    X_scaled[:, 0],
    X_scaled[:, 1],
    c=labels,
    alpha=0.7
)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Agglomerative Clustering")

plt.grid(alpha=0.2)
plt.show()

score = silhouette_score(
    X_scaled,
    labels
)

print(f"Silhouette Score: {score:.4f}")


"""
============================================================
DAY 37 — HIERARCHICAL CLUSTERING
============================================================

CORE IDEA

Build a hierarchy of clusters.

Agglomerative = bottom-up.

Start:
A B C D E

Then:
AB C DE

Then:
ABC DE

Then:
ABCDE


DENDROGRAM
│
└── Tree showing how clusters merge.


IMPORTANT CONCEPT

HEIGHT OF MERGE
→ distance / dissimilarity at which clusters were merged.


LINKAGE METHODS

Single
→ minimum pairwise distance
→ can cause chaining

Complete
→ maximum pairwise distance
→ tends toward compact clusters

Average
→ average pairwise distance

Ward
→ minimizes increase in within-cluster variance


ADVANTAGES

✓ Reveals hierarchical structure
✓ Dendrogram provides visual interpretation
✓ Final K can be selected after examining hierarchy
✓ Useful for exploratory analysis
✓ Doesn't require centroid representation


LIMITATIONS

✗ Can be computationally expensive
✗ Less suitable for very large datasets
✗ Sensitive to distance metric
✗ Results depend on linkage method
✗ Early merges generally cannot be undone


K-MEANS

Centroid-based
Choose K
Fast/scalable
No hierarchy


DBSCAN

Density-based
No explicit K
Detects noise
Handles arbitrary shapes


HIERARCHICAL

Hierarchy-based
Build dendrogram
Multiple levels of grouping
Useful for structure exploration


MEMORY TRICK

K-Means:
"Find centers."

DBSCAN:
"Find density."

Hierarchical:
"Build relationships."


============================================================

             UNSUPERVISED LEARNING
                     │
       ┌─────────────┼─────────────┐
       ▼             ▼             ▼
    K-MEANS        DBSCAN     HIERARCHICAL
       │             │             │
    Centroid       Density       Hierarchy
       │             │             │
    Choose K       eps/k        Dendrogram
       │             │             │
 Compact groups   Noise        Nested groups
 
 
 """