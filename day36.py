#----------day36.py----------#

import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

X, _ =make_moons(n_samples=600, noise=0.08, random_state=42)
plt.figure(figsize=(8, 5))

plt.scatter(
    X[:, 0],
    X[:, 1],
    alpha=0.7
)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Non-Spherical Data")

plt.grid(alpha=0.2)
plt.show()

kmeans = KMeans(
    n_clusters=2,
    random_state=42,
    n_init=10
)

kmeans_labels = kmeans.fit_predict(X)
plt.figure(figsize=(8, 5))

plt.scatter(
    X[:, 0],
    X[:, 1],
    c=kmeans_labels,
    alpha=0.7
)

plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    marker="X",
    s=200,
    edgecolor="black"
)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("k-Means on Non-Spherical Data")

plt.grid(alpha=0.2)
plt.show()

dbscan = DBSCAN(
    eps=0.2,
    min_samples=5
)

dbscan_labels = dbscan.fit_predict(X)
plt.figure(figsize=(8, 5))

plt.scatter(
    X[:, 0],
    X[:, 1],
    c=dbscan_labels,
    alpha=0.7
)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("DBSCAN Clustering")

plt.grid(alpha=0.2)
plt.show()

noise_mask = dbscan_labels == -1

noise_count = noise_mask.sum()

print("Number of noise points:", noise_count)

noise_percentage = (
    noise_count / len(X)
) * 100

print(
    f"Noise percentage: {noise_percentage:.2f}%"
)
plt.figure(figsize=(9, 6))

# Clustered points
clustered_mask = dbscan_labels != -1

plt.scatter(
    X[clustered_mask, 0],
    X[clustered_mask, 1],
    c=dbscan_labels[clustered_mask],
    alpha=0.7,
    label="Clustered points"
)

# Noise points
plt.scatter(
    X[noise_mask, 0],
    X[noise_mask, 1],
    marker="x",
    s=80,
    label="Noise"
)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("DBSCAN: Clusters vs Noise")

plt.legend()
plt.grid(alpha=0.2)

plt.show()
"""
============================================================
DAY 36 — DBSCAN REVISION
============================================================

DBSCAN
│
├── Density-based clustering
│
├── eps
│     └── neighborhood radius
│
├── min_samples
│     └── minimum density requirement
│
├── Core Point
│     └── enough neighbors
│
├── Border Point
│     └── near a core point
│
└── Noise
      └── does not belong to dense region


CORE IDEA

Find dense regions
       ↓
Grow clusters through density connectivity
       ↓
Identify sparse points as noise


IMPORTANT PARAMETERS

eps
→ How large is the neighborhood?

min_samples
→ How many samples are needed for sufficient density?


IF eps IS TOO SMALL

Too much noise
Clusters may fragment


IF eps IS TOO LARGE

Different clusters may merge


IF min_samples IS TOO SMALL

Weak density requirement
More points become core points


IF min_samples IS TOO LARGE

Strict density requirement
More points may become noise


ADVANTAGES

✓ No need to specify number of clusters
✓ Detects arbitrary-shaped clusters
✓ Can identify noise/outliers
✓ Useful for spatial data


LIMITATIONS

✗ Sensitive to eps
✗ Sensitive to min_samples
✗ Struggles with varying-density clusters
✗ Distance becomes problematic in high dimensions
✗ Parameter selection can be difficult


SCIKIT-LEARN

from sklearn.cluster import DBSCAN

model = DBSCAN(
    eps=0.2,
    min_samples=5
)

labels = model.fit_predict(X)


SPECIAL LABEL

-1 = noise


MEMORY TRICK

k-Means:
"Which centroid am I closest to?"

DBSCAN:
"Am I inside a sufficiently dense neighborhood?"

============================================================
"""